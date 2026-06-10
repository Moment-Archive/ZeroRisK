import torch
import torch.utils.data as data
import torchvision.transforms.functional as FT
import numpy as np
import cv2
import os
import glob
import argparse
import json
import csv
from tqdm import tqdm
from PIL import Image
import math
import shutil

from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from utils.box import non_max_suppression, box_iou, opacity
from utils.flops import get_info


# Default black list of class IDs (1-indexed) to exclude from drawing (same as ava_eval.py / custom_infer.py)
DEFAULT_BLACK_LIST = [2, 4, 7, 9, 13, 16, 18, 19, 21, 22, 23, 24, 25, 29, 31, 32, 33, 35, 36, 39, 40, 41, 42, 44, 45, 46, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]

class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip):
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

class VideoFramesDataset(data.Dataset):
    def __init__(self, frame_paths, keyframe_indices, clip_length, sampling_rate, transform):
        self.frame_paths = frame_paths
        self.keyframe_indices = keyframe_indices
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.transform = transform

    def __len__(self):
        return len(self.keyframe_indices)

    def __getitem__(self, index):
        key_idx = self.keyframe_indices[index]
        clip = []
        for i in reversed(range(self.clip_length)):
            cur_idx = key_idx - i * self.sampling_rate
            if cur_idx < 1:
                cur_idx = 1
            # frame_paths is 0-indexed, frame index is 1-based
            frame_path = self.frame_paths[cur_idx - 1]
            img = Image.open(frame_path).convert('RGB')
            clip.append(self.transform(img))
        
        # Stack to shape [C, T, H, W]
        clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3).contiguous()
        return clip, key_idx

def draw_box_with_labels(img, box, labels_with_scores, is_accident_box=False):
    """Draw one bbox with its top action labels inside the box (top-left corner)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

    color = (0, 60, 220) if is_accident_box else (50, 180, 50)
    thickness = 3 if is_accident_box else 2

    cv2.rectangle(img, (px1, py1), (px2, py2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thick = 1
    line_h = 17

    n_lines = len(labels_with_scores)
    text_block_h = line_h * n_lines + 4
    text_block_w = min(px2 - px1, 230)  # don't exceed box width

    # Draw text inside the box at top-left; if box is too short, draw above box
    box_h = py2 - py1
    if box_h >= text_block_h + 4:
        # Inside the box
        bg_y1 = py1 + 2
        bg_y2 = py1 + 2 + text_block_h
        text_start_y = py1 + 2 + line_h - 3
    else:
        # Above the box (fallback)
        bg_y1 = max(0, py1 - text_block_h - 2)
        bg_y2 = py1
        text_start_y = bg_y1 + line_h - 3

    bg_x2 = min(W, px1 + text_block_w)

    overlay = img.copy()
    cv2.rectangle(overlay, (px1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    for i, (name, score) in enumerate(labels_with_scores):
        text = f"{name} {score:.2f}"
        ty = text_start_y + line_h * i
        cv2.putText(img, text, (px1 + 3, ty), font, font_scale,
                    (0, 255, 255), text_thick, cv2.LINE_AA)

def custom_frame_infer(config, frames_dir, output_dir, conf_threshold=0.3, top_k=3, 
                       sample_rate=30, batch_size=8, make_video=False, video_fps=2, 
                       use_blacklist=True, onnx_path=None):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if onnx_path:
        import onnxruntime
        print(f"Loading ONNX model from {onnx_path}...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        print("ONNX model loaded successfully.")
    else:
        # Build model
        model = build_yowov3(config)
        get_info(config, model)
        model.to(device)
        model.eval()
        if device == "cuda":
            model = model.half()

    mapping = config['idx2name']
    clip_length = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size = config['img_size']

    black_list = DEFAULT_BLACK_LIST if use_blacklist else []

    # Get list of video subfolders (find any directory containing at least one .jpg file)
    video_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        if any(f.lower().endswith('.jpg') for f in files):
            video_dirs.append(root)
    video_dirs = sorted(video_dirs)
    
    if not video_dirs:
        print(f"No video subfolders containing .jpg files found in {frames_dir}")
        return

    print(f"Found {len(video_dirs)} video folders to process.")
    os.makedirs(output_dir, exist_ok=True)

    transform = live_transform(img_size)

    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        print(f"\nProcessing video: {video_name}")

        # Determine output directories to preserve the relative path structure under frames_dir
        rel_path = os.path.relpath(video_dir, frames_dir)
        rel_parent = os.path.dirname(rel_path)

        if rel_parent and rel_parent != ".":
            img_out_dir = os.path.join(output_dir, rel_parent, "images", video_name)
            lbl_out_dir = os.path.join(output_dir, rel_parent, "labels")
            vid_out_dir = os.path.join(output_dir, rel_parent, "videos")
        else:
            img_out_dir = os.path.join(output_dir, "images", video_name)
            lbl_out_dir = os.path.join(output_dir, "labels")
            vid_out_dir = os.path.join(output_dir, "videos")

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        if make_video:
            os.makedirs(vid_out_dir, exist_ok=True)

        # Find all JPG frames
        frame_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        num_frames = len(frame_paths)
        if num_frames == 0:
            print(f"  No frame images found in {video_dir}, skipping.")
            continue

        print(f"  Found {num_frames} frames.")

        # Determine keyframe indices (1-based index)
        # We start at clip_length and step by sample_rate
        keyframe_indices = list(range(clip_length, num_frames + 1, sample_rate))
        if not keyframe_indices:
            # If the video is too short but has at least 1 frame, run on the last frame
            keyframe_indices = [num_frames]

        # Dataset & Dataloader
        dataset = VideoFramesDataset(frame_paths, keyframe_indices, clip_length, sampling_rate, transform)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

        # Store predictions: key_idx -> list of {'box': [x1, y1, x2, y2], 'actions': [(class_name, score)]}
        predictions = {}

        with torch.no_grad():
            for clips, key_idxs in tqdm(dataloader, desc=f"  Inference ({video_name})"):
                if onnx_path:
                    clips_np = clips.numpy()
                    input_name = ort_session.get_inputs()[0].name
                    ort_inputs = {input_name: clips_np}
                    ort_outputs = ort_session.run(None, ort_inputs)
                    outputs = torch.from_numpy(ort_outputs[0]).to(device)
                else:
                    clips = clips.to(device)
                    if device == "cuda":
                        clips = clips.half()
                    outputs = model(clips)  # shape [B, 4 + num_classes, num_anchors]
                    outputs = outputs.float()
                
                # Apply NMS per item in batch
                # non_max_suppression expects [B, 4 + num_classes, num_anchors]
                # returns list of tensors of shape [num_dets, 6] (x1, y1, x2, y2, score, class_id)
                nms_outputs = non_max_suppression(outputs, conf_threshold=conf_threshold, iou_threshold=0.5)

                for idx, key_idx in enumerate(key_idxs):
                    key_idx = key_idx.item()
                    dets = nms_outputs[idx]

                    if dets is None or dets.size(0) == 0:
                        predictions[key_idx] = []
                        continue

                    # Group predictions by rounded bounding boxes to combine multiple actions for the same person
                    grouped_dets = {}
                    for det in dets:
                        x1, y1, x2, y2, score, label = det.tolist()
                        label_1indexed = int(label) + 1
                        
                        if use_blacklist and (label_1indexed in black_list):
                            continue

                        if score < conf_threshold:
                            continue

                        # Round box coordinates to group close boxes
                        box_key = (round(x1 / img_size, 2), round(y1 / img_size, 2), 
                                   round(x2 / img_size, 2), round(y2 / img_size, 2))
                        
                        if box_key not in grouped_dets:
                            grouped_dets[box_key] = []
                        grouped_dets[box_key].append((label_1indexed, score))

                    # Format predictions for this keyframe
                    predictions[key_idx] = []
                    for box_key, actions in grouped_dets.items():
                        # Sort actions descending by score
                        actions.sort(key=lambda x: x[1], reverse=True)
                        predictions[key_idx].append({
                            'box': list(box_key),  # relative coordinates (0 to 1)
                            'actions': actions
                        })

        # Save annotated frames
        out_img_dir = img_out_dir

        print(f"  Saving annotated frames to {out_img_dir}")
        # Helper function for drawing only box
        def draw_only_box(img_canvas, box_coords, box_color, box_thickness=2):
            H_c, W_c = img_canvas.shape[:2]
            bx1, by1, bx2, by2 = box_coords
            pbx1, pby1, pbx2, pby2 = int(bx1 * W_c), int(by1 * H_c), int(bx2 * W_c), int(by2 * H_c)
            cv2.rectangle(img_canvas, (pbx1, pby1), (pbx2, pby2), box_color, box_thickness)

        for key_idx in sorted(predictions.keys()):
            out_path = os.path.join(out_img_dir, f"{video_name}_t{key_idx:06d}.jpg")
            frame_path = frame_paths[key_idx - 1]

            dets = predictions[key_idx]
            # Filter detections that have at least one valid action
            valid_dets = [d for d in dets if d['actions']]
            if not valid_dets:
                shutil.copy(frame_path, out_path)
                continue

            # Read original keyframe
            img = cv2.imread(frame_path)
            if img is None:
                continue

            # Sort detections by the score of their top action in descending order
            valid_dets.sort(key=lambda x: x['actions'][0][1], reverse=True)
            # Draw only the single best bounding box
            best_det = valid_dets[0]
            color = (0, 0, 255)  # Red in BGR
            draw_only_box(img, best_det['box'], color, box_thickness=2)

            cv2.imwrite(out_path, img)

        # ── Save per-video JSON ──────────────────────────────────────────────
        json_records = []
        for key_idx in sorted(predictions.keys()):
            frame_filename = os.path.basename(frame_paths[key_idx - 1])
            dets = predictions[key_idx]
            detections_out = []
            for det in dets:
                kept = det['actions'][:top_k]
                if not kept:
                    continue
                detections_out.append({
                    'box': {'x1': det['box'][0], 'y1': det['box'][1],
                            'x2': det['box'][2], 'y2': det['box'][3]},
                    'actions': [
                        {'class_id': aid, 'class_name': mapping.get(aid - 1, f'id_{aid}'),
                         'score': round(sc, 4)}
                        for aid, sc in kept
                    ]
                })
            json_records.append({
                'frame_idx': key_idx,
                'frame_file': frame_filename,
                'detections': detections_out
            })

        json_path = os.path.join(lbl_out_dir, f"{video_name}.json")
        with open(json_path, 'w') as jf:
            json.dump({'video': video_name, 'frames': json_records}, jf, indent=2)
        print(f"  Saved JSON  → {json_path}")


        # Assemble video if requested
        if make_video:
            out_vid_path = os.path.join(vid_out_dir, f"{video_name}.mp4")
            print(f"  Compiling video to {out_vid_path}")
            annotated_images = sorted(glob.glob(os.path.join(out_img_dir, '*.jpg')))
            if annotated_images:
                first = cv2.imread(annotated_images[0])
                H, W = first.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_vid_path, fourcc, video_fps, (W, H))

                for img_path in annotated_images:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    if (frame.shape[0], frame.shape[1]) != (H, W):
                        frame = cv2.resize(frame, (W, H))
                    writer.write(frame)

                writer.release()
                print(f"  Video created: {out_vid_path}")

    print(f"\n[DONE] Inference complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOWOv3 Custom Video Frame Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to YOWOv3 config (e.g. config/cf/ava_i3d.yaml)')
    parser.add_argument('--frames_dir', type=str, default='/data2/cache/smart_data/frames', help='Root directory containing folders of frames')
    parser.add_argument('--output_dir', type=str, default='/data2/cache/smart_data/viz_custom', help='Directory to save outputs')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--top_k', type=int, default=2, help='Max actions to show per bounding box')
    parser.add_argument('--sample_rate', type=int, default=10, help='Run inference every N frames (e.g. 30 matches AVA 1Hz)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--make_video', action='store_true', help='Assemble annotated frames into video')
    parser.add_argument('--video_fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--no_blacklist', action='store_true', help='Do not filter out background action classes')

    args = parser.parse_args()

    config = build_config(args.config)
    custom_frame_infer(
        config=config,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        top_k=args.top_k,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        make_video=args.make_video,
        video_fps=args.video_fps,
        use_blacklist=not args.no_blacklist
    )
