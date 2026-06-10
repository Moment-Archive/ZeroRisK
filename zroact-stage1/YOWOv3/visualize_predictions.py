"""
Visualize YOWOv3 predicted bounding boxes + action labels on keyframes.

Inputs
------
    --predictions   : the model's output CSV (e.g. ava_predicted_i3d.csv)
                      Format: video_id, timestamp, x1, y1, x2, y2, action_id(1-80), score
    --frames_dir    : root frames dir (e.g. /data2/cache/smart_data/frames)
    --output_dir    : where annotated images / videos go

Outputs
-------
    output_dir/
      images/
        {video_id}/
          {video_id}_t{timestamp}.jpg     ← annotated keyframe
      videos/
        {video_id}.mp4                    ← (optional) keyframes assembled

Mapping notes
-------------
  - Predicted action_id is 1-indexed (1..80), matching ava_action_list pbtxt.
  - YOWOv3 yaml's idx2name is 0-indexed (0..79).
    → label_name = IDX2NAME[action_id - 1]
  - Keyframe filename: {video_id}_NNNNNN.jpg
    where NNNNNN = (timestamp - 900) * 30 + 1   (matches load_data.py)

Usage
-----
    python visualize_predictions.py \\
        --predictions /data2/cache/YOWOv3/ava_result/ava_predicted_i3d.csv \\
        --frames_dir  /data2/cache/smart_data/frames \\
        --output_dir  /data2/cache/smart_data/viz_i3d \\
        --score_threshold 0.3 \\
        --top_k 3 \\
        --make_video
"""

import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import cv2


# ===== AVA 80 action classes (from yaml idx2name, 0-indexed) =====
IDX2NAME = {
    0: "bend/bow", 1: "crawl", 2: "crouch/kneel", 3: "dance", 4: "fall down",
    5: "get up", 6: "jump/leap", 7: "lie/sleep", 8: "martial art", 9: "run/jog",
    10: "sit", 11: "stand", 12: "swim", 13: "walk", 14: "answer phone",
    15: "brush teeth", 16: "carry/hold", 17: "catch", 18: "chop", 19: "climb",
    20: "clink glass", 21: "close", 22: "cook", 23: "cut", 24: "dig",
    25: "dress", 26: "drink", 27: "drive", 28: "eat", 29: "enter",
    30: "exit", 31: "extract", 32: "fishing", 33: "hit (object)",
    34: "kick (object)", 35: "lift/pick up", 36: "listen", 37: "open",
    38: "paint", 39: "play board game", 40: "play instrument",
    41: "play with pets", 42: "point to", 43: "press", 44: "pull",
    45: "push (object)", 46: "put down", 47: "read", 48: "ride",
    49: "row boat", 50: "sail boat", 51: "shoot", 52: "shovel",
    53: "smoke", 54: "stir", 55: "take a photo", 56: "text on cellphone",
    57: "throw", 58: "touch", 59: "turn", 60: "watch (TV)",
    61: "work on computer", 62: "write", 63: "fight/hit (person)",
    64: "give/serve", 65: "grab (person)", 66: "hand clap", 67: "hand shake",
    68: "hand wave", 69: "hug", 70: "kick (person)", 71: "kiss",
    72: "lift (person)", 73: "listen to (person)", 74: "play with kids",
    75: "push (person)", 76: "sing to", 77: "take from (person)",
    78: "talk to", 79: "watch (person)",
}

# Highlight accident-related classes with a brighter color
ACCIDENT_RELATED = {4, 5, 7, 63, 65, 70, 75}  # fall down, get up, lie/sleep, fight, grab, kick, push


def timestamp_to_frame_idx(timestamp: int) -> int:
    """Mirror of load_data.py: key_frame_idx = (sec - 900) * 30 + 1"""
    return (timestamp - 900) * 30 + 1


def load_predictions(csv_path: Path):
    """
    Group predictions by (video_id, timestamp, rounded_bbox).
    Same box appears in many rows (one per action class). We aggregate
    so that each box gets the top-K highest-scoring actions.
    """
    grouped = defaultdict(list)  # key: (video, ts, box_key) → list of (action_id, score)

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            video_id  = row[0]
            timestamp = int(row[1])
            x1, y1, x2, y2 = map(float, row[2:6])
            action_id = int(row[6])    # 1..80 (1-indexed)
            score     = float(row[7])

            # Round box to 2 decimals so per-class jitter merges into one box
            box_key = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
            full_key = (video_id, timestamp, box_key)
            grouped[full_key].append((action_id, score, (x1, y1, x2, y2)))

    return grouped


def draw_box_with_labels(img, box, labels_with_scores, is_accident_box=False):
    """Draw one bbox with its top action labels above it."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

    color = (0, 0, 255) if is_accident_box else (0, 255, 0)  # red for accident, green otherwise
    thickness = 3 if is_accident_box else 2

    cv2.rectangle(img, (px1, py1), (px2, py2), color, thickness)

    # Draw labels above the box
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thick = 1
    line_h     = 18

    # Background strip for readability
    n_lines = len(labels_with_scores)
    strip_y2 = py1
    strip_y1 = max(0, py1 - line_h * n_lines - 4)
    cv2.rectangle(img, (px1, strip_y1), (px1 + 220, strip_y2), color, -1)

    for i, (name, score) in enumerate(labels_with_scores):
        text = f"{name} {score:.2f}"
        ty = strip_y1 + line_h * (i + 1) - 4
        cv2.putText(img, text, (px1 + 3, ty), font, font_scale,
                    (255, 255, 255), text_thick, cv2.LINE_AA)


def annotate_frame(frame_path: Path, detections, score_threshold: float, top_k: int):
    """
    detections: list of {box: (x1,y1,x2,y2), actions: [(action_id, score), ...]}
                where actions is already sorted desc by score.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    for det in detections:
        # Filter top_k actions above threshold
        kept = [(aid, sc) for aid, sc in det["actions"] if sc >= score_threshold][:top_k]
        if not kept:
            continue

        labels = [(IDX2NAME.get(aid - 1, f"id_{aid}"), sc) for aid, sc in kept]
        is_accident = any((aid - 1) in ACCIDENT_RELATED for aid, _ in kept)
        draw_box_with_labels(img, det["box"], labels, is_accident_box=is_accident)

    return img


def make_video(image_dir: Path, video_path: Path, fps: int = 2):
    """Stitch annotated keyframes (in timestamp order) into an MP4."""
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        return False

    first = cv2.imread(str(images[0]))
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        if (frame.shape[0], frame.shape[1]) != (H, W):
            frame = cv2.resize(frame, (W, H))
        writer.write(frame)

    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="Path to predicted CSV (e.g. ava_predicted_i3d.csv)")
    ap.add_argument("--frames_dir", required=True,
                    help="Root frames dir (e.g. /data2/cache/smart_data/frames)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to put annotated images / videos")
    ap.add_argument("--score_threshold", type=float, default=0.3,
                    help="Drop predictions below this confidence (default 0.3)")
    ap.add_argument("--top_k", type=int, default=3,
                    help="Show at most this many actions per box (default 3)")
    ap.add_argument("--make_video", action="store_true",
                    help="Also assemble per-video MP4 from annotated keyframes")
    ap.add_argument("--video_fps", type=int, default=2,
                    help="FPS of summary video (default 2 — slow so you can read)")
    ap.add_argument("--limit_videos", type=int, default=None,
                    help="(debug) only process this many videos")
    args = ap.parse_args()

    predictions_path = Path(args.predictions)
    frames_dir       = Path(args.frames_dir)
    output_dir       = Path(args.output_dir)

    img_root = output_dir / "images"
    vid_root = output_dir / "videos"
    img_root.mkdir(parents=True, exist_ok=True)
    if args.make_video:
        vid_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading predictions from {predictions_path}")
    grouped = load_predictions(predictions_path)
    print(f"      Loaded {len(grouped)} unique (video, ts, box) groups")

    # Reorganize: per (video, timestamp) → list of detections
    per_keyframe = defaultdict(list)
    for (video_id, ts, _box_key), entries in grouped.items():
        # entries: [(action_id, score, (x1,y1,x2,y2)), ...]
        entries.sort(key=lambda e: e[1], reverse=True)
        # Use the highest-scoring entry's exact box for drawing
        best_box = entries[0][2]
        per_keyframe[(video_id, ts)].append({
            "box": best_box,
            "actions": [(aid, sc) for aid, sc, _ in entries],
        })

    # Group by video for processing
    per_video = defaultdict(list)
    for (video_id, ts), dets in per_keyframe.items():
        per_video[video_id].append((ts, dets))

    print(f"[2/3] Annotating keyframes for {len(per_video)} videos")
    n_videos_done = 0
    n_frames_done = 0
    n_frames_skipped = 0

    for video_id in sorted(per_video.keys()):
        if args.limit_videos and n_videos_done >= args.limit_videos:
            break

        out_img_dir = img_root / video_id
        out_img_dir.mkdir(parents=True, exist_ok=True)

        keyframes = sorted(per_video[video_id], key=lambda x: x[0])
        for ts, dets in keyframes:
            frame_idx  = timestamp_to_frame_idx(ts)
            frame_name = f"{video_id}_{frame_idx:06d}.jpg"
            frame_path = frames_dir / video_id / frame_name

            if not frame_path.exists():
                n_frames_skipped += 1
                continue

            annotated = annotate_frame(
                frame_path, dets,
                score_threshold=args.score_threshold,
                top_k=args.top_k,
            )
            if annotated is None:
                n_frames_skipped += 1
                continue

            out_path = out_img_dir / f"{video_id}_t{ts:03d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            n_frames_done += 1

        n_videos_done += 1
        if n_videos_done % 5 == 0:
            print(f"      ... {n_videos_done}/{len(per_video)} videos processed")

    print(f"      Annotated {n_frames_done} keyframes "
          f"({n_frames_skipped} skipped due to missing files)")

    if args.make_video:
        print(f"[3/3] Assembling summary videos")
        for video_id in sorted(per_video.keys())[:args.limit_videos] \
                        if args.limit_videos else sorted(per_video.keys()):
            img_dir = img_root / video_id
            if not img_dir.exists():
                continue
            vid_path = vid_root / f"{video_id}.mp4"
            ok = make_video(img_dir, vid_path, fps=args.video_fps)
            if not ok:
                print(f"      [warn] no images for {video_id}, video not created")

    print()
    print(f"[DONE] Output written to: {output_dir}")
    print(f"       Annotated images:  {img_root}")
    if args.make_video:
        print(f"       Summary videos:    {vid_root}")


if __name__ == "__main__":
    main()
