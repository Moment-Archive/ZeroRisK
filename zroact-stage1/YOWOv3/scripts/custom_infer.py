import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os
import cv2
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
from tqdm import tqdm
from PIL import Image

from utils.box import box_iou, opacity
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from utils.flops import get_info

class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip, mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        W, H = img.size
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

def box_label(image, box, label=None, color=(0, 255, 0), txt_color=(0, 0, 0)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw-1, lineType=cv2.LINE_AA)
    if label is not None:
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 10, thickness=tf)[0]
        outside = p1[1] - h >= 3
        y0, dy = 0, 10
        y0 = p1[1] - 2 if outside else p1[1] + h + 2
        for i, line in enumerate(label.split('\n')):
            y = y0 + i*dy
            x = p1[0]
            text_size, _ = cv2.getTextSize(line, cv2.LINE_AA, lw/5, tf)
            text_width, text_height = text_size
            opacity(image, (x, int(y - text_height)), (int(x + text_width + 1), int(y + 2)))
            cv2.putText(image, line, (x, y), 0, lw / 5, txt_color, thickness=tf, lineType=cv2.LINE_AA)

def draw_bounding_box_custom(image, bboxes, labels, confs, map_labels, black_list):
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)[:, :, (2, 1, 0)].contiguous()
        image = image.detach().cpu().numpy()
    
    H, W, C = image.shape 

    pre_box   = []
    meta_data = []

    if bboxes is not None:
        for box, label, conf in zip(bboxes, labels, confs):
            label_int = int(label.item()) + 1
            if label_int in black_list:
                continue

            box = box.clone().detach()
            text = str(map_labels[int(label.item())] + " : " + str(round(conf.item()*100, 2)))
            pre_box.append(box)
            meta_data.append([text, H, W])

        if len(pre_box) == 0:
            return

        res_box = []
        res_meta_data = []

        for idx1, box1 in enumerate(pre_box):
            flag = 1
            for idx2, box2 in enumerate(res_box):
                iou = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0, 0]
                if (iou >= 0.9):
                    res_meta_data[idx2].append(meta_data[idx1])
                    flag = 0
                    break
            if flag:
                res_box.append(box1)
                res_meta_data.append([meta_data[idx1]])

        for box, meta in zip(res_box, res_meta_data):
            text = ''
            for sub_meta in meta:
                if text == '':
                    text = sub_meta[0]
                else:
                    text += '\n' + sub_meta[0]

            H = meta[0][1]
            W = meta[0][2]
            
            bbox = []
            bbox.append(int(max(0, box[0])))
            bbox.append(int(max(0, box[1])))
            bbox.append(int(min(W, box[2])))
            bbox.append(int(min(H, box[3])))

            box_label(image, bbox, text)

def custom_infer(config, input_dir, output_dir):
    model = build_yowov3(config) 
    get_info(config, model)
    model.to("cuda")
    model.eval()
    mapping = config['idx2name']
    
    black_list = [2, 4, 7, 9, 13, 16, 18, 19, 21, 22, 23, 24, 25, 29, 31, 32, 33, 35, 36, 39, 40, 41, 42, 44, 45, 46, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(input_dir, '**', '*.mp4'), recursive=True) + glob.glob(os.path.join(input_dir, '**', '*.avi'), recursive=True)
    
    if len(video_files) == 0:
        print(f"No video files found in {input_dir}")
        return

    transform = live_transform(config['img_size'])
    from utils.box import non_max_suppression

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, video_name)
        print(f"Processing {video_path} -> {output_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (config['img_size'], config['img_size']))

        frame_list = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            origin_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_list.append(transform(origin_image))
            
            if len(frame_list) > 16:
                frame_list.pop(0)
            if len(frame_list) < 16:
                origin_image_bgr = cv2.resize(frame, (config['img_size'], config['img_size']))
                out.write(origin_image_bgr)
                pbar.update(1)
                continue

            clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous()
            clip = clip.unsqueeze(0).to("cuda")
            
            with torch.no_grad():
                outputs = model(clip)
            
            outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]

            origin_image_bgr = cv2.resize(frame, (config['img_size'], config['img_size']))

            if outputs is not None and outputs.size(0) > 0:
                draw_bounding_box_custom(origin_image_bgr, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping, black_list)

            out.write(origin_image_bgr)
            pbar.update(1)

        cap.release()
        out.release()
        pbar.close()
        print(f"Finished {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOWOv3 Custom Video Inference')
    parser.add_argument('--input_dir', type=str, default='/data2/cache/smart_data/video', help='Directory of input videos')
    parser.add_argument('--output_dir', type=str, default='/data2/cache/smart_data/results', help='Directory for output videos')
    parser.add_argument('--config', type=str, default='config/cf2/ava_config.yaml', help='Path to YOWOv3 config')
    
    args = parser.parse_args()

    config = build_config(args.config)
    custom_infer(config, args.input_dir, args.output_dir)
