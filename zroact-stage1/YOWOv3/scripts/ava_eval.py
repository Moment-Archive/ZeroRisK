import cv2
import torch
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3
from utils.box import non_max_suppression
from utils.build_config import build_config
import csv 
import tqdm
from evaluator.Evaluation import get_ava_performance
from utils.flops import get_info
import torch.utils.data as data

def ava_eval_collate_fn(batch):
    clips  = []
    vid_n  = []
    secs   = []

    for b in batch:
        clips.append(b[0])
        vid_n.append(b[3])
        secs.append(b[4])
    
    clips = torch.stack(clips, dim=0)
    return clips, vid_n, secs

def eval(config):

    print("=" * 60)
    print(f"  Backbone2D     : {config['backbone2D']}")
    print(f"  Backbone3D     : {config['backbone3D']}")
    print(f"  Fusion Module  : {config['fusion_module']}")
    print(f"  Pretrain Path  : {config['pretrain_path']}")
    print("=" * 60)

    dataset    = build_dataset(config, phase='test')
    dataloader = data.DataLoader(dataset, 32, False, collate_fn=ava_eval_collate_fn, num_workers=6, pin_memory=True)
    model      = build_yowov3(config)
    get_info(config, model)
    model.to("cuda")
    model.eval()
    
    black_list = [2, 4, 7, 9, 13, 16, 18, 19, 21, 22, 23, 24, 25, 29, 31, 32, 33, 35, 36, 39, 40, 41, 42, 44, 45, 46, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
    # og_black_list = [2, 16, 18, 19, 21, 23, 25, 31, 32, 33, 35, 39, 40, 42, 44, 50, 53, 55, 71, 75]
    config['ava_eval_excluded_ids'] = black_list
    # 파일명에 backbone3D 이름 삽입 (예: ava_predicted_file.csv → ava_predicted_shufflenetv2.csv)
    import os
    det_path = config['detections']
    det_dir = os.path.dirname(det_path)
    backbone3d_name = config['backbone3D']
    det_filename = f"ava_predicted_{backbone3d_name}.csv"
    ava_result_file = os.path.join(det_dir, det_filename)
    config['detections'] = ava_result_file
    print(f"  Result File    : {ava_result_file}")

    results = []
    total_before_nms = 0
    total_after_nms = 0

    with torch.no_grad():
        for clips, video_names, secs in tqdm.tqdm(dataloader):
            clips = clips.to('cuda')
            outputss = model(clips)
            outputss = outputss.cpu()
            
            for outputs, video_name, sec in zip(outputss, video_names, secs):   
                outputs = outputs.unsqueeze(0)
                total_before_nms += outputs.shape[1]
                outputs = non_max_suppression(outputs, 0.01, 0.5)[0]
                total_after_nms += outputs.shape[0]

                H = config['img_size']
                W = config['img_size']

                outputs[:, 0] /= W
                outputs[:, 1] /= H
                outputs[:, 2] /= W
                outputs[:, 3] /= H
        
                for output in outputs:
                    if (int(output[5].item()) + 1) in black_list:
                        continue
                        
                    results.append([video_name, sec, round(output[0].item(), 3), 
                        round(output[1].item(), 3), round(output[2].item(), 3), 
                        round(output[3].item(), 3), int(output[5].item()) + 1, 
                        round(output[4].item(), 3)])


        with open(ava_result_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

        print("AVA eval debug - total before NMS:", total_before_nms, flush=True)
        print("AVA eval debug - total after NMS:", total_after_nms, flush=True)
        print("AVA eval debug - saved rows:", len(results), flush=True)

        get_ava_performance.eval(config)
