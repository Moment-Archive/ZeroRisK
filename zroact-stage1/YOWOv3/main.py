from scripts import train, ava_eval, ucf_eval, detect, live, onnx
import argparse
from utils.build_config import build_config
import time
import datetime

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="YOWOv3")

    parser.add_argument('-m', '--mode', type=str, help='train/eval/live/detect/onnx')
    parser.add_argument('-cf', '--config', type=str, help='path to config file')

    args, _ = parser.parse_known_args()

    config = build_config(args.config)

    if args.mode == 'train':
        train.train_model(config=config)

    elif args.mode == 'eval':
        if config['dataset'] == 'ucf' or config['dataset'] == 'jhmdb' or config['dataset'] == 'ucfcrime':
            ucf_eval.eval(config=config)
        elif config['dataset'] == 'ava':
            ava_eval.eval(config=config)

    elif args.mode == 'detect':
        detect.detect(config=config)

    elif args.mode == 'live':
        live.detect(config=config)
    
    elif args.mode == 'onnx':
        onnx.export2onnx(config=config)

    elif args.mode == 'custom_frame_infer':
        from scripts.custom_frame_infer import custom_frame_infer
        import argparse
        import sys
        # Extend parser if called from command line, or just extract values from config / defaults
        parser.add_argument('--video_dir', type=str, default=None, help='Path to raw video directory (if provided, extracts frames first)')
        parser.add_argument('--frames_dir', type=str, default='/data2/cache/smart_data/frames')
        parser.add_argument('--output_dir', type=str, default='/data2/cache/smart_data/viz_custom')
        parser.add_argument('--conf_threshold', type=float, default=0.3)
        parser.add_argument('--top_k', type=int, default=3)
        parser.add_argument('--sample_rate', type=int, default=30)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--make_video', action='store_true')
        parser.add_argument('--video_fps', type=int, default=2)
        parser.add_argument('--no_blacklist', action='store_true')
        parser.add_argument('--extraction_method', type=str, choices=['fps', 'interpolate'], default='interpolate')
        parser.add_argument('--extraction_workers', type=int, default=4)
        parser.add_argument('--onnx_path', type=str, default=None, help='Path to exported YOWOv3 ONNX model')
        
        # Parse args again with the new arguments
        ns, _ = parser.parse_known_args()

        if ns.video_dir:
            from make_30fps import extract_frames
            print(f"\n[*] Extracting frames from {ns.video_dir} -> {ns.frames_dir}...")
            success = extract_frames(
                src_dir=ns.video_dir,
                dest_dir=ns.frames_dir,
                method=ns.extraction_method,
                workers=ns.extraction_workers
            )
            if not success:
                print("[!] Frame extraction failed or no videos found. Exiting.")
                sys.exit(1)
        
        custom_frame_infer(
            config=config,
            frames_dir=ns.frames_dir,
            output_dir=ns.output_dir,
            conf_threshold=ns.conf_threshold,
            top_k=ns.top_k,
            sample_rate=ns.sample_rate,
            batch_size=ns.batch_size,
            make_video=ns.make_video,
            video_fps=ns.video_fps,
            use_blacklist=not ns.no_blacklist,
            onnx_path=ns.onnx_path
        )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n=== 소요 시간(Execution Time): {datetime.timedelta(seconds=int(elapsed_time))} ===")