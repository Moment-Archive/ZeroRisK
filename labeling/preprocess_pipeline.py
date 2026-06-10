import os
import shutil
import re
import argparse
from pathlib import Path

# Category configurations for Step 2
CATEGORIES = [
    {"name": "unsafe", "start": 0, "end": 1, "target": 7},
    {"name": "no_hardhat", "start": 2, "end": 3, "target": 8},
    {"name": "climb_over_fence", "start": 5, "end": 4, "target": 9},
]

def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        box = [float(x) for x in parts[1:]]
        return class_id, box
    except ValueError:
        return None

def format_label_line(class_id, box):
    return f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"

# 1. Step 1: Cutoff 시퀀스 개수 제한
def run_step1_cutoff(dataset_dir, limit_name, limit_count):
    dataset_path = Path(dataset_dir)
    obj_train_data = dataset_path / "obj_train_data"
    images_dir = dataset_path / "images"
    
    # obj_train_data 하위의 시퀀스 디렉토리 찾기
    seq_dirs = []
    for root, dirs, files in os.walk(obj_train_data):
        if any(f.endswith('.txt') and '30fps_frame' in f for f in files):
            seq_dirs.append(Path(root))
            
    seq_dirs = sorted(seq_dirs, key=lambda x: x.name)
    print(f"[{dataset_dir}] Found {len(seq_dirs)} raw sequences.")
    
    # 갯수 제한
    kept_dirs = seq_dirs[:limit_count]
    deleted_dirs = seq_dirs[limit_count:]
    
    # 보존 마지막 폴더 검증
    if kept_dirs:
        last_kept = kept_dirs[-1].name
        print(f"[{dataset_dir}] Last kept sequence: {last_kept} (Expected: {limit_name})")
        if last_kept != limit_name:
            print(f"Warning: Last kept sequence '{last_kept}' does not match expected '{limit_name}'")
            
    # 삭제 수행
    for d in deleted_dirs:
        if d.exists():
            shutil.rmtree(d)
        rel_to_obj = d.relative_to(obj_train_data)
        img_seq_dir = images_dir / rel_to_obj
        if img_seq_dir.exists():
            shutil.rmtree(img_seq_dir)
            
    print(f"[{dataset_dir}] Cutoff complete. Kept {len(kept_dirs)} sequences.")

# 2. Step 2: Interpolation (process_labels.py 로직 내장)
def process_sequence_interpolation(seq_path):
    txt_files = []
    for f in os.listdir(seq_path):
        if f.endswith('.txt'):
            match = re.search(r'30fps_frame_(\d+)\.txt', f)
            if match:
                frame_idx = int(match.group(1))
                txt_files.append((frame_idx, os.path.join(seq_path, f)))
                
    if not txt_files:
        return
        
    txt_files.sort(key=lambda x: x[0])
    
    frame_to_path = {}
    frame_to_base_annos = {}
    starts_by_cat = {i: [] for i in range(len(CATEGORIES))}
    ends_by_cat = {i: [] for i in range(len(CATEGORIES))}
    
    for frame_idx, file_path in txt_files:
        frame_to_path[frame_idx] = file_path
        base_annos = []
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                parsed = parse_label_line(line)
                if parsed:
                    class_id, box = parsed
                    if class_id in [7, 8, 9]:
                        continue
                    base_annos.append((class_id, box))
                    
                    for cat_idx, cat in enumerate(CATEGORIES):
                        if class_id == cat["start"]:
                            starts_by_cat[cat_idx].append((frame_idx, box))
                        elif class_id == cat["end"]:
                            ends_by_cat[cat_idx].append((frame_idx, box))
                            
        frame_to_base_annos[frame_idx] = base_annos

    new_annos_to_add = {frame_idx: [] for frame_idx in frame_to_path.keys()}
    
    for cat_idx, cat in enumerate(CATEGORIES):
        starts = sorted(starts_by_cat[cat_idx], key=lambda x: x[0])
        ends = sorted(ends_by_cat[cat_idx], key=lambda x: x[0])
        pairs = []
        
        for start_idx, start_box in starts:
            matched_end = None
            for end_idx, end_box in ends:
                if end_idx >= start_idx:
                    matched_end = (end_idx, end_box)
                    break
            if matched_end:
                pairs.append((start_idx, start_box, matched_end[0], matched_end[1]))
            else:
                last_frame_idx = txt_files[-1][0]
                pairs.append((start_idx, start_box, last_frame_idx, start_box))
                
        for end_idx, end_box in ends:
            covered = False
            for s_idx, _, e_idx, _ in pairs:
                if s_idx <= end_idx <= e_idx:
                    covered = True
                    break
            if not covered:
                matched_start = None
                for start_idx, start_box in reversed(starts):
                    if start_idx <= end_idx:
                        matched_start = (start_idx, start_box)
                        break
                if matched_start:
                    pairs.append((matched_start[0], matched_start[1], end_idx, end_box))
                else:
                    first_frame_idx = txt_files[0][0]
                    pairs.append((first_frame_idx, end_box, end_idx, end_box))
                    
        for start_idx, start_box, end_idx, end_box in pairs:
            for t in range(start_idx, end_idx + 1):
                if t not in frame_to_path:
                    continue
                if start_idx == end_idx:
                    box_t = start_box
                else:
                    ratio = (t - start_idx) / (end_idx - start_idx)
                    box_t = [
                        start_box[i] + ratio * (end_box[i] - start_box[i])
                        for i in range(4)
                    ]
                new_annos_to_add[t].append((cat["target"], box_t))

    for frame_idx, file_path in txt_files:
        base_annos = frame_to_base_annos[frame_idx]
        new_annos = new_annos_to_add[frame_idx]
        all_annos = base_annos + new_annos
        with open(file_path, 'w', encoding='utf-8') as f:
            for class_id, box in all_annos:
                f.write(format_label_line(class_id, box))

def run_step2_interpolation(dataset_dir):
    dataset_path = Path(dataset_dir)
    obj_train_data = dataset_path / "obj_train_data"
    
    seq_dirs = []
    for root, dirs, files in os.walk(obj_train_data):
        if any(f.endswith('.txt') and '30fps_frame' in f for f in files):
            seq_dirs.append(Path(root))
            
    for seq in sorted(seq_dirs):
        process_sequence_interpolation(seq)
    print(f"[{dataset_dir}] Interpolation complete.")

# 3. Step 3: Back Frame Cutoff
def run_step3_back_cutoff(dataset_dir):
    dataset_path = Path(dataset_dir)
    obj_train_data = dataset_path / "obj_train_data"
    images_dir = dataset_path / "images"
    
    seq_dirs = []
    for root, dirs, files in os.walk(obj_train_data):
        if any(f.endswith('.txt') and '30fps_frame' in f for f in files):
            seq_dirs.append(Path(root))
            
    total_deleted_frames = 0
    
    for seq in sorted(seq_dirs):
        txt_files = []
        for f in os.listdir(seq):
            if f.endswith('.txt') and '30fps_frame' in f:
                match = re.search(r'30fps_frame_(\d+)\.txt', f)
                if match:
                    idx = int(match.group(1))
                    txt_files.append((idx, seq / f))
                    
        txt_files.sort(key=lambda x: x[0])
        
        first_back_idx = None
        for idx, path in txt_files:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts and int(parts[0]) == 6:  # class 6 is back
                    first_back_idx = idx
                    break
            if first_back_idx is not None:
                break
                
        if first_back_idx is not None:
            rel_to_obj = seq.relative_to(obj_train_data)
            img_seq = images_dir / rel_to_obj
            deleted_count = 0
            
            for idx, path in txt_files:
                if idx > first_back_idx:
                    if path.exists():
                        os.remove(path)
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = img_seq / f"30fps_frame_{idx:03d}{ext}"
                        if img_path.exists():
                            os.remove(img_path)
                            break
                    deleted_count += 1
            if deleted_count > 0:
                print(f"[{dataset_dir}] Sequence {seq.name}: Cutoff after frame {first_back_idx} (deleted {deleted_count} frames).")
                total_deleted_frames += deleted_count
            
    print(f"[{dataset_dir}] Back cutoff complete. Total {total_deleted_frames} frames deleted.")

# 4. Step 4: Class Reduction & Shift
def run_step4_class_reduction(dataset_dir):
    dataset_path = Path(dataset_dir)
    obj_train_data = dataset_path / "obj_train_data"
    
    seq_dirs = []
    for root, dirs, files in os.walk(obj_train_data):
        if any(f.endswith('.txt') and '30fps_frame' in f for f in files):
            seq_dirs.append(Path(root))
            
    for seq in sorted(seq_dirs):
        for f in os.listdir(seq):
            if f.endswith('.txt') and '30fps_frame' in f:
                path = seq / f
                with open(path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    
                new_annos = []
                has_unsafe = False
                has_danger = False
                
                for line in lines:
                    parsed = parse_label_line(line)
                    if parsed:
                        class_id, box = parsed
                        if class_id == 7:  # unsafe interpolation
                            new_annos.append((0, box))
                            has_unsafe = True
                        elif class_id == 9:  # climb_over_fence interpolation
                            new_annos.append((1, box))
                            has_danger = True
                            
                # If neither unsafe nor danger exists, it is normal
                if not has_unsafe and not has_danger:
                    new_annos.append((2, [0.5, 0.5, 1.0, 1.0]))
                    
                with open(path, 'w', encoding='utf-8') as file:
                    for class_id, box in new_annos:
                        file.write(format_label_line(class_id, box))
                        
    # Update obj.names and obj.data
    with open(dataset_path / "obj.names", 'w', encoding='utf-8') as file:
        file.write("unsafe\ndanger\nnormal\n")
        
    with open(dataset_path / "obj.data", 'w', encoding='utf-8') as file:
        file.write("classes = 3\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
        
    print(f"[{dataset_dir}] Class reduction and metadata shift complete.")

# 5. Rebuild train.txt based on actual surviving images
def rebuild_train_txt(dataset_dir):
    dataset_path = Path(dataset_dir)
    obj_train_data = dataset_path / "obj_train_data"
    
    img_files = []
    for root, dirs, files in os.walk(obj_train_data):
        for f in files:
            if f.endswith('.txt') and '30fps_frame' in f:
                txt_path = Path(root) / f
                rel = txt_path.relative_to(obj_train_data)
                rel_jpg = rel.with_suffix('.jpg')
                line = f"data/obj_train_data/{rel_jpg}"
                img_files.append(line)
                
    img_files = sorted(img_files)
    
    with open(dataset_path / "train.txt", 'w', encoding='utf-8') as file:
        for line in img_files:
            file.write(line + "\n")
            
    print(f"[{dataset_dir}] train.txt rebuilt with {len(img_files)} image references (based on surviving labels).")

def main():
    parser = argparse.ArgumentParser(description="Integrated 4-Step Preprocessing Pipeline for ZroAct.")
    parser.add_argument("--no-cutoff", action="store_true", help="Skip Step 1 sequence cutoff (process all sequences).")
    args = parser.parse_args()
    
    current_dir = Path.cwd()
    print(f"Executing preprocessing pipeline under: {current_dir}\n")
    
    # Configurations for Step 1 Cutoff
    configs = {
        "smart_climb": {"limit_name": "intrusion_climb-over-fence_rgb_0107_cctv1", "limit_count": 10},
        "plant_climb": {"limit_name": "intrusion_climb-over-fence_rgb_0054_cctv1", "limit_count": 10},
        "normal_plant": {"limit_name": "intrusion_normal_rgb_0166_cctv1", "limit_count": 20},
    }
    
    for d, cfg in configs.items():
        dataset_path = current_dir / d
        if not dataset_path.exists():
            print(f"Skipping missing dataset: {d}")
            continue
            
        print(f"=== PROCESSING DATASET: {d} ===")
        # Step 1
        if args.no_cutoff:
            print(f"[{d}] Skipping Step 1 Cutoff (processing all sequences as requested).")
        else:
            run_step1_cutoff(dataset_path, cfg["limit_name"], cfg["limit_count"])
        # Step 2
        run_step2_interpolation(dataset_path)
        # Step 3
        run_step3_back_cutoff(dataset_path)
        # Step 4
        run_step4_class_reduction(dataset_path)
        # Step 5
        rebuild_train_txt(dataset_path)
        print(f"=== DATASET {d} PROCESSED SUCCESSFULLY ===\n")
        
if __name__ == "__main__":
    main()
