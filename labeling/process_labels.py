import os
import re
import argparse
from pathlib import Path

# Category configurations: (start_class, end_class, target_class)
CATEGORIES = [
    {"name": "unsafe", "start": 0, "end": 1, "target": 7},
    {"name": "no_hardhat", "start": 2, "end": 3, "target": 8},
    {"name": "climb_over_fence", "start": 5, "end": 4, "target": 9},
]

def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    class_id = int(parts[0])
    box = [float(x) for x in parts[1:]]
    return class_id, box

def format_label_line(class_id, box):
    return f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"

def process_sequence(seq_path, dry_run=False):
    # Find all txt files
    txt_files = []
    for f in os.listdir(seq_path):
        if f.endswith('.txt'):
            match = re.search(r'30fps_frame_(\d+)\.txt', f)
            if match:
                frame_idx = int(match.group(1))
                txt_files.append((frame_idx, os.path.join(seq_path, f)))
    
    if not txt_files:
        return
    
    # Sort txt files by frame index
    txt_files.sort(key=lambda x: x[0])
    
    # Map from frame_idx to original annotations (only classes 0-6)
    # and map from frame_idx to file path
    frame_to_path = {}
    frame_to_base_annos = {}
    
    # To keep track of starts and ends per category
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
                    # If it's one of the target state classes (7, 8, 9), discard it to make it idempotent
                    if class_id in [7, 8, 9]:
                        continue
                    base_annos.append((class_id, box))
                    
                    # Check if it matches any category start/end
                    for cat_idx, cat in enumerate(CATEGORIES):
                        if class_id == cat["start"]:
                            starts_by_cat[cat_idx].append((frame_idx, box))
                        elif class_id == cat["end"]:
                            ends_by_cat[cat_idx].append((frame_idx, box))
                            
        frame_to_base_annos[frame_idx] = base_annos

    # Map from frame_idx to list of new target annotations to add
    new_annos_to_add = {frame_idx: [] for frame_idx in frame_to_path.keys()}
    
    # Pair starts and ends for each category, and interpolate
    for cat_idx, cat in enumerate(CATEGORIES):
        starts = sorted(starts_by_cat[cat_idx], key=lambda x: x[0])
        ends = sorted(ends_by_cat[cat_idx], key=lambda x: x[0])
        
        pairs = []
        
        # Match each start to the first succeeding end
        for start_idx, start_box in starts:
            matched_end = None
            for end_idx, end_box in ends:
                if end_idx >= start_idx:
                    matched_end = (end_idx, end_box)
                    break
            if matched_end:
                pairs.append((start_idx, start_box, matched_end[0], matched_end[1]))
            else:
                print(f"Warning: Category '{cat['name']}' in {seq_path} has unmatched start at frame {start_idx}")
                last_frame_idx = txt_files[-1][0]
                pairs.append((start_idx, start_box, last_frame_idx, start_box))
                
        # Match each end to the closest preceding start if not already covered
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
                    print(f"Warning: Category '{cat['name']}' in {seq_path} has unmatched end at frame {end_idx}")
                    first_frame_idx = txt_files[0][0]
                    pairs.append((first_frame_idx, end_box, end_idx, end_box))
                
        # Perform interpolation for matched pairs
        for start_idx, start_box, end_idx, end_box in pairs:
            # Interpolate for each frame in [start_idx, end_idx]
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

    # Write back to files if not dry_run
    if not dry_run:
        for frame_idx, file_path in txt_files:
            base_annos = frame_to_base_annos[frame_idx]
            new_annos = new_annos_to_add[frame_idx]
            
            # Merge them
            all_annos = base_annos + new_annos
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for class_id, box in all_annos:
                    f.write(format_label_line(class_id, box))

def process_dataset(dataset_dir, dry_run=False):
    print(f"\nProcessing dataset: {dataset_dir}")
    obj_train_data_dir = os.path.join(dataset_dir, "obj_train_data")
    if not os.path.exists(obj_train_data_dir):
        print(f"Directory not found: {obj_train_data_dir}")
        return
    
    # Discover all sequences
    sequences = []
    for root, dirs, files in os.walk(obj_train_data_dir):
        # We look for directories that contain frame txt files
        if any(f.endswith('.txt') and '30fps_frame' in f for f in files):
            sequences.append(root)
            
    print(f"Found {len(sequences)} sequence directories in {dataset_dir}")
    for seq in sorted(sequences):
        process_sequence(seq, dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(description="Expand start/end labels into interval labels with interpolation.")
    parser.add_argument("--dir", help="Specific dataset directory to process (e.g. plant_climb). If not specified, processes all 3.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without writing any file changes.")
    args = parser.parse_args()
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    datasets = ["normal_plant", "plant_climb", "smart_climb"]
    if args.dir:
        if args.dir in datasets:
            datasets = [args.dir]
        else:
            print(f"Error: Directory '{args.dir}' must be one of {datasets}")
            return
            
    for d in datasets:
        dataset_path = os.path.join(current_dir, d)
        if os.path.exists(dataset_path):
            process_dataset(dataset_path, dry_run=args.dry_run)
        else:
            print(f"Dataset path not found: {dataset_path}")

if __name__ == "__main__":
    main()
