import os
import re
from pathlib import Path

def validate_line(line, file_path):
    parts = line.strip().split()
    if len(parts) != 5:
        return f"Error: Line does not have 5 parts: '{line.strip()}'"
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return f"Error: Could not parse float values: '{line.strip()}'"
        
    if class_id < 0 or class_id > 9:
        return f"Error: Class ID {class_id} is out of bounds [0, 9]"
        
    for i, c in enumerate(coords):
        if c < 0.0 or c > 1.0:
            return f"Warning: Coordinate {c} is out of bounds [0, 1] in line '{line.strip()}'"
            
    return None

def validate_dataset(dataset_dir):
    print(f"Validating dataset: {dataset_dir}")
    obj_train_data_dir = os.path.join(dataset_dir, "obj_train_data")
    if not os.path.exists(obj_train_data_dir):
        print(f"Skipping missing dir: {obj_train_data_dir}")
        return
        
    total_files = 0
    total_errors = 0
    total_warnings = 0
    
    for root, dirs, files in os.walk(obj_train_data_dir):
        for f in files:
            if f.endswith('.txt') and '30fps_frame' in f:
                total_files += 1
                file_path = os.path.join(root, f)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                for line_idx, line in enumerate(lines):
                    err = validate_line(line, file_path)
                    if err:
                        rel_path = os.path.relpath(file_path, dataset_dir)
                        if "Error" in err:
                            print(f"[{rel_path} L{line_idx+1}] {err}")
                            total_errors += 1
                        else:
                            # Print only first few warnings to avoid output bloat
                            if total_warnings < 5:
                                print(f"[{rel_path} L{line_idx+1}] {err}")
                            total_warnings += 1
                            
    print(f"Finished. Validated {total_files} files. Total errors: {total_errors}, Total warnings: {total_warnings}")
    return total_errors

def main():
    datasets = ["normal_plant", "plant_climb", "smart_climb"]
    current_dir = Path.cwd()
    
    grand_errors = 0
    for d in datasets:
        dataset_path = os.path.join(current_dir, d)
        if os.path.exists(dataset_path):
            errors = validate_dataset(dataset_path)
            grand_errors += errors
            
    if grand_errors == 0:
        print("\nAll datasets validated successfully! No format errors found.")
    else:
        print(f"\nValidation failed with {grand_errors} errors.")

if __name__ == "__main__":
    main()
