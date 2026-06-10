import os
import glob
import subprocess
import argparse
from multiprocessing import Pool
from tqdm import tqdm

def process_video(args):
    video_path, src_dir, dest_dir, method = args
    
    # 원본 비디오의 하위 디렉토리 구조를 그대로 유지하여 프레임 저장 경로 생성
    rel_dir = os.path.relpath(os.path.dirname(video_path), src_dir)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if rel_dir == ".":
        out_video_dir = os.path.join(dest_dir, video_name)
    else:
        out_video_dir = os.path.join(dest_dir, rel_dir, video_name)
    os.makedirs(out_video_dir, exist_ok=True)
    
    # 출력될 프레임의 파일명 포맷 설정 (예: 30fps_frame_001.jpg, 30fps_frame_002.jpg ...)
    out_name = os.path.join(out_video_dir, "30fps_frame_%03d.jpg")
    
    # 프레임 추출 방식 선택
    if method == "interpolate":
        # 모션 보간(Motion Interpolation) 방식: 프레임 간 움직임을 계산하여 자연스러운 30fps 생성 (화질 우수, 매우 느림)
        filter_str = "minterpolate=fps=30:mi_mode=blend:mc_mode=aobmc:vsbmc=1"
    else:
        # 표준 FPS 조정 방식: 프레임을 복사하거나 드롭하여 30fps 생성 (단순 작업, 속도 빠름)
        filter_str = "fps=30"
        
    cmd = [
        "ffmpeg", "-y",
        "-threads", "1", # 멀티프로세싱 시 개별 프로세스의 과도한 CPU/메모리 사용을 제한하기 위해 단일 스레드로 실행
        "-i", video_path,
        "-vf", filter_str,
        "-q:v", "2",
        out_name
    ]
    
    # FFmpeg 명령어를 백그라운드에서 조용히 실행
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_frames(src_dir, dest_dir, method="interpolate", workers=4):
    src_dir = os.path.abspath(src_dir)
    dest_dir = os.path.abspath(dest_dir)
    
    if not os.path.exists(src_dir):
        print(f"[Error] Source directory not found: {src_dir}")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    
    # 소스 디렉토리 내에서 탐색할 비디오 확장자 목록 정의
    extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi', '*.mov']
    video_files = []
    
    # 대소문자를 모두 고려하여 디렉토리 하위의 모든 비디오 파일 탐색
    for ext in extensions:
        pattern_lower = os.path.join(src_dir, "**", ext)
        video_files.extend(glob.glob(pattern_lower, recursive=True))
        pattern_upper = os.path.join(src_dir, "**", ext.upper())
        video_files.extend(glob.glob(pattern_upper, recursive=True))
        
    # 중복 제거 및 경로 정렬
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"[Error] No videos found in: {src_dir}")
        return False
        
    print(f"Processing {len(video_files)} videos found in {src_dir}...")
    print(f"Output destination: {dest_dir}")
    print(f"Method: {method}")
    print(f"Max processes: {workers}")
    
    # 멀티프로세싱 풀(Pool)에 전달할 인수 튜플 리스트 생성
    tasks = [(video, src_dir, dest_dir, method) for video in video_files]
    
    # 설정한 프로세스 수(workers)만큼 병렬로 비디오 프레임 추출 실행 (tqdm으로 진행 상황 표시)
    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, tasks), total=len(video_files)))
        
    print("\nFrame extraction completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract video frames at exactly 30 FPS using FFmpeg.")
    parser.add_argument("--src", "-s", required=True, help="Path to the source video directory.")
    parser.add_argument("--dest", "-d", required=True, help="Path to the destination frame directory.")
    parser.add_argument(
        "--method", 
        "-m", 
        choices=["fps", "interpolate"], 
        default="interpolate", 
        help="Frame extraction method: 'fps' (fast standard) or 'interpolate' (slow motion-compensated)."
    )
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of concurrent worker processes.")
    
    args = parser.parse_args()
    extract_frames(args.src, args.dest, args.method, args.workers)


if __name__ == "__main__":
    main()