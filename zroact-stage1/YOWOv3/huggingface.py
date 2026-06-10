import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
	parser = argparse.ArgumentParser(description="Download a checkpoint from Hugging Face Hub")
	parser.add_argument(
		"--local-dir",
		default="/data2/lju_capstone/ckpt/YOWOv3",
		help="Directory where the downloaded file will be saved",
	)
	args = parser.parse_args()

	repo_id = "manh6054/YOWOv3"
	file_path_in_repo = "checkpoint/avav2.2/M23/ema_epoch_9.pth"
	local_dir = Path(args.local_dir)
	local_dir.mkdir(parents=True, exist_ok=True)

	weights_path = hf_hub_download(
		repo_id=repo_id,
		filename=file_path_in_repo,
		local_dir=str(local_dir),
	)
	print(f"가중치 다운로드 완료: {weights_path}")


if __name__ == "__main__":
	main()