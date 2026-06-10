import time
from transformers import Qwen2VLProcessor
from PIL import Image

def main():
    model_path = "/home/capstone2/zroact-stage2/benchmark2/models/Qwen3.5-2B"
    print("⏳ Loading processor...")
    processor = Qwen2VLProcessor.from_pretrained(model_path)

    img_paths = [
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000016.jpg",
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000026.jpg",
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000036.jpg"
    ]
    
    print("⏳ Resizing images in PIL...")
    images = [Image.open(p).convert("RGB").resize((768, 432)) for p in img_paths]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "위 3장의 연속된 프레임에서 고위험 행동이나 위험 요소를 찾아 감지해줘."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Measure HF processor call
    print("🔥 Measuring processor.__call__...")
    start_time = time.time()
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt"
    )
    print(f"⏱️ processor.__call__ took {time.time() - start_time:.4f} seconds")
    
    # Print the shape of pixel values and grid_thw to inspect
    print("Pixel values shape:", inputs["pixel_values"].shape)
    print("Image grid THW:", inputs["image_grid_thw"])

if __name__ == '__main__':
    main()
