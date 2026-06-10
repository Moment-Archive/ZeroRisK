import time
import cProfile
import pstats
import io
from vllm import LLM
from PIL import Image

def main():
    model_path = "/home/capstone2/zroact-stage2/benchmark2/models/Qwen3.5-2B"
    print("⏳ vLLM 모델 로딩 중...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 3},
        gpu_memory_utilization=0.3
    )

    img_paths = [
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000016.jpg",
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000026.jpg",
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000036.jpg"
    ]
    images = [Image.open(p).convert("RGB").resize((768, 432)) for p in img_paths]

    tokenizer = llm.get_tokenizer()
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
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    mm_kwargs = {
        "min_pixels": 28 * 28 * 16,
        "max_pixels": 768 * 432
    }

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": images
        },
        "mm_processor_kwargs": mm_kwargs
    }

    print("🔥 Profiling preprocess_cmpl...")
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the preprocess step that vllm runs during generate
    res = llm._preprocess_cmpl([inputs], mm_processor_kwargs=mm_kwargs)
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())

if __name__ == '__main__':
    main()
