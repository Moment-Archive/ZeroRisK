import time
from vllm import LLM, SamplingParams
from PIL import Image

def main():
    model_path = "/home/capstone2/zroact-stage2/benchmark2/models/Qwen3.5-2B"

    print("⏳ vLLM 모델 로딩 중...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=8192,  # 이미지 3장의 토큰 합(약 6160)을 수용하기 위해 8192로 설정
        limit_mm_per_prompt={"image": 3},  # 3장의 이미지를 처리할 수 있도록 설정
        gpu_memory_utilization=0.3 # RTX A6000의 30%인 약 14GB만 할당
    )

    img_paths = [
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000016.jpg",
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000026.jpg",
        "/home/capstone2/test_data/frames/intrusion_climb-over-fence_rgb_0004_cctv1_t000036.jpg"
    ]
    images = [Image.open(p).convert("RGB").resize((768, 432)) for p in img_paths]

    # 토크나이저를 사용해 Qwen3.5 VLM 형식에 맞게 프롬프트 구성
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
        "min_pixels": 28 * 28 * 16,   # 약 12,544 픽셀
        "max_pixels": 768 * 432       # 약 331,776 픽셀 (기존 파이프라인 크기)
    }

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": images
        },
        "mm_processor_kwargs": mm_kwargs
    }

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    print("🔥 첫 번째 추론 시작 (Cold Start)...")
    start_time = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    end_time = time.time()

    print(f"✨ 첫 번째 결과: {outputs[0].outputs[0].text}")
    print(f"⏱️ 첫 번째 추론 소요 시간: {end_time - start_time:.2f}초")

    print("\n🔥 두 번째 추론 시작 (Warm Start)...")
    start_time2 = time.time()
    outputs2 = llm.generate(inputs, sampling_params=sampling_params)
    end_time2 = time.time()

    print(f"✨ 두 번째 결과: {outputs2[0].outputs[0].text}")
    print(f"⏱️ 두 번째 추론 소요 시간: {end_time2 - start_time2:.2f}초")

if __name__ == '__main__':
    main()