import time
import numpy as np
import onnxruntime

def main():
    onnx_model_path = "yowov3.onnx"
    print(f"Loading ONNX model from {onnx_model_path}...")
    
    # Initialize session with CUDA execution provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    
    # Get active providers
    active_providers = ort_session.get_providers()
    print(f"Active execution providers: {ort_session.get_provider_options().keys()}")
    
    # Generate a dummy input of batch size 8
    # Shape: [batch_size=8, channels=3, clip_length=16, height=224, width=224]
    batch_size = 8
    clip_length = 16
    img_size = 224
    dummy_input = np.random.randn(batch_size, 3, clip_length, img_size, img_size).astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input}

    print("\n--- Running CUDA Warm-up (1st run) ---")
    start_warmup = time.perf_counter()
    _ = ort_session.run(None, ort_inputs)
    warmup_time = time.perf_counter() - start_warmup
    print(f"CUDA Warm-up (First Run) Time: {warmup_time:.4f} seconds ({warmup_time * 1000:.1f} ms)")

    print("\n--- Running Benchmark (50 iterations of batch size 8) ---")
    iterations = 50
    latencies = []

    for idx in range(iterations):
        start_iter = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        iter_time = time.perf_counter() - start_iter
        latencies.append(iter_time)

    latencies_ms = np.array(latencies) * 1000
    avg_batch_ms = np.mean(latencies_ms)
    std_batch_ms = np.std(latencies_ms)
    min_batch_ms = np.min(latencies_ms)
    max_batch_ms = np.max(latencies_ms)

    # Calculate per-clip time (batch size is 8)
    avg_clip_ms = avg_batch_ms / batch_size
    
    # Calculate FPS: (batch_size * clip_length) frames processed per batch duration
    # Each clip has 16 frames
    total_frames_per_batch = batch_size * clip_length
    achieved_fps = total_frames_per_batch / (avg_batch_ms / 1000.0)

    print("\n=== BENCHMARK STATISTICS ===")
    print(f"Total batches processed: {iterations}")
    print(f"Batch Size:              {batch_size}")
    print(f"Average Batch Latency:   {avg_batch_ms:.2f} ms  (Std: {std_batch_ms:.2f} ms)")
    print(f"Min Batch Latency:       {min_batch_ms:.2f} ms")
    print(f"Max Batch Latency:       {max_batch_ms:.2f} ms")
    print("----------------------------------------")
    print(f"Average Latency per Clip: {avg_clip_ms:.2f} ms")
    print(f"Model Processing Speed:   {achieved_fps:.1f} FPS (Frames Per Second)")
    print("========================================\n")

if __name__ == "__main__":
    main()
