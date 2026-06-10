import argparse
import sys
import torch
import numpy as np
import onnxruntime

# Make sure we can import from parent directory
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config

def main():
    parser = argparse.ArgumentParser(description="Verify PyTorch vs ONNX output equivalence")
    parser.add_argument('--config', type=str, required=True, help="Path to YOWOv3 config yaml")
    parser.add_argument('--onnx', type=str, required=True, help="Path to exported ONNX model file")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    config = build_config(args.config)
    
    # We want to make sure it loads the pretrain path during initialization
    print(f"Weights path configured: {config.get('pretrain_path')}")

    print("Building PyTorch model...")
    pytorch_model = build_yowov3(config)
    pytorch_model.eval()

    print(f"Loading ONNX model from {args.onnx}...")
    # Use CPU provider for numeric comparison to avoid driver mismatches
    providers = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(args.onnx, providers=providers)

    # Generate dummy input of shape [batch_size=1, channels=3, clip_length=16, height=224, width=224]
    clip_length = config.get('clip_length', 16)
    img_size = config.get('img_size', 224)
    dummy_input = torch.randn(1, 3, clip_length, img_size, img_size)
    print(f"Generated dummy input of shape: {dummy_input.shape}")

    print("Running PyTorch inference...")
    with torch.no_grad():
        py_output = pytorch_model(dummy_input)
    
    # Convert PyTorch tensor to numpy
    py_output_np = py_output.cpu().numpy()
    print(f"PyTorch output shape: {py_output_np.shape}")

    print("Running ONNX Runtime inference...")
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output_np = ort_outputs[0]
    print(f"ONNX output shape: {onnx_output_np.shape}")

    # Verify shapes match
    if py_output_np.shape != onnx_output_np.shape:
        print(f"FAIL: Output shapes do not match! PyTorch: {py_output_np.shape}, ONNX: {onnx_output_np.shape}")
        sys.exit(1)

    print("Comparing outputs element-wise...")
    try:
        # We check with a relative tolerance (rtol) and absolute tolerance (atol) of 1e-3
        np.testing.assert_allclose(py_output_np, onnx_output_np, rtol=1e-3, atol=1e-3)
        print("SUCCESS: PyTorch and ONNX outputs are mathematically equivalent!")
    except AssertionError as e:
        print("FAIL: Outputs do not match within tolerance!")
        print(e)
        
        # Print some stats for debugging
        diff = np.abs(py_output_np - onnx_output_np)
        print(f"Max absolute difference: {np.max(diff)}")
        print(f"Mean absolute difference: {np.mean(diff)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
