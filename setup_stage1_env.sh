#!/bin/bash
# Stage1 (YOWOv3 ONNX) 환경 설치 스크립트
# 기존 환경과 충돌 방지: 새 env 'zroact_s1' 생성 (기존 yowov3 건드리지 않음)
# CUDA 12.1 호환 (cuDNN 8.x, Driver 535+)
#
# 사용법:
#   CONDA_ROOT=/home/user/miniconda3   # conda 설치 경로
#   ENV_NAME=zroact_s1                 # (선택) 다른 env명 사용시
#   bash setup_stage1_env.sh

set -e

CONDA_ROOT="${CONDA_ROOT:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
ENV_NAME="${ENV_NAME:-zroact_s1}"
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

echo "========================================"
echo " Stage1 환경 설치 (새 env, 기존 영향 없음)"
echo " env name : $ENV_NAME"
echo " CUDA     : 12.1 호환 빌드"
echo "========================================"

# ── Step 1: conda env 생성 ────────────────────
echo ""
echo "[1/5] conda env '$ENV_NAME' 생성 (Python 3.10)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  이미 존재 → 스킵 (재설치하려면 conda env remove -n $ENV_NAME 먼저 실행)"
else
    conda create -n "$ENV_NAME" python=3.10 -y
    echo "  완료"
fi

PIP="$PYTHON -m pip"

# ── Step 2: PyTorch (CUDA 12.1) ──────────────────────
echo ""
echo "[2/5] PyTorch 2.1.2 (CUDA 12.1) 설치..."
$PIP install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121 -q
echo "  완료"

# ── Step 3: onnxruntime-gpu 1.17.1 (CUDA 12 + cuDNN 8 호환) ──
echo ""
echo "[3/5] onnxruntime-gpu 1.17.1 설치 (cuDNN 8 호환)..."
$PIP install onnxruntime-gpu==1.17.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ -q
echo "  완료"

# ── Step 4: numpy 호환성 픽스 ────────────────────────
echo ""
echo "[4/5] numpy<2 설치 (onnxruntime 1.17 호환)..."
$PIP install "numpy<2" -q
echo "  완료"

# ── Step 5: 서빙/유틸 패키지 ─────────────────────────
echo ""
echo "[5/5] 서빙 패키지 설치..."
$PIP install \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.29.0" \
    "requests>=2.31.0" \
    "Pillow>=10.0.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.66.0" \
    -q
echo "  완료"

# ── 검증 ─────────────────────────────────────────────
echo ""
echo "========================================"
echo " 설치 검증"
echo "========================================"
$PYTHON -c "
import onnxruntime as ort
import torch
print(f'  onnxruntime : {ort.__version__}')
print(f'  torch       : {torch.__version__}')
print(f'  CUDA 사용   : {torch.cuda.is_available()}')
print(f'  providers   : {ort.get_available_providers()}')
"

echo ""
echo "Stage1 환경 설치 완료 (env: $ENV_NAME)"
echo "config.json 에서 stage1_python 경로:"
echo "  $PYTHON"
