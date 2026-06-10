#!/bin/bash
# Stage2 (Qwen3.5-2B vLLM) 환경 설치 스크립트
# 기존 환경과 충돌 방지: 새 env 'zroact_s2' 생성 (기존 qwen35 건드리지 않음)
# CUDA 12.1 호환 (Driver 535+, cu121 stable vLLM)
#
# 사용법:
#   CONDA_ROOT=/home/user/miniconda3
#   ENV_NAME=zroact_s2       # (선택) 다른 env명 사용시
#   bash setup_stage2_env.sh

set -e

CONDA_ROOT="${CONDA_ROOT:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
ENV_NAME="${ENV_NAME:-zroact_s2}"
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python3"

echo "========================================"
echo " Stage2 환경 설치 (새 env, 기존 영향 없음)"
echo " env name : $ENV_NAME"
echo " CUDA     : 12.1 호환 빌드 (stable vLLM)"
echo " ※ 총 소요시간: 10~20분"
echo "========================================"

# ── Step 1: conda env 생성 ────────────────────────────
echo ""
echo "[1/5] conda env '$ENV_NAME' 생성 (Python 3.11)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  이미 존재 → 스킵 (재설치하려면 conda env remove -n $ENV_NAME 먼저 실행)"
else
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "  완료"
fi

PIP="$PYTHON -m pip"

# ── Step 2: PyTorch stable (CUDA 12.1) ───────────────
echo ""
echo "[2/5] PyTorch stable (CUDA 12.1) 설치..."
echo "  (약 2~5분 소요)"
$PIP install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q
echo "  완료"

# ── Step 3: vLLM stable (CUDA 12.1 호환) ────────────
echo ""
echo "[3/5] vLLM stable 설치 (cu121 호환)..."
echo "  (약 5~10분 소요)"
# stable vLLM은 cu121 wheel을 자동으로 선택함
$PIP install vllm -q
echo "  완료"

# ── Step 4: aiohttp (pipeline 비동기 통신용) ─────────
echo ""
echo "[4/5] aiohttp 설치..."
$PIP install "aiohttp>=3.9.0" -q
echo "  완료"

# ── Step 5: vLLM NVML 패치 (필요시) ─────────────────
echo ""
echo "[5/5] vLLM NVML 패치 확인..."
DEPLOY_ROOT="$(cd "$(dirname "$0")" && pwd)"
VLLM_PATCH="$DEPLOY_ROOT/patches/vllm_platforms_init.py"
VLLM_PLATFORM=$("$PYTHON" -c \
    "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__),'platforms','__init__.py'))" 2>/dev/null || echo "")

if [ -z "$VLLM_PLATFORM" ]; then
    echo "  [SKIP] vllm import 실패 — 설치 확인 필요"
elif [ ! -f "$VLLM_PATCH" ]; then
    echo "  [SKIP] 패치 파일 없음 ($VLLM_PATCH)"
else
    # CUDA 12.1 환경에서는 NVML 오류가 없을 수 있음 — 먼저 테스트 후 필요시 적용
    echo "  패치 대상: $VLLM_PLATFORM"
    echo "  NVML 오류('pynvml.NVMLError_DriverNotLoaded') 발생시 아래 명령 실행:"
    echo "    cp $VLLM_PATCH $VLLM_PLATFORM"
    echo "  (CUDA 12.1 환경에서는 불필요할 수 있어 자동 적용 안 함)"
fi

# ── 검증 ─────────────────────────────────────────────
echo ""
echo "========================================"
echo " 설치 검증"
echo "========================================"
$PYTHON -c "
import torch, vllm
print(f'  torch   : {torch.__version__}')
print(f'  vllm    : {vllm.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
print(f'  GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"없음\"}')"

echo ""
echo "Stage2 환경 설치 완료 (env: $ENV_NAME)"
echo "config.json 에서 stage2_python 경로:"
echo "  $PYTHON"
