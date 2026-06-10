#!/bin/bash
# ZroAct 최소 배포 셋업 스크립트
#
# 전제:
#   - yowov3 conda env 존재 (Stage1 YOWOv3용)
#   - qwen35 conda env 존재 (Stage2 vLLM용)
#   - Qwen3.5-2B 모델 이미 존재
#
# 추가 공간 사용량:
#   - yowov3: onnxruntime-gpu 1.17.1 (~170MB) + numpy downgrade (거의 0)
#   - qwen35: 파일 복사만 (0 bytes)
#
# 사용법:
#   TARGET_ROOT=/home/capstone2 \
#   CONDA_ROOT=/home/capstone2/miniconda3 \
#   QWEN_MODEL_PATH=/path/to/Qwen3.5-2B \
#   bash setup_on_target.sh

set -e

# ════════════════════════════════════════════
#  환경 변수 (수정 필요)
# ════════════════════════════════════════════
TARGET_ROOT="${TARGET_ROOT:-/home/capstone2}"
CONDA_ROOT="${CONDA_ROOT:-$TARGET_ROOT/miniconda3}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$TARGET_ROOT/zroact-stage2/benchmark2/models/Qwen3.5-2B}"

# 기존 env 이름 (변경 불필요시 그대로)
S1_ENV="${S1_ENV:-yowov3}"
S2_ENV="${S2_ENV:-qwen35}"

STAGE2_ROOT="$TARGET_ROOT/zroact-stage2"
STAGE1_ROOT="$TARGET_ROOT/zroact-stage1/YOWOv3"

S1_PYTHON="$CONDA_ROOT/envs/$S1_ENV/bin/python"
S2_PYTHON="$CONDA_ROOT/envs/$S2_ENV/bin/python3"

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo " ZroAct 최소 배포 셋업"
echo " TARGET_ROOT     : $TARGET_ROOT"
echo " CONDA_ROOT      : $CONDA_ROOT"
echo " QWEN_MODEL_PATH : $QWEN_MODEL_PATH"
echo " Stage1 env      : $S1_ENV  ($S1_PYTHON)"
echo " Stage2 env      : $S2_ENV  ($S2_PYTHON)"
echo " 추가 공간 사용  : ~170MB (onnxruntime만)"
echo "========================================"

# ── Step 1: env 존재 확인 ────────────────────
echo ""
echo "[0/5] 기존 env 확인..."
if ! "$S1_PYTHON" --version &>/dev/null; then
    echo "  [ERROR] Stage1 python 없음: $S1_PYTHON"
    echo "  S1_ENV 변수를 올바른 env 이름으로 설정하세요."
    exit 1
fi
if ! "$S2_PYTHON" --version &>/dev/null; then
    echo "  [ERROR] Stage2 python 없음: $S2_PYTHON"
    echo "  S2_ENV 변수를 올바른 env 이름으로 설정하세요."
    exit 1
fi
echo "  Stage1: $($S1_PYTHON --version)"
echo "  Stage2: $($S2_PYTHON --version)"

# ── Step 1: 디렉터리 생성 ────────────────────
echo ""
echo "[1/5] 디렉터리 생성..."
mkdir -p "$STAGE2_ROOT/pipeline_ver2"
mkdir -p "$STAGE2_ROOT/serving/workers"
mkdir -p "$STAGE2_ROOT/benchmark2/prompts"
mkdir -p "$TARGET_ROOT/zroact-stage1/ckpt"
echo "  완료"

# ── Step 2: 코드 파일 배치 ───────────────────
echo ""
echo "[2/5] 코드 파일 배치..."
cp "$DEPLOY_DIR/code/pipeline_ver2/"*.py  "$STAGE2_ROOT/pipeline_ver2/"
cp "$DEPLOY_DIR/code/serving/workers/"*.py "$STAGE2_ROOT/serving/workers/"
cp "$DEPLOY_DIR/code/benchmark2/prompts/"*.txt "$STAGE2_ROOT/benchmark2/prompts/"
touch "$STAGE2_ROOT/serving/__init__.py" "$STAGE2_ROOT/serving/workers/__init__.py"
echo "  완료"

# ── Step 3: Stage1 모델 파일 확인 ────────────
echo ""
echo "[3/5] Stage1 모델 파일 확인..."
if [ ! -f "$STAGE1_ROOT/yowov3.onnx" ]; then
    echo "  [WARN] yowov3.onnx 없음"
    echo "  먼저 실행: tar -xzf zroact_deploy_models_stage1.tar.gz -C $TARGET_ROOT"
else
    echo "  yowov3.onnx 확인됨: $STAGE1_ROOT/yowov3.onnx"
fi

# ── Step 4: config.json 생성 ─────────────────
echo ""
echo "[4/5] config.json 생성..."
cat > "$STAGE2_ROOT/serving/config.json" << EOF
{
  "stage2_root":            "$STAGE2_ROOT",
  "stage1_root":            "$STAGE1_ROOT",
  "stage1_pretrain_path":   "$TARGET_ROOT/zroact-stage1/ckpt/YOWOv3/checkpoint/M23/ema_epoch_9.pth",

  "stage1_python":          "$S1_PYTHON",
  "stage1_server_script":   "serving/workers/stage1_launcher.py",

  "stage2_python":          "$S2_PYTHON",
  "stage2_server_script":   "serving/workers/stage2_vllm_server.py",

  "stage2_ld_library_path_prepend": "",

  "vlm_model_path":         "$QWEN_MODEL_PATH",
  "prompt":                 "benchmark2/prompts/action_timev1.txt",

  "fps":                    30,
  "stage1_window":          16,
  "stage1_sample_rate":     10,
  "stage1_conf_threshold":  0.3,
  "stage1_batch_size":      8,
  "action_top_k":           2,
  "stage2_gap":             10,
  "stage2_stride":          30,

  "max_new_tokens":         256,
  "gpu_memory_utilization": 0.45,
  "max_model_len":          4096,

  "stage2_startup_timeout": 600,

  "stage1_daemon_host":     "127.0.0.1",
  "stage1_daemon_port":     8001,
  "stage2_daemon_host":     "127.0.0.1",
  "stage2_daemon_port":     8002
}
EOF
echo "  생성됨: $STAGE2_ROOT/serving/config.json"

# ── Step 5: 최소 패키지 수정 ─────────────────
echo ""
echo "[5/5] 최소 패키지 수정 (기존 env에 최소한만 추가)..."

# Stage1: onnxruntime-gpu 1.17.1 + numpy<2
# (~170MB 추가 다운로드, 기존 onnxruntime 대체)
echo "  [Stage1] onnxruntime-gpu 1.17.1 설치 (~170MB)..."
"$S1_PYTHON" -m pip install onnxruntime-gpu==1.17.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    -q
echo "  [Stage1] numpy<2 설치..."
"$S1_PYTHON" -m pip install "numpy<2" -q
echo "  Stage1 완료"

# Stage2: NVML 패치 (파일 복사, 0 bytes 추가)
VLLM_PATCH="$DEPLOY_DIR/patches/vllm_platforms_init.py"
VLLM_PLATFORM=$("$S2_PYTHON" -c \
    "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__),'platforms','__init__.py'))" 2>/dev/null || echo "")
if [ -n "$VLLM_PLATFORM" ] && [ -f "$VLLM_PLATFORM" ] && [ -f "$VLLM_PATCH" ]; then
    cp "$VLLM_PATCH" "$VLLM_PLATFORM"
    echo "  [Stage2] NVML 패치 적용: $VLLM_PLATFORM"
else
    echo "  [Stage2] NVML 패치 스킵 (vllm 없거나 패치 파일 없음 — 오류 발생시 수동 적용)"
fi

# ── 검증 ────────────────────────────────────
echo ""
echo "========================================"
echo " 검증"
echo "========================================"
"$S1_PYTHON" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'  Stage1 onnxruntime : {ort.__version__}')
print(f'  Stage1 providers   : {providers}')
cuda_ok = 'CUDAExecutionProvider' in providers
print(f'  Stage1 GPU 사용    : {\"YES\" if cuda_ok else \"NO (CPU fallback)\"}')
" 2>/dev/null || echo "  [WARN] Stage1 검증 실패"

"$S2_PYTHON" -c "
import vllm
print(f'  Stage2 vllm : {vllm.__version__}')
" 2>/dev/null || echo "  [WARN] Stage2 vllm import 실패"

echo ""
echo "========================================"
echo " 셋업 완료. 실행 방법:"
echo "========================================"
echo ""
echo "  cd $STAGE2_ROOT"
echo ""
echo "  # 1회차 (서버 시작 + 유지)"
echo "  $S2_PYTHON pipeline_ver2/streaming_pipeline.py \\"
echo "    --video /path/to/video.mp4 --keep-servers"
echo ""
echo "  # 2회차 이후 (서버 재사용)"
echo "  $S2_PYTHON pipeline_ver2/streaming_pipeline.py \\"
echo "    --video /path/to/video.mp4 --no-spawn"
