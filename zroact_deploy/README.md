# ZroAct 스트리밍 파이프라인 — 배포 패키지

Stage1 (YOWOv3 ONNX · GPU) + Stage2 (Qwen3.5-2B vLLM · GPU) 실시간 침입 탐지

---

## 1. 패키지 구성

```
zroact_deploy/
├── code/                          ← 파이프라인 코드 (전부 복사)
│   ├── pipeline_ver2/
│   │   ├── streaming_pipeline.py  ← 메인 실행 파일
│   │   ├── realtime_pipeline.py   ← 서버 유틸 (import 용)
│   │   └── main.py                ← 공통 유틸
│   ├── serving/
│   │   ├── workers/
│   │   │   ├── stage1_launcher.py
│   │   │   ├── stage1_server.py
│   │   │   └── stage2_vllm_server.py
│   │   └── config.json.template   ← ※ 경로 수정 후 config.json 으로 저장
│   └── benchmark2/
│       └── prompts/
│           └── action_timev1.txt  ← VLM 프롬프트
├── wheels/
│   └── causal_conv1d-1.6.2.post1-cp311-cp311-linux_x86_64.whl
├── patches/
│   └── vllm_platforms_init.py     ← vLLM NVML 패치 (NVIDIA 드라이버 mismatch 대응)
├── requirements/
│   ├── stage1_requirements.txt
│   └── stage2_requirements.txt
├── setup_stage1_env.sh            ← Stage1 환경 자동 설치
├── setup_stage2_env.sh            ← Stage2 환경 자동 설치
└── README.md                      ← 이 파일
```

별도 전송 필요 (용량 큼):
```
models/
├── zroact-stage1/YOWOv3/yowov3.onnx                          ← 170MB
├── zroact-stage1/YOWOv3/config/cf/custom_shufflenet.yaml
├── zroact-stage1/YOWOv3/utils/{box.py, build_config.py}
├── zroact-stage1/ckpt/YOWOv3/checkpoint/M23/ema_epoch_9.pth  ← ~170MB
└── zroact-stage2/benchmark2/models/Qwen3.5-2B/               ← 4.3GB
```

---

## 2. 시스템 요구사항

| 항목 | 최소 | 검증된 환경 |
|------|------|------------|
| GPU | RTX 3090 (24GB) 이상 | A6000 (48GB) |
| CUDA | 12.x | 12.9 |
| cuDNN | 8.x | 8.9.x |
| RAM | 16GB 이상 | 64GB |
| 저장공간 | 25GB 이상 | - |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| conda/miniconda | 필수 | miniforge3 |
| ffmpeg | 필수 | yowov3 env 또는 시스템 PATH |

---

## 3. 설치 절차

### 3-1. 디렉터리 구성

```bash
# 대상 서버에서 실행
TARGET=/home/user            # ← 원하는 경로로 변경
DEPLOY=$TARGET/zroact_deploy

mkdir -p $TARGET/zroact-stage1/YOWOv3/utils
mkdir -p $TARGET/zroact-stage1/YOWOv3/config/cf
mkdir -p $TARGET/zroact-stage1/ckpt/YOWOv3/checkpoint/M23
mkdir -p $TARGET/zroact-stage2/pipeline_ver2
mkdir -p $TARGET/zroact-stage2/serving/workers
mkdir -p $TARGET/zroact-stage2/benchmark2/{models,prompts}
```

### 3-2. 파일 배치

```bash
# code/ → zroact-stage2/
cp -r $DEPLOY/code/pipeline_ver2/*    $TARGET/zroact-stage2/pipeline_ver2/
cp -r $DEPLOY/code/serving/workers/*  $TARGET/zroact-stage2/serving/workers/

# config.json 생성 (경로 수정 필수)
cp $DEPLOY/code/serving/config.json.template $TARGET/zroact-stage2/serving/config.json
# ↓ config.json 열어서 <TARGET>, <CONDA_ROOT> 를 실제 경로로 치환
#   예) sed -i "s|<TARGET>|/home/user|g; s|<CONDA_ROOT>|/home/user/miniconda3|g" \
#            $TARGET/zroact-stage2/serving/config.json

# 프롬프트
cp $DEPLOY/code/benchmark2/prompts/action_timev1.txt \
   $TARGET/zroact-stage2/benchmark2/prompts/

# 모델 파일 (별도 전송)
# yowov3.onnx, custom_shufflenet.yaml, utils/*.py, ema_epoch_9.pth
# Qwen3.5-2B/ 디렉터리 전체
```

### 3-3. Stage1 환경 설치

```bash
# conda env 가 이미 있으면 STAGE1_ENV_PYTHON 지정, 없으면 자동 생성
export STAGE1_ENV_PYTHON=/home/user/miniconda3/envs/yowov3/bin/python
bash $DEPLOY/setup_stage1_env.sh
```

### 3-4. Stage2 환경 설치

```bash
export STAGE2_ENV_PYTHON=/home/user/miniconda3/envs/qwen35/bin/python3
bash $DEPLOY/setup_stage2_env.sh
```

---

## 4. 실행 방법

```bash
cd $TARGET/zroact-stage2

# 1회차: 서버 시작 + 추론 + 서버 유지 (약 45초 startup)
/home/user/miniconda3/envs/qwen35/bin/python3 \
    pipeline_ver2/streaming_pipeline.py \
    --video /path/to/video.mp4 \
    --keep-servers

# 2회차 이후: 기존 서버 재사용 (약 3-4초 완료)
/home/user/miniconda3/envs/qwen35/bin/python3 \
    pipeline_ver2/streaming_pipeline.py \
    --video /path/to/video.mp4 \
    --no-spawn

# 서버 종료
pkill -f stage1_launcher && pkill -f stage2_vllm_server
```

---

## 5. 트러블슈팅

### Stage1 GPU 모드 확인

서버 시작 로그에서 확인:
```
[Stage 1 Server] Loaded ONNX session | device: cuda | providers: ['CUDAExecutionProvider', ...]
```
`device: cpu` 가 출력되면 → onnxruntime-gpu 1.17.1 이 CUDA 12 빌드로 설치됐는지 확인

### Stage2 vLLM NVML 오류

```
pynvml.NVMLError_DriverNotLoaded: Driver Not Loaded
```
→ `patches/vllm_platforms_init.py` 가 `vllm/platforms/__init__.py` 에 적용됐는지 확인

### GPU OOM

```
Free memory on device cuda:0 (...) is less than desired GPU memory utilization
```
→ 스테일 프로세스 확인: `nvidia-smi` 실행 후 `kill <PID>`
→ config.json 에서 `gpu_memory_utilization` 값 낮추기 (예: 0.40)

### --no-spawn 실패

```
RuntimeError: Stage1 서버 응답 없음
```
→ 먼저 `--keep-servers` 로 실행하거나, 서버가 이미 실행 중인지 확인

---

## 6. 검증된 성능 (warm 서버 기준)

| 항목 | 값 |
|------|-----|
| 첫 번째 경보 | **0.59s** (영상 시작 1.8초 내) |
| 전체 스트리밍 루프 | **2.76s** (13.97초 영상) |
| 실시간 처리 비율 | **5배** (영상 길이 대비) |
| Stage1 | GPU (CUDAExecutionProvider) |
| Stage2 | GPU (vLLM V1, fp16) |
