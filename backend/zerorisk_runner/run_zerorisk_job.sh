#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:?video path required}"
RUN_ID="${2:?run id required}"

PIPELINE_DIR="${ZROACT_STREAMING_DIR:-/home/capstone2/zroact-stage2/pipeline_ver2}"
SEARCH_ROOT="${ZROACT_SEARCH_ROOT:-/home/capstone2}"
OUTPUT_ROOT="${ZERORISK_OUTPUT_ROOT:-/home/capstone2/backend/zerorisk_outputs}"
FINAL_DIR="${OUTPUT_ROOT}/${RUN_ID}/final"
MARKER="/tmp/zerorisk_${RUN_ID}_start.marker"

mkdir -p "${FINAL_DIR}"
touch "${MARKER}"

cd "${PIPELINE_DIR}"

if command -v conda >/dev/null 2>&1; then
  conda run -n qwen35 python streaming_pipeline.py --video "${VIDEO_PATH}" --no-spawn
else
  if [ -f /home/capstone2/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/capstone2/miniconda3/etc/profile.d/conda.sh
    conda activate qwen35
    python streaming_pipeline.py --video "${VIDEO_PATH}" --no-spawn
  else
    python streaming_pipeline.py --video "${VIDEO_PATH}" --no-spawn
  fi
fi

copy_newest_file() {
  local pattern="$1"
  local target_name="$2"
  local found

  found=$(find "${SEARCH_ROOT}" -type f -name "${pattern}" -newer "${MARKER}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)

  if [ -n "${found}" ] && [ -f "${found}" ]; then
    cp "${found}" "${FINAL_DIR}/${target_name}"
  fi
}

copy_newest_file risk_logs.jsonl risk_logs.jsonl
copy_newest_file status_logs.jsonl status_logs.jsonl
copy_newest_file timings.json timings.json
copy_newest_file risk_logs.csv risk_logs.csv
copy_newest_file status_logs.csv status_logs.csv
copy_newest_file timings.csv timings.csv
copy_newest_file result_summary.json result_summary.json

OVERLAY_DIR=$(find "${SEARCH_ROOT}" -type d -name overlay_images -newer "${MARKER}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)

if [ -n "${OVERLAY_DIR}" ] && [ -d "${OVERLAY_DIR}" ]; then
  rm -rf "${FINAL_DIR}/overlay_images"
  cp -a "${OVERLAY_DIR}" "${FINAL_DIR}/overlay_images"
fi

if [ ! -f "${FINAL_DIR}/risk_logs.jsonl" ]; then
  : > "${FINAL_DIR}/risk_logs.jsonl"
fi

cat > "${FINAL_DIR}/zerorisk_job_info.json" <<JSON
{
  "runId": "${RUN_ID}",
  "videoPath": "${VIDEO_PATH}",
  "condaEnv": "qwen35",
  "pipeline": "streaming_pipeline.py",
  "mode": "no-spawn",
  "searchRoot": "${SEARCH_ROOT}",
  "finalDir": "${FINAL_DIR}",
  "note": "Created by ZeroRisk wrapper around streaming_pipeline.py --no-spawn"
}
JSON

echo "ZeroRisk streaming AI wrapper completed."
echo "final=${FINAL_DIR}"
