#!/usr/bin/env bash
# Week 5: GR00T N1.6 파인튜닝 (OMX 6-dim)
# 사전 조건: Week 4의 데이터 변환 완료, configs/omx_modality_config.py 존재
set -euo pipefail

echo "=== Week 5: GR00T N1.6 Fine-tuning for OMX ==="

# --- 설정 ---
GROOT_DIR="${HOME}/Isaac-GR00T"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="${PROJECT_DIR}/data/omx_groot_v2"
MODALITY_CONFIG="${PROJECT_DIR}/configs/omx_modality_config.py"
OUTPUT_DIR="${PROJECT_DIR}/outputs/groot_omx"

# GPU 설정 (기본: 단일 GPU)
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

# --- 검증 ---
echo "[1/3] Validating prerequisites..."

if [ ! -d "${GROOT_DIR}" ]; then
    echo "ERROR: Isaac-GR00T not found at ${GROOT_DIR}"
    echo "  git clone https://github.com/NVIDIA/Isaac-GR00T ${GROOT_DIR}"
    exit 1
fi

if [ ! -f "${MODALITY_CONFIG}" ]; then
    echo "ERROR: Modality config not found: ${MODALITY_CONFIG}"
    exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset not found: ${DATASET_DIR}"
    echo "  Run week4_convert_omx_to_groot.py first."
    exit 1
fi

# VRAM 확인
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "  GPU memory: ${GPU_MEM:-unknown} MiB"
echo "  Dataset: ${DATASET_DIR}"
echo "  Config: ${MODALITY_CONFIG}"
echo "  Output: ${OUTPUT_DIR}"

# --- 파인튜닝 실행 ---
echo ""
echo "[2/3] Starting GR00T fine-tuning..."
echo "  Base model: nvidia/GR00T-N1.6-3B"
echo "  Embodiment: NEW_EMBODIMENT (OMX 6-dim)"
echo "  Steps: ${MAX_STEPS}, Batch: ${BATCH_SIZE}, LR: ${LEARNING_RATE}"
echo ""

cd "${GROOT_DIR}"

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path "${DATASET_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --max-steps "${MAX_STEPS}" \
    --global-batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-steps "${SAVE_STEPS}" \
    --save-total-limit 3 \
    --tune-llm false \
    --tune-visual false \
    --tune-projector true \
    --tune-diffusion-model true \
    --use-wandb false

echo ""
echo "[3/3] Fine-tuning complete."
echo "  Checkpoints: ${OUTPUT_DIR}/"
echo ""
echo "Next: Run week6_eval_omx.py to compare ACT vs GR00T on OMX"
