#!/usr/bin/env bash
# Week 5: ACT 베이스라인 학습 (ROBOTIS fork LeRobot)
# ACT는 OMX 공식 지원 정책, GR00T 비교 베이스라인
set -euo pipefail

echo "=== Week 5: ACT Baseline Training ==="

# --- 설정 ---
LEROBOT_DIR="${HOME}/lerobot"
export HF_USER="${HF_USER:-$(huggingface-cli whoami | head -1)}"

NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

# --- 작업별 ACT 학습 ---
TASKS=(
    "omx_pick"
    "omx_place"
    "omx_stack"
)

echo "HF User: ${HF_USER}"
echo "LeRobot: ${LEROBOT_DIR}"
echo "Epochs: ${NUM_EPOCHS}, Batch: ${BATCH_SIZE}"
echo ""

cd "${LEROBOT_DIR}"
eval "$(conda shell.bash hook)"
conda activate lerobot

for TASK in "${TASKS[@]}"; do
    REPO_ID="${HF_USER}/${TASK}"
    echo "=== Training ACT on: ${REPO_ID} ==="

    lerobot-train \
        --dataset.repo_id="${REPO_ID}" \
        --policy.type=act \
        --policy.device="${DEVICE}" \
        --training.num_epochs="${NUM_EPOCHS}" \
        --training.batch_size="${BATCH_SIZE}"

    echo "Completed: ${TASK}"
    echo "  Output: outputs/train/act_${TASK}/"
    echo "  Files: config.json + model.safetensors"
    echo ""
done

echo "=== ACT Baseline Training Complete ==="
echo ""
echo "Results:"
for TASK in "${TASKS[@]}"; do
    echo "  - outputs/train/act_${TASK}/config.json"
    echo "  - outputs/train/act_${TASK}/model.safetensors"
done
echo ""
echo "Next: Run week5_finetune_groot_omx.sh for GR00T fine-tuning"
