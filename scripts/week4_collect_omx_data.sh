#!/usr/bin/env bash
# Week 4: OMX 체계적 데이터 수집 (LeRobot CLI)
# Physical AI Tools Web UI는 별도 브라우저에서 http://localhost:7860 접속
set -euo pipefail

echo "=== Week 4: OMX Data Collection ==="

# HuggingFace 인증
if ! huggingface-cli whoami &>/dev/null; then
    echo "HuggingFace login required."
    huggingface-cli login
fi
export HF_USER=$(huggingface-cli whoami | head -1)
echo "HF User: ${HF_USER}"

# 설정
EPISODES_PER_TASK=50
FPS=15
ROBOT_PORT="/dev/ttyACM0"
TELEOP_PORT="/dev/ttyACM1"

# --- 작업별 데이터 수집 ---
TASKS=(
    "omx_pick:Pick up the object"
    "omx_place:Place the object on the target"
    "omx_stack:Stack the cubes"
)

for task_entry in "${TASKS[@]}"; do
    TASK_ID="${task_entry%%:*}"
    TASK_DESC="${task_entry#*:}"

    echo ""
    echo "=== Collecting: ${TASK_DESC} (${EPISODES_PER_TASK} episodes) ==="
    echo "Controls: → complete episode, ← retry, ESC stop"
    echo ""

    lerobot-record \
        --dataset.repo_id="${HF_USER}/${TASK_ID}" \
        --dataset.num_episodes="${EPISODES_PER_TASK}" \
        --dataset.single_task="${TASK_DESC}" \
        --dataset.fps="${FPS}" \
        --robot.type=omx_follower \
        --robot.port="${ROBOT_PORT}" \
        --teleop.type=omx_leader \
        --teleop.port="${TELEOP_PORT}" \
        --robot.cameras='[{type=opencv, key=cam1, index=0, width=640, height=480, fps=30}]'

    echo "Completed: ${TASK_ID}"
done

echo ""
echo "=== Data Collection Summary ==="
echo "Datasets uploaded to HuggingFace:"
for task_entry in "${TASKS[@]}"; do
    TASK_ID="${task_entry%%:*}"
    echo "  - https://huggingface.co/datasets/${HF_USER}/${TASK_ID}"
done
echo ""
echo "Next: Run week4_convert_omx_to_groot.py to convert to GR00T LeRobot v2 format"
