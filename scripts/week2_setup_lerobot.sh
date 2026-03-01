#!/usr/bin/env bash
# Week 2: ROBOTIS fork LeRobot 설치 및 텔레오퍼레이션 테스트
# ⚠️ 원본 huggingface/lerobot이 아닌 ROBOTIS-GIT/lerobot 사용 필수
set -euo pipefail

echo "=== Week 2: LeRobot Setup & Teleoperation ==="

# --- 1. ROBOTIS fork LeRobot 설치 ---
echo "[1/4] Installing ROBOTIS fork LeRobot..."
LEROBOT_DIR="${HOME}/lerobot"

if [ ! -d "${LEROBOT_DIR}" ]; then
    # conda 환경 생성
    conda create -y -n lerobot python=3.10
    eval "$(conda shell.bash hook)"
    conda activate lerobot
    conda install -c conda-forge ffmpeg=6.1.1 -y

    # ROBOTIS fork 클론 및 설치
    git clone https://github.com/ROBOTIS-GIT/lerobot.git "${LEROBOT_DIR}"
    cd "${LEROBOT_DIR}"
    pip install -e ".[dynamixel]"
else
    echo "  LeRobot already installed at ${LEROBOT_DIR}"
    eval "$(conda shell.bash hook)"
    conda activate lerobot
    cd "${LEROBOT_DIR}"
fi

# --- 2. USB 포트 검색 ---
echo "[2/4] Finding OMX ports..."
echo "Run: lerobot-find-port"
echo "  Expected: Follower → /dev/ttyACM0, Leader → /dev/ttyACM1"
echo ""

# --- 3. 텔레오퍼레이션 테스트 명령어 ---
echo "[3/4] Teleoperation commands:"
cat << 'TELEOP_CMD'

# 기본 텔레오퍼레이션 (카메라 없이)
python -m lerobot.teleoperate \
    --robot.type=omx_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=omx_leader \
    --teleop.port=/dev/ttyACM1

# 카메라 포함 텔레오퍼레이션
python -m lerobot.teleoperate \
    --robot.type=omx_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=omx_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.cameras='[{type=opencv, key=cam1, index=0, width=640, height=480, fps=30}]'

TELEOP_CMD

# --- 4. 샘플 데이터 녹화 ---
echo "[4/4] Sample recording commands:"
cat << 'RECORD_CMD'

# HuggingFace 인증
huggingface-cli login
export HF_USER=$(huggingface-cli whoami | head -1)

# 5개 에피소드 샘플 녹화
lerobot-record \
    --dataset.repo_id=${HF_USER}/omx-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the red cube"

# 제어: → 에피소드 완료, ← 재녹화, ESC 종료

RECORD_CMD

echo "=== Week 2 Setup Complete ==="
