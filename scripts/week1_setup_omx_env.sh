#!/usr/bin/env bash
# Week 1: OMX 하드웨어 환경 및 Physical AI Tools Docker 구축
# 사전 요구사항: Ubuntu 22.04+, Docker Engine, NVIDIA Container Toolkit
set -euo pipefail

echo "=== Week 1: OMX Environment Setup ==="

# --- 1. Docker & NVIDIA Container Toolkit 확인 ---
echo "[1/5] Checking Docker and NVIDIA Container Toolkit..."
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not installed. Install Docker Engine first."
    echo "  https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "ERROR: NVIDIA Container Toolkit not working."
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

sudo usermod -aG docker "$USER"
echo "Docker + NVIDIA GPU verified."

# --- 2. Open Manipulator 컨테이너 ---
echo "[2/5] Setting up Open Manipulator container..."
WORKSPACE_DIR="${HOME}/omx_workspace"
mkdir -p "${WORKSPACE_DIR}"
cd "${WORKSPACE_DIR}"

if [ ! -d "open_manipulator" ]; then
    git clone https://github.com/ROBOTIS-GIT/open_manipulator
fi
cd open_manipulator/docker
./container.sh start
echo "Open Manipulator container started."

# --- 3. Physical AI Tools 컨테이너 ---
echo "[3/5] Setting up Physical AI Tools container..."
cd "${WORKSPACE_DIR}"
if [ ! -d "physical_ai_tools" ]; then
    git clone --recurse-submodules https://github.com/ROBOTIS-GIT/physical_ai_tools.git
fi
cd physical_ai_tools/docker
./container.sh start
echo "Physical AI Tools container started."

# --- 4. USB 포트 확인 ---
echo "[4/5] Checking USB devices..."
echo "Connected serial devices:"
ls -al /dev/serial/by-id/ 2>/dev/null || echo "  No serial devices found. Connect OMX leader/follower."
echo ""
echo "NOTE: Enter the container to configure port settings:"
echo "  ./container.sh enter"
echo "  # Edit launch files with your serial IDs:"
echo "  #   omx_l_leader_ai.launch.py  → Leader serial ID"
echo "  #   omx_f_follower_ai.launch.py → Follower serial ID"

# --- 5. 카메라 확인 ---
echo "[5/5] Checking cameras..."
CAMERA_DEVICES=$(ls /dev/video* 2>/dev/null || true)
if [ -z "${CAMERA_DEVICES}" ]; then
    echo "  No camera devices found. Connect USB camera."
else
    echo "  Found cameras: ${CAMERA_DEVICES}"
fi

echo ""
echo "=== Week 1 Setup Complete ==="
echo "Next steps:"
echo "  1. Assemble OMX-L (leader) and OMX-F (follower)"
echo "  2. Connect USB cables and verify serial IDs"
echo "  3. Configure camera topic in omx_f_config.yaml"
echo "  4. Test homing: ros2 launch ... omx_f_follower_ai.launch.py"
