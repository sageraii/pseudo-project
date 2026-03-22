# OMX Digital Twin - Isaac Sim 5.1.0

실물 OMX_F 매니퓰레이터를 Isaac Sim에서 실시간 미러링하는 디지털 트윈.

## 요구사항

### 하드웨어
- NVIDIA GPU: RTX 2070+ (최소), RTX 4080+ (권장)
- OMX_F 팔로워 + OMX_L 리더
- USB 카메라 (UVC 호환)

### 소프트웨어
- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.11
- Docker (open_manipulator용)

## 1단계: Isaac Sim 설치

```bash
# uv가 없으면 먼저 설치
# curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/claude/isaacsim
uv venv --python 3.11
source .venv/bin/activate
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install pyyaml

# EULA 동의 (최초 1회)
python -c "import isaacsim"
# "Yes" 입력
```

> 첫 실행 시 extension 다운로드에 10분 이상 걸릴 수 있습니다.

## 2단계: open_manipulator 준비

```bash
# 저장소 clone (이미 있으면 생략)
cd ~/workspace
git clone -b main https://github.com/ROBOTIS-GIT/open_manipulator.git

# Docker 이미지 빌드 (최초 1회)
cd ~/workspace/open_manipulator/docker
docker compose build

# 환경변수 설정 (~/.bashrc에 추가 권장)
echo "export OPEN_MANIPULATOR_PATH=$HOME/workspace/open_manipulator" >> ~/.bashrc
source ~/.bashrc
```

> 주의: 반드시 **main 브랜치** (v4.1.2 이상)를 사용하세요. jazzy 브랜치는 그리퍼 반전 버그가 있습니다.

## 3단계: 로봇 연결 및 시작

### 터미널 1 — open_manipulator Docker

```bash
cd ~/workspace/open_manipulator/docker

# Docker 컨테이너 시작
./container.sh start

# 컨테이너 진입
./container.sh enter

# 텔레오퍼레이션 시작 (리더 + 팔로워)
ros2 launch open_manipulator_bringup omx_ai.launch.py
```

정상 시작 시 다음 로그가 표시됩니다:
```
[INFO] Successful initialization of hardware 'OMXFSystem'
[INFO] Successful initialization of hardware 'OMXLEADERSYSTEM'
[INFO] Successfully switched controllers!
```

### 터미널 2 — ROS 2 토픽 확인 (선택)

```bash
export ROS_DOMAIN_ID=30
source /opt/ros/jazzy/setup.bash

# joint_states 토픽이 보이는지 확인
ros2 topic list | grep joint_states
# 출력: /joint_states

# 데이터가 들어오는지 확인
ros2 topic hz /joint_states
# 출력: average rate: 100.xxx Hz
```

## 4단계: Isaac Sim 디지털 트윈 실행

### 터미널 3 — Isaac Sim

```bash
# 환경변수 설정
export OPEN_MANIPULATOR_PATH=~/workspace/open_manipulator
export ROS_DOMAIN_ID=30
source /opt/ros/jazzy/setup.bash

# Isaac Sim venv 활성화
cd ~/claude/isaacsim && source .venv/bin/activate

# 디지털 트윈 실행
python ~/claude/pseudo-project/scripts/isaac_sim/omx_digital_twin.py
```

정상 실행 시 다음 메시지가 표시됩니다:
```
Config loaded. Robot: omx_f, ROS_DOMAIN_ID: 30
Robot imported at: /omx_f
World created. physics_dt=0.01s, rendering_dt=0.0333s
OmniGraph configured: SubscribeJointState -> ArticulationController
============================================================
OMX Digital Twin running.
  Subscribing to: /joint_states
  Publishing to:  /isaac_joint_states
  ROS_DOMAIN_ID:  30
  Press Ctrl+C or close the window to stop.
============================================================
```

Isaac Sim 창에 OMX_F 로봇 모델이 나타나고, 리더를 조작하면 실물 로봇과 시뮬레이션 로봇이 동시에 움직입니다.

## 5단계: 종료

```bash
# Isaac Sim: Ctrl+C 또는 창 닫기
# open_manipulator Docker: Ctrl+C 후
cd ~/workspace/open_manipulator/docker
./container.sh stop
```

## 트러블슈팅

### Isaac Sim에서 로봇이 움직이지 않을 때

1. **ROS_DOMAIN_ID 확인**
   ```bash
   # 호스트에서
   echo $ROS_DOMAIN_ID  # 30이어야 함
   # Docker 안에서
   echo $ROS_DOMAIN_ID  # 30이어야 함
   ```

2. **토픽 확인**
   ```bash
   ros2 topic list  # /joint_states가 보여야 함
   ros2 topic hz /joint_states  # 100Hz 수신 확인
   ```

3. **ROS 2 환경 소싱 확인**
   ```bash
   source /opt/ros/jazzy/setup.bash  # 빠뜨리면 토픽 수신 불가
   ```

### URDF import 실패 시

1. **환경변수 확인**
   ```bash
   echo $OPEN_MANIPULATOR_PATH
   ls $OPEN_MANIPULATOR_PATH/open_manipulator_description/urdf/omx_f/omx_f.urdf
   ```

2. **메시 파일 존재 확인**
   ```bash
   ls $OPEN_MANIPULATOR_PATH/open_manipulator_description/meshes/omx_f/
   ```

### Dynamixel 하드웨어 에러

```
[WriteItem][ID:0XX] RX_PACKET_ERROR: Hardware error occurred
```
→ 로봇 전원을 껐다 켜거나, `ros2 launch`를 재시작하면 해결됩니다.

## 설정 변경

`config/omx_digital_twin.yaml`에서 수정:

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `ros2.domain_id` | 30 | Docker와 일치해야 함 |
| `ros2.joint_state_topic` | joint_states | 실물 로봇 관절 상태 토픽 |
| `ros2.isaac_joint_state_topic` | isaac_joint_states | 시뮬레이션 상태 퍼블리시 토픽 |
| `urdf_import.urdf_path` | `${OPEN_MANIPULATOR_PATH}/...` | URDF 파일 경로 |
| `physics.timestep` | 0.01 | 물리 dt (100Hz) |
| `rendering.fps` | 30 | 렌더링 fps |

## 다른 사용자 환경에서 사용 시

`OPEN_MANIPULATOR_PATH` 환경변수만 본인의 경로로 설정하면 됩니다:

```bash
export OPEN_MANIPULATOR_PATH=/path/to/your/open_manipulator
```

URDF 내 `package://` 메시 경로는 스크립트가 config의 `package_paths` 매핑을 기반으로 자동 치환합니다.

## 관련 문서

- [설계 Spec](../../docs/specs/2026-03-22-omx-digital-twin-design.md)
- [분산 모방학습 가이드](../../docs/OMX_DISTRIBUTED_IMITATION_LEARNING_GUIDE.md)
- [ROBOTIS OMX 공식 문서](https://ai.robotis.com/omx/introduction_omx.html)
- [Isaac Sim ROS 2 튜토리얼](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/index.html)
