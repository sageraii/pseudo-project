# OMX Digital Twin Design Spec

> Isaac Sim 5.1.0에서 OMX 매니퓰레이터의 디지털 트윈을 구현하여 실물 로봇의 움직임을 실시간 미러링하고, 가상 카메라로 합성 이미지를 생성한다.

## 1. 목표

### A단계 (본 설계 범위)
- 실물 OMX_F 팔로워 로봇의 관절 상태를 Isaac Sim에서 실시간 미러링
- 가상 카메라로 시뮬레이션 영상 렌더링 (향후 합성 데이터 준비)

### B단계 (향후)
- 가상 카메라 이미지를 LeRobot 데이터셋 포맷으로 저장
- 도메인 랜덤화(조명, 배경, 텍스처) 적용

### C단계 (향후)
- 시뮬레이션에서 학습된 정책을 실물 로봇에 배포 (양방향 연동)

### 성공 기준
- 실물 로봇 움직임과 시뮬레이션 로봇 움직임의 시각적 일치 확인
- 미러링 지연 100ms 이하 (ROS2 DDS 네트워크 지연 + 렌더링 지연)
- 가상 카메라 640x480/30fps 안정 렌더링

## 2. 시스템 아키텍처

```
[open_manipulator Docker]              [Host: Isaac Sim 5.1.0]
(network_mode: host)                   (~/claude/isaacsim/.venv)

robot_state_publisher                  URDF Import (동기식)
  └─ /robot_description                  └─ URDFParseAndImportFile
     (파라미터 서비스)                      └─ package:// 수동 치환 후 import
                                           └─ OMX_F USD 모델 생성 → /World/omx_f

joint_state_broadcaster
  └─ /joint_states ────────────────▶ ROS2SubscribeJointState (OmniGraph)
     (sensor_msgs/JointState)          └─ jointNames 매핑 (ROS↔USD 순서 차이 대응)
     (100Hz)                           └─ IsaacArticulationController
                                          └─ 시뮬레이션 로봇 관절 구동

                                    PublishJointState
                                       └─ /isaac_joint_states (시뮬 상태 퍼블리시)

                                    가상 카메라 (640x480, 30fps)
                                       └─ 실물 카메라와 유사한 앵글
                                       └─ (향후 B단계) ROS2 토픽 퍼블리시
```

## 3. 환경 요구사항

### 소프트웨어

| 항목 | 버전 | 비고 |
|------|------|------|
| Isaac Sim | 5.1.0 (pip, `isaacsim[all,extscache]`) | `~/claude/isaacsim/.venv` |
| Python | 3.11 | uv로 관리 |
| open_manipulator | main v4.1.2+ | Docker (network_mode: host) |
| ROS 2 | Jazzy | Ubuntu 24.04 기준 |
| ROS_DOMAIN_ID | 30 | Docker와 호스트 일치 필수 |
| PyYAML | (any) | `uv pip install pyyaml` |

### 하드웨어

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | NVIDIA RTX 2070 (Isaac Sim 공식 최소) | RTX 4080+ (실시간 렌더링) |
| VRAM | 8GB | 12GB+ |
| OMX_F (팔로워) | Dynamixel ID 11-16 | |
| OMX_L (리더) | Dynamixel ID 1-6 | |
| USB 카메라 | UVC 호환, 640x480/30fps | |

### 사전 조건

```bash
# 1. open_manipulator Docker 실행 중
cd ~/workspace/open_manipulator/docker && ./container.sh start
# 컨테이너 진입 후:
ros2 launch open_manipulator_bringup omx_ai.launch.py

# 2. ROS_DOMAIN_ID 일치 확인
echo $ROS_DOMAIN_ID  # Docker: 30, Host에서도 30으로 설정

# 3. Isaac Sim venv 활성화
cd ~/claude/isaacsim && source .venv/bin/activate

# 4. ROS 2 환경 소싱 (Isaac Sim이 ROS2 토픽 접근하기 위해)
source /opt/ros/jazzy/setup.bash
```

## 4. OMX_F 로봇 명세

### 관절 구성

| Index | Joint Name | Type | Axis | Range | Dynamixel ID |
|-------|-----------|------|------|-------|--------------|
| 0 | joint1 | revolute | Z | -2pi ~ +2pi | 11 |
| 1 | joint2 | revolute | Y | -2pi ~ +2pi | 12 |
| 2 | joint3 | revolute | Y | -2pi ~ +2pi | 13 |
| 3 | joint4 | revolute | Y | -2pi ~ +2pi | 14 |
| 4 | joint5 | revolute | X | -2pi ~ +2pi | 15 |
| 5 | gripper_joint_1 | revolute | Z | -2pi ~ +2pi | 16 |
| 6 | gripper_joint_2 | revolute + mimic | Z | mimic gripper_joint_1 x -1 | - |

> gripper_joint_2는 URDF에서 `type="revolute"`이며, `<mimic>` 자식 요소로 mimic 관계를 정의한다. Isaac Sim의 `parse_mimic=True` 설정으로 자동 처리된다.

### ROS 2 토픽

| 토픽 | 메시지 타입 | 방향 | 주기 |
|------|------------|------|------|
| `/joint_states` | sensor_msgs/JointState | 로봇 → Isaac Sim | 100Hz |
| `/leader/joint_trajectory` | trajectory_msgs/JointTrajectory | 리더 → 팔로워 | 100Hz |
| `/camera1/image_raw/compressed` | sensor_msgs/CompressedImage | 카메라 → 서버 | 30fps |

### URDF 파일

```
~/workspace/open_manipulator/open_manipulator_description/
├── urdf/omx_f/omx_f.urdf          # 팔로워 (xacro에서 생성됨)
├── urdf/omx_l/omx_l.urdf          # 리더
└── meshes/omx_f/*.stl             # 메시 파일 (scale: 0.001)
```

URDF 내 메시 경로는 `package://open_manipulator_description/meshes/...` 형식.
Isaac Sim의 URDF importer는 `package://` 경로를 직접 해결하지 못하므로, 스크립트에서 절대 경로로 치환 후 import 한다 (Section 7 참조).

## 5. 구현 설계

### URDF Import 방식 결정

| 방식 | 장점 | 단점 |
|------|------|------|
| `URDFImportFromROS2Node` (5.1.0 신규) | `package://` 자동 해결 | **비동기 실행** — 스크립트에서 완료 시점 제어 어려움 |
| **`URDFParseAndImportFile` (채택)** | **동기 실행** — import 완료 후 바로 OmniGraph 구성 가능 | `package://` 수동 치환 필요 |

> **결정:** Standalone 스크립트에서는 동기식 `URDFParseAndImportFile`을 사용한다.
> `URDFImportFromROS2Node`는 비동기로 동작하여 (ROS2 서비스 호출 → 백그라운드 스레드 → 콜백), 스크립트에서 import 완료를 기다리는 것이 복잡하고 불안정하다.
> `package://` 경로 치환은 정규식으로 간단히 해결 가능하다.

### 파일 구조

```
~/claude/pseudo-project/scripts/isaac_sim/
├── README.md                          # 환경 설정 및 실행 방법
├── config/
│   └── omx_digital_twin.yaml          # 설정 파일
└── omx_digital_twin.py                # 메인 스크립트
```

### config/omx_digital_twin.yaml

```yaml
# ROS 2 설정
ros2:
  domain_id: 30
  joint_state_topic: "joint_states"
  # 주의: PublishJointState 기본 토픽이 "joint_states"이므로,
  # 아래 값을 변경하지 않으면 Subscribe와 Publish가 같은 토픽을 사용하여 피드백 루프 발생
  isaac_joint_state_topic: "isaac_joint_states"

# 로봇 설정
robot:
  name: "omx_f"
  # URDF import 후 생성되는 prim 경로 (URDFParseAndImportFile이 결정)
  # make_default_prim=True 시 stage root에 생성됨
  joint_names:
    - joint1
    - joint2
    - joint3
    - joint4
    - joint5
    - gripper_joint_1

# URDF Import 설정
urdf_import:
  # URDF 파일 경로 (환경변수 OPEN_MANIPULATOR_PATH 기반)
  urdf_path: "${OPEN_MANIPULATOR_PATH}/open_manipulator_description/urdf/omx_f/omx_f.urdf"
  # package:// 치환 매핑
  package_paths:
    open_manipulator_description: "${OPEN_MANIPULATOR_PATH}/open_manipulator_description"
  # ImportConfig 속성 (isaacsim.asset.importer.urdf._urdf.ImportConfig 객체에 설정)
  fix_base: true
  merge_fixed_joints: false          # world_fixed, end_effector_joint 보존
  parse_mimic: true                  # gripper_joint_2 mimic 자동 처리
  create_physics_scene: true
  make_default_prim: true
  import_inertia_tensor: true
  default_drive_type: 1              # 1 = position drive
  default_drive_strength: 1000.0
  default_position_drive_damping: 100.0

# 물리 시뮬레이션 설정
physics:
  timestep: 0.01                     # 10ms (100Hz, joint_states와 동기화)
  gravity: [0.0, 0.0, -9.81]        # m/s^2
  solver_type: "TGS"                 # Temporal Gauss-Seidel

# 가상 카메라 설정
camera:
  prim_path: "/World/Camera"
  resolution: [640, 480]
  fps: 30
  # 실물 카메라와 유사한 위치 (사용자 조정 가능)
  position: [0.5, 0.0, 0.4]         # x, y, z (meters)
  target: [0.0, 0.0, 0.15]          # look-at point
```

### omx_digital_twin.py 핵심 흐름

```python
import os
import re
import sys
import tempfile

import yaml

# ── 1. Isaac Sim 초기화 ──────────────────────────────────────────────
# SimulationApp은 반드시 다른 Omniverse import보다 먼저 인스턴스화해야 함
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ── 2. Omniverse/Isaac Sim import (SimulationApp 이후에만 가능) ──────
import omni
import omni.graph.core as og
import omni.kit.commands
import usdrt.Sdf
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

# ── 3. Extension 활성화 ──────────────────────────────────────────────
enable_extension("isaacsim.ros2.bridge")
# URDFParseAndImportFile 커맨드를 위해 URDF importer도 활성화
enable_extension("isaacsim.asset.importer.urdf")

simulation_app.update()  # extension 로드 대기

# ── 4. 설정 파일 로드 ────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "config", "omx_digital_twin.yaml")
with open(config_path) as f:
    raw = f.read()
raw = os.path.expandvars(raw)  # ${OPEN_MANIPULATOR_PATH} 등 환경변수 치환
config = yaml.safe_load(raw)

# ── 5. URDF Import (동기식) ──────────────────────────────────────────
# 5a. URDF 파일 읽기 + package:// 경로를 절대 경로로 치환
urdf_path = os.path.expandvars(config["urdf_import"]["urdf_path"])
with open(urdf_path, "r") as f:
    urdf_content = f.read()

for pkg_name, pkg_path in config["urdf_import"]["package_paths"].items():
    pkg_path = os.path.expandvars(pkg_path)
    urdf_content = re.sub(
        rf"package://{pkg_name}/",
        pkg_path + "/",
        urdf_content,
    )

# 5b. 치환된 URDF를 임시 파일로 저장
with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as tmp:
    tmp.write(urdf_content)
    resolved_urdf_path = tmp.name

# 5c. ImportConfig 생성 및 설정
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = config["urdf_import"]["merge_fixed_joints"]
import_config.fix_base = config["urdf_import"]["fix_base"]
import_config.make_default_prim = config["urdf_import"]["make_default_prim"]
import_config.create_physics_scene = config["urdf_import"]["create_physics_scene"]
import_config.import_inertia_tensor = config["urdf_import"]["import_inertia_tensor"]
import_config.parse_mimic = config["urdf_import"]["parse_mimic"]
import_config.default_drive_type = config["urdf_import"]["default_drive_type"]
import_config.default_drive_strength = config["urdf_import"]["default_drive_strength"]
import_config.default_position_drive_damping = config["urdf_import"]["default_position_drive_damping"]

# 5d. URDF Import 실행 (동기 — 완료 후 반환)
status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=resolved_urdf_path,
    import_config=import_config,
    dest_path="",                    # 빈 문자열 = 현재 stage에 직접 import
    get_articulation_root=True,      # articulation root prim 경로 반환
)

# 5e. 임시 파일 정리
os.unlink(resolved_urdf_path)

# import 실패 시 종료
if not robot_prim_path:
    print("ERROR: URDF import failed. Is the URDF path correct?")
    simulation_app.close()
    sys.exit(1)

print(f"Robot imported at: {robot_prim_path}")

# ── 6. World 생성 및 물리 설정 ───────────────────────────────────────
world = World(
    stage_units_in_meters=1.0,
    physics_dt=config["physics"]["timestep"],       # 0.01s = 100Hz
    rendering_dt=1.0 / config["camera"]["fps"],     # 1/30s ≈ 33ms
)
world.scene.add_default_ground_plane()

# ── 7. OmniGraph 구성 ───────────────────────────────────────────────
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ],
        og.Controller.Keys.CONNECT: [
            # Tick -> 모든 노드 실행
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            # ROS2 Context 연결
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            # 시간 동기화
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            # JointState 구독 -> ArticulationController
            # jointNames 연결은 필수: ROS JointState의 관절 순서와
            # USD articulation의 관절 순서가 다를 수 있으므로 이름 기반 매핑 필요
            ("SubscribeJointState.outputs:positionCommand",
             "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand",
             "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand",
             "ArticulationController.inputs:effortCommand"),
            ("SubscribeJointState.outputs:jointNames",
             "ArticulationController.inputs:jointNames"),
        ],
        og.Controller.Keys.SET_VALUES: [
            # ROS_DOMAIN_ID를 명시적으로 설정 (환경변수 의존 제거)
            ("Context.inputs:domain_id", config["ros2"]["domain_id"]),
            ("Context.inputs:useDomainIDEnvVar", False),
            # /joint_states 토픽 구독 (기본값이 "joint_command"이므로 반드시 변경!)
            ("SubscribeJointState.inputs:topicName",
             config["ros2"]["joint_state_topic"]),
            # 로봇 prim 경로 설정
            ("ArticulationController.inputs:targetPrim",
             [usdrt.Sdf.Path(robot_prim_path)]),
            ("PublishJointState.inputs:targetPrim",
             [usdrt.Sdf.Path(robot_prim_path)]),
            ("PublishJointState.inputs:topicName",
             config["ros2"]["isaac_joint_state_topic"]),
        ],
    },
)

# ── 8. 가상 카메라 생성 ──────────────────────────────────────────────
# Camera prim 추가 (향후 B단계에서 ROS2 퍼블리시 연결)
# 위치/방향은 config에서 로드

# ── 9. 시뮬레이션 루프 ──────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()

# world.reset()이 내부적으로 timeline.play()를 호출하므로
# timeline.play()를 별도로 호출할 필요 없음
world.reset()

try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    # ── 10. 정리 ────────────────────────────────────────────────────
    timeline.stop()
    simulation_app.close()
```

## 6. OmniGraph 노드 연결도

```
                          ┌─────────────┐
                          │ ROS2Context │
                          │ domain_id=30│
                          │ useDomainID │
                          │ EnvVar=False│
                          └──────┬──────┘
                                 │ context
                    ┌────────────┼────────────┐
                    ▼            │            ▼
         SubscribeJointState    │    PublishJointState
         topicName=             │    topicName=
         "joint_states"         │    "isaac_joint_states"
              │                 │         ▲
              │ positionCommand │         │
              │ velocityCommand │         │
              │ effortCommand   │         │
              │ jointNames      │         │
              ▼                 │         │
     ArticulationController    │    ReadSimTime
     targetPrim=               │         │
     {robot_prim_path}         │    simulationTime
                               │         │
                               │         ▼
┌──────────────┐               │    PublishJointState
│OnPlaybackTick├──tick─────────┼──▶ .inputs:timeStamp
│              ├──tick──▶ SubscribeJointState.inputs:execIn
│              ├──tick──▶ ArticulationController.inputs:execIn
│              └──tick──▶ PublishJointState.inputs:execIn
└──────────────┘
```

## 7. 메시 경로 해결 (다른 사용자를 위한 가이드)

### 문제

OMX_F URDF 파일 내 메시 경로가 ROS `package://` 형식으로 되어 있어 Isaac Sim의 URDF importer에서 직접 로드 불가:

```xml
<mesh filename="package://open_manipulator_description/meshes/omx_f/follower_01_base.stl"
      scale="0.001 0.001 0.001"/>
```

Isaac Sim의 `URDFParseAndImportFile`은 절대 파일 경로만 지원한다 (C++ 바인딩 레이어에서 `os.path.abspath()` 사용).

### 해결: 스크립트 내 자동 치환

본 스크립트는 URDF 파일을 읽어 `package://` 경로를 config에 정의된 절대 경로로 치환한 후 임시 파일로 저장하여 import 한다:

```python
# config에서 패키지 경로 매핑 로드
# package_paths:
#   open_manipulator_description: "/home/user/workspace/open_manipulator/open_manipulator_description"

for pkg_name, pkg_path in package_paths.items():
    urdf_content = re.sub(rf"package://{pkg_name}/", pkg_path + "/", urdf_content)
```

### 다른 사용자 환경 설정

```bash
# 환경변수로 open_manipulator 경로 지정
export OPEN_MANIPULATOR_PATH=~/workspace/open_manipulator

# 또는 config/omx_digital_twin.yaml에서 직접 수정:
# urdf_import:
#   urdf_path: "/절대경로/open_manipulator_description/urdf/omx_f/omx_f.urdf"
#   package_paths:
#     open_manipulator_description: "/절대경로/open_manipulator_description"
```

### 대안: URDFImportFromROS2Node (GUI/인터랙티브용)

Isaac Sim 5.1.0의 `URDFImportFromROS2Node`는 ROS 2의 `robot_state_publisher`에서 URDF를 가져와 `ament_index_python`으로 `package://` 경로를 자동 해결한다. 단, **비동기로 동작**하므로 standalone 스크립트보다는 GUI 워크플로우에 적합하다.

## 8. 주의사항

### ROS_DOMAIN_ID
- open_manipulator Docker: `ROS_DOMAIN_ID=30` (.bashrc에 설정)
- OmniGraph `ROS2Context` 노드에서 `domain_id=30`, `useDomainIDEnvVar=False`로 명시 설정
- 환경변수에 의존하지 않으므로, 스크립트 실행 시 `export ROS_DOMAIN_ID`를 잊어도 동작

### SubscribeJointState 토픽 이름
- 기본값이 `"joint_command"`임 (NOT `"joint_states"`)
- 반드시 `topicName`을 `"joint_states"`로 명시적 설정

### jointNames 연결 필수
- `SubscribeJointState.outputs:jointNames` → `ArticulationController.inputs:jointNames` 연결 필수
- ROS `/joint_states` 메시지의 관절 순서와 USD articulation의 관절 순서가 다를 수 있음
- 이름 기반 매핑으로 정확한 관절 대응 보장

### Isaac Sim ROS 2 환경
- 5.1.0에서는 Ubuntu 버전을 자동 감지하여 ROS distro 매핑 (24.04 → jazzy)
- 터미널에서 `source /opt/ros/jazzy/setup.bash` 필수

### URDF ImportConfig 설정
- `merge_fixed_joints=False`: `world_fixed`, `end_effector_joint` 보존 (5.1.0에서 관련 버그 수정됨)
- `parse_mimic=True`: `gripper_joint_2`의 mimic 관계 자동 처리
- `fix_base=True`: 로봇 베이스 고정
- `make_default_prim=True`: stage root의 default prim으로 설정
- ImportConfig는 `_urdf.ImportConfig()` C++ 바인딩 객체이며, Python 속성으로 설정

### 물리 시뮬레이션
- physics_dt = 0.01s (100Hz): `/joint_states` 100Hz와 동기화
- rendering_dt = 1/30s: 가상 카메라 30fps와 동기화
- 기본 중력: -9.81 m/s^2 (Z축)

### open_manipulator 브랜치
- 반드시 **main 브랜치 (v4.1.2 이상)** 사용
- jazzy v4.0.9는 그리퍼 Drive Mode 버그 있음 (이전 분석 참조)

### 에러 처리
- URDF import 실패 시 스크립트 종료 (prim 경로 None 체크)
- ROS2 토픽 미수신 시 시뮬레이션 로봇이 정지 상태 유지 (silent, 로그에서 확인)
- `KeyboardInterrupt`로 안전 종료 (timeline.stop() + simulation_app.close())

## 9. 실행 방법

```bash
# 1. open_manipulator Docker 시작 + 텔레오퍼레이션
cd ~/workspace/open_manipulator/docker
./container.sh start && ./container.sh enter
ros2 launch open_manipulator_bringup omx_ai.launch.py

# 2. 환경변수 설정 (별도 터미널)
export OPEN_MANIPULATOR_PATH=~/workspace/open_manipulator
export ROS_DOMAIN_ID=30
source /opt/ros/jazzy/setup.bash

# 3. Isaac Sim 디지털 트윈 실행
cd ~/claude/isaacsim && source .venv/bin/activate
python ~/claude/pseudo-project/scripts/isaac_sim/omx_digital_twin.py

# 4. 리더를 조작하면 실물 로봇 + Isaac Sim 로봇이 동시에 움직임

# 5. 종료: Ctrl+C 또는 Isaac Sim 창 닫기
```

## 10. 단계별 로드맵

### 진행 순서

```
[A단계] 디지털 트윈 미러링        ✅ 설계 + 구현 완료, 실물 테스트 필요
    │
    ▼
[A 검증] 실물 로봇 연결 테스트     ❌ 미수행
    │
    ▼
[B단계] 합성 데이터 생성           ❌ 미착수
    │
    ▼
[B 검증] 합성 데이터 학습 성능     ❌ 미착수
    │
    ▼
[C단계] 양방향 Sim2Real           ❌ 미착수
```

### A단계 검증 (B단계 진입 전 필수)

실물 로봇을 연결하여 다음을 확인해야 B단계로 진행 가능:

- [ ] Isaac Sim에 OMX_F 모델이 정상 import되는가 (메시, 관절 구조)
- [ ] `/joint_states` 토픽 수신 시 시뮬레이션 로봇이 실물과 동일하게 움직이는가
- [ ] 관절 방향이 일치하는가 (특히 gripper_joint_1 반전 없는지)
- [ ] 미러링 지연이 체감 가능 수준 이하인가 (목표: 100ms 이하)

### B단계: 합성 데이터 생성

**전제 조건:**

| 항목 | 설명 |
|------|------|
| A단계 실물 테스트 통과 | 미러링이 동작해야 카메라/물리 신뢰 가능 |
| 가상 카메라 구현 | `UsdGeom.Camera` 생성, 실물 카메라와 동일 위치/앵글 설정 |
| 도메인 랜덤화 설계 | Isaac Sim Replicator 활용 (조명, 배경, 텍스처 변경) |
| LeRobot 저장 파이프라인 | 가상 카메라 렌더링 → parquet + mp4 (LeRobot v2.1 포맷) |
| 품질 비교 기준 | 실물 데이터 vs 합성 데이터의 sim-real gap 측정 방법 정의 |

**구현 범위:**
- 가상 카메라 이미지를 LeRobot v2.1 데이터셋 포맷으로 저장
- Isaac Sim Replicator를 활용한 도메인 랜덤화
- `observation.images.camera1`와 동일한 640x480/30fps 규격
- 실물 데이터셋 (`omx_f_test`)과 동일한 스키마:
  - `observation.state`: [6] float32 (joint1~gripper_joint_1)
  - `action`: [6] float32
  - `observation.images.camera1`: 640x480x3 video

### C단계: 양방향 연동 (Sim2Real)

**전제 조건:**

| 항목 | 설명 |
|------|------|
| B단계 완료 | 합성 데이터로 학습된 정책 모델이 있어야 함 |
| 학습된 정책 체크포인트 | ACT/Diffusion 등으로 학습된 모델 파일 |
| Isaac Sim → 실물 명령 경로 | `ROS2PublishJointState` → `/leader/joint_trajectory` 연결 |
| Safety boundary 설계 | 관절 제한, 속도 제한, 충돌 감지 (실물 로봇 보호) |
| Sim 내 정책 추론 루프 | Isaac Sim에서 정책 로드 → observation 수집 → action 생성 → 퍼블리시 |

**구현 범위:**
- `ROS2PublishJointState`로 시뮬레이션 → 실물 명령 전송
- 학습된 정책을 Isaac Sim에서 실행, 검증 후 실물 배포
- Safety boundary 설정 (관절 제한, 충돌 감지)
- 시뮬레이션 성공률 → 실물 성공률 비교 (Sim2Real gap 측정)

## 11. 리뷰 반영 사항

spec review에서 지적된 사항과 반영 내역:

| 지적 사항 | 심각도 | 반영 |
|----------|--------|------|
| `URDFImportFromROS2Node`의 `dest_path`는 USD 파일 경로이나 prim 경로로 오용 | CRITICAL | 동기식 `URDFParseAndImportFile`로 변경, `dest_path=""` 사용 |
| `URDFImportFromROS2Node`가 비동기라 OmniGraph 설정 시 로봇 미존재 | CRITICAL | 동기식 import로 변경하여 문제 해소 |
| `isaacsim.ros2.urdf` extension 활성화 누락 | MAJOR | `isaacsim.asset.importer.urdf` 활성화 추가 |
| `ROS2Context`에 `domain_id` 미설정 | MAJOR | `domain_id=30`, `useDomainIDEnvVar=False` 명시 |
| `World` 객체 미생성 | MAJOR | `World()` 생성 + physics_dt/rendering_dt 설정 |
| cleanup/shutdown 미처리 | MAJOR | try/finally 블록으로 안전 종료 |
| `enable_extension` import 누락 | MAJOR | `from isaacsim.core.utils.extensions import enable_extension` 추가 |
| 물리 시뮬레이션 설정 누락 | MINOR | physics timestep, gravity, solver 설정 추가 |
| `usdrt` import 누락 | MINOR | import 목록에 추가 |

---

**작성일:** 2026-03-22
**리뷰일:** 2026-03-22 (critic agent review 반영)
**Isaac Sim:** 5.1.0 (pip, Python 3.11)
**검증 기반:** 5.1.0 extension 소스 코드 직접 분석
