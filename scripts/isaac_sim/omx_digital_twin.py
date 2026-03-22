#!/usr/bin/env python3
"""OMX Digital Twin - Isaac Sim 5.1.0

실물 OMX_F 매니퓰레이터의 관절 상태를 Isaac Sim에서 실시간 미러링한다.
ROS 2 /joint_states 토픽을 구독하여 시뮬레이션 로봇을 구동한다.

사전 조건:
  - open_manipulator Docker 실행 중 (ros2 launch open_manipulator_bringup omx_ai.launch.py)
  - 환경변수: OPEN_MANIPULATOR_PATH=~/workspace/open_manipulator
  - 환경변수: ROS_DOMAIN_ID=30
  - ROS 2 환경 소싱: source /opt/ros/jazzy/setup.bash

실행:
  cd ~/claude/isaacsim && source .venv/bin/activate
  python ~/claude/pseudo-project/scripts/isaac_sim/omx_digital_twin.py
"""

import os
import re
import sys
import tempfile

import yaml

# ── 1. Isaac Sim 초기화 ──────────────────────────────────────────────
# SimulationApp은 반드시 다른 Omniverse import보다 먼저 인스턴스화해야 함
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

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
simulation_app.update()

enable_extension("isaacsim.asset.importer.urdf")
simulation_app.update()


def load_config():
    """설정 파일 로드 + 환경변수 치환."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "config", "omx_digital_twin.yaml")
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: Config file not found: {config_path}")

    if not os.environ.get("OPEN_MANIPULATOR_PATH"):
        sys.exit(
            "ERROR: OPEN_MANIPULATOR_PATH 환경변수가 설정되지 않았습니다.\n"
            "  export OPEN_MANIPULATOR_PATH=~/workspace/open_manipulator"
        )

    with open(config_path) as f:
        raw = f.read()
    raw = os.path.expandvars(raw)
    config = yaml.safe_load(raw)

    # ~ 틸드 경로 확장 (expandvars는 $VAR만 처리, ~는 expanduser 필요)
    urdf_path = config["urdf_import"].get("urdf_path", "")
    if "~" in urdf_path:
        config["urdf_import"]["urdf_path"] = os.path.expanduser(urdf_path)
    for pkg_name in list(config["urdf_import"].get("package_paths", {})):
        pkg_path = config["urdf_import"]["package_paths"][pkg_name]
        if "~" in pkg_path:
            config["urdf_import"]["package_paths"][pkg_name] = os.path.expanduser(pkg_path)

    # 필수 config 키 검증
    required = [
        ("ros2", "domain_id"),
        ("ros2", "joint_state_topic"),
        ("ros2", "isaac_joint_state_topic"),
        ("robot", "name"),
        ("urdf_import", "urdf_path"),
        ("physics", "timestep"),
        ("rendering", "fps"),
    ]
    for section, key in required:
        if section not in config or key not in config[section]:
            sys.exit(f"ERROR: Missing config key: {section}.{key}")

    return config


def resolve_urdf(config):
    """URDF 파일의 package:// 경로를 절대 경로로 치환하여 임시 파일로 저장."""
    urdf_path = config["urdf_import"]["urdf_path"]

    if not os.path.isfile(urdf_path):
        sys.exit(
            f"ERROR: URDF file not found: {urdf_path}\n"
            f"OPEN_MANIPULATOR_PATH 환경변수를 확인하세요.\n"
            f"  export OPEN_MANIPULATOR_PATH=~/workspace/open_manipulator"
        )

    with open(urdf_path, "r") as f:
        urdf_content = f.read()

    for pkg_name, pkg_path in config["urdf_import"].get("package_paths", {}).items():
        if not os.path.isdir(pkg_path):
            print(f"WARNING: Package path not found: {pkg_path} (for {pkg_name})")
        urdf_content = re.sub(
            rf"package://{re.escape(pkg_name)}/",
            pkg_path.rstrip("/") + "/",
            urdf_content,
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(urdf_content)
    tmp.close()
    return tmp.name


def import_urdf(resolved_urdf_path, config):
    """URDF를 Isaac Sim stage에 import (동기식)."""
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not import_config:
        sys.exit("ERROR: Failed to create URDF import config. "
                 "Is 'isaacsim.asset.importer.urdf' extension loaded?")

    ic = config["urdf_import"]
    import_config.merge_fixed_joints = ic["merge_fixed_joints"]
    import_config.fix_base = ic["fix_base"]
    import_config.make_default_prim = ic["make_default_prim"]
    import_config.create_physics_scene = ic["create_physics_scene"]
    import_config.import_inertia_tensor = ic["import_inertia_tensor"]
    import_config.parse_mimic = ic["parse_mimic"]
    import_config.default_drive_type = ic["default_drive_type"]
    import_config.default_drive_strength = ic["default_drive_strength"]
    import_config.default_position_drive_damping = ic["default_position_drive_damping"]

    try:
        status, robot_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=resolved_urdf_path,
            import_config=import_config,
            dest_path="",
            get_articulation_root=True,
        )
    finally:
        os.unlink(resolved_urdf_path)

    if not robot_prim_path:
        sys.exit("ERROR: URDF import failed. Check URDF file and mesh paths.")

    print(f"Robot imported at: {robot_prim_path}")
    return robot_prim_path


def setup_omnigraph(robot_prim_path, config):
    """OmniGraph로 ROS 2 JointState 구독 → ArticulationController 연결."""
    ros2_config = config["ros2"]

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
                # jointNames 연결 필수: ROS JointState 관절 순서와
                # USD articulation 관절 순서가 다를 수 있으므로 이름 기반 매핑
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
                # ROS_DOMAIN_ID 명시 설정 (환경변수 의존 제거)
                ("Context.inputs:domain_id", ros2_config["domain_id"]),
                ("Context.inputs:useDomainIDEnvVar", False),
                # /joint_states 토픽 구독 (기본값 "joint_command" → 변경 필수)
                ("SubscribeJointState.inputs:topicName", ros2_config["joint_state_topic"]),
                # 로봇 prim 경로
                ("ArticulationController.inputs:targetPrim",
                 [usdrt.Sdf.Path(robot_prim_path)]),
                ("PublishJointState.inputs:targetPrim",
                 [usdrt.Sdf.Path(robot_prim_path)]),
                # 시뮬레이션 상태 퍼블리시 토픽 (기본값 "joint_states"와 충돌 방지)
                ("PublishJointState.inputs:topicName",
                 ros2_config["isaac_joint_state_topic"]),
            ],
        },
    )
    print("OmniGraph configured: SubscribeJointState -> ArticulationController")


def main():
    # ── 4. 설정 로드 ────────────────────────────────────────────────
    config = load_config()
    print(f"Config loaded. Robot: {config['robot']['name']}, "
          f"ROS_DOMAIN_ID: {config['ros2']['domain_id']}")

    # ── 5. URDF Import ──────────────────────────────────────────────
    resolved_urdf_path = resolve_urdf(config)
    robot_prim_path = import_urdf(resolved_urdf_path, config)

    # ── 6. World 생성 ───────────────────────────────────────────────
    physics_dt = config["physics"]["timestep"]
    rendering_fps = config["rendering"]["fps"]
    rendering_dt = 1.0 / rendering_fps

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
    )
    world.scene.add_default_ground_plane()
    print(f"World created. physics_dt={physics_dt}s, rendering_dt={rendering_dt:.4f}s")

    # ── 7. OmniGraph 구성 ───────────────────────────────────────────
    setup_omnigraph(robot_prim_path, config)

    # ── 8. 시뮬레이션 루프 ──────────────────────────────────────────
    timeline = omni.timeline.get_timeline_interface()

    # world.reset()이 내부적으로 timeline.play()를 호출
    world.reset()

    print("=" * 60)
    print("OMX Digital Twin running.")
    print(f"  Subscribing to: /{config['ros2']['joint_state_topic']}")
    print(f"  Publishing to:  /{config['ros2']['isaac_joint_state_topic']}")
    print(f"  ROS_DOMAIN_ID:  {config['ros2']['domain_id']}")
    print("  Press Ctrl+C or close the window to stop.")
    print("=" * 60)

    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        timeline.stop()
        simulation_app.close()
        print("OMX Digital Twin stopped.")


if __name__ == "__main__":
    main()
