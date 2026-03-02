"""
OMX 로봇 공통 상수 및 유틸리티

4개 스크립트(week3, week4, week6_deploy, week6_eval)에 산재된 OMX 상수와
더미 관측 함수를 통합합니다. GR00T IDM과의 관절 이름 매핑,
비디오 dtype 변환 유틸리티, URDF 기반 kinematic 파라미터도 포함합니다.

URDF 소스:
    OMX-F: open_manipulator_description/urdf/omx_f/omx_f.urdf
    OMX-L: open_manipulator_description/urdf/omx_l/omx_l.urdf

Usage:
    from utils.omx_constants import OMX_DOF, OMX_IMG_SIZE, create_omx_observation
"""

import numpy as np

# =============================================================================
# OMX 로봇 기본 상수
# =============================================================================

OMX_DOF = 6  # 5 joints + 1 gripper
OMX_IMG_SIZE = 224
OMX_CONTROL_HZ = 100  # ROS 2 ros2_control 주기

# VLA 관절 이름 (GR00T N1.6 VLA에서 사용하는 기능 무관 명칭)
OMX_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]

# =============================================================================
# URDF 기반 Kinematic 파라미터
# =============================================================================
# 각 관절의 회전축과 부모→자식 링크 간 오프셋 (단위: meter)
# 형식: (axis, (offset_x, offset_y, offset_z))
#   axis: "Z" | "Y" | "X" - 회전축
#   offset: 부모 링크 원점에서 자식 링크 원점까지의 변위
#
# 관절 구조 (OMX-F, OMX-L 공통):
#   joint1: 베이스 수평 회전 (Z축)
#   joint2: 어깨 수직 회전 (Y축)
#   joint3: 팔꿈치 굴곡 (Y축)
#   joint4: 손목 굴곡 (Y축)
#   joint5: 손목 회전 (X축)

# OMX-F (Follower) - 실제 매니퓰레이션용 (작업 반경 400mm)
OMX_F_JOINT_PARAMS = [
    ("Z", (-0.01125, 0.0, 0.034)),    # joint1: link0 → link1
    ("Y", (0.0, 0.0, 0.0635)),        # joint2: link1 → link2
    ("Y", (0.0415, 0.0, 0.11315)),    # joint3: link2 → link3
    ("Y", (0.162, 0.0, 0.0)),         # joint4: link3 → link4
    ("X", (0.0287, 0.0, 0.0)),        # joint5: link4 → link5
]

# OMX-F End-Effector 오프셋 (link5 → end_effector_link, fixed joint)
OMX_F_EE_OFFSET = (0.09193, -0.0016, 0.0)

# OMX-F 그리퍼 (link5 기준, 2핑거 mimic)
OMX_F_GRIPPER_PARAMS = {
    "gripper_joint_1": {"axis": "Z", "offset": (0.0295, 0.0075, 0.0)},
    "gripper_joint_2": {"axis": "Z", "offset": (0.0295, -0.0108, 0.0), "mimic": ("gripper_joint_1", -1)},
}

# OMX-L (Leader) - 텔레오퍼레이션용 (작업 반경 335mm)
OMX_L_JOINT_PARAMS = [
    ("Z", (-0.0095, 0.0, 0.0545)),    # joint1: link0 → link1
    ("Y", (0.0, 0.0, 0.042)),         # joint2: link1 → link2
    ("Y", (0.0375, 0.0, 0.09)),       # joint3: link2 → link3
    ("Y", (0.1275, 0.0, 0.0)),        # joint4: link3 → link4
    ("X", (0.0287, 0.0, 0.0)),        # joint5: link4 → link5
]

# OMX-L은 URDF에 end_effector_link가 정의되지 않음
# OMX-L 그리퍼 (link5 기준, 1핑거, Y축)
OMX_L_GRIPPER_PARAMS = {
    "gripper_joint_1": {"axis": "Y", "offset": (0.0298, 0.0, 0.0175)},
}

# 관절 한계 (OMX-F, OMX-L 공통: 모든 관절 ±2π rad)
OMX_JOINT_LIMITS = {
    "lower": -2 * np.pi,  # -6.283185307179586
    "upper": 2 * np.pi,   # 6.283185307179586
    "velocity": 4.8,       # rad/s
    "effort": 1000,        # N (URDF 값, 실제와 다를 수 있음)
}

# 관절 축 요약 (Z, Y, Y, Y, X) - OMX-F, OMX-L 공통
OMX_JOINT_AXES = ["Z", "Y", "Y", "Y", "X"]

# =============================================================================
# IDM ↔ VLA 관절 이름 매핑
# =============================================================================
# GR00T IDM은 해부학적(anatomical) 관절 이름을 사용하고,
# GR00T N1.6 VLA는 기능 무관(generic) 이름을 사용합니다.
# 두 모델 간 데이터를 교환할 때 이 매핑이 필요합니다.

OMX_IDM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# VLA → IDM 매핑 (예: joint1 → shoulder_pan)
OMX_JOINT_MAPPING = dict(zip(OMX_JOINT_NAMES, OMX_IDM_JOINT_NAMES))

# IDM → VLA 매핑 (예: shoulder_pan → joint1)
OMX_JOINT_MAPPING_INV = dict(zip(OMX_IDM_JOINT_NAMES, OMX_JOINT_NAMES))

# =============================================================================
# 비디오 dtype 상수
# =============================================================================
# GR00T N1.6 VLA와 GR00T IDM은 비디오 입력의 dtype이 다릅니다.
#   - VLA: float32 [0.0, 1.0] (정규화된 부동소수점)
#   - IDM: uint8 [0, 255] (원본 픽셀값)
# 파이프라인 간 데이터 전달 시 반드시 변환이 필요합니다.

VIDEO_DTYPE_VLA = np.float32  # GR00T N1.6 VLA: [0.0, 1.0]
VIDEO_DTYPE_IDM = np.uint8  # GR00T IDM: [0, 255]


# =============================================================================
# 더미 관측 생성 함수
# =============================================================================


def create_omx_observation(task: str = "pick up the red cube") -> dict:
    """OMX 로봇 더미 관측 생성 (GR00T N1.6 VLA 포맷).

    GR00T VLA의 get_action()에 전달할 수 있는 관측 딕셔너리를 생성합니다.
    비디오는 float32 [0,1] 범위입니다.

    Args:
        task: 자연어 작업 지시 문자열

    Returns:
        dict: video, state, annotation 키를 포함하는 관측 딕셔너리
            - video.cam1: (1, 1, 224, 224, 3) float32
            - state.joint1~gripper: 각각 (1, 1, 1) float32
            - annotation.task: [[task_string]]
    """
    return {
        "video": {
            "cam1": np.random.rand(1, 1, OMX_IMG_SIZE, OMX_IMG_SIZE, 3).astype(
                np.float32
            )
        },
        "state": {
            "joint1": np.random.rand(1, 1, 1).astype(np.float32),
            "joint2": np.random.rand(1, 1, 1).astype(np.float32),
            "joint3": np.random.rand(1, 1, 1).astype(np.float32),
            "joint4": np.random.rand(1, 1, 1).astype(np.float32),
            "joint5": np.random.rand(1, 1, 1).astype(np.float32),
            "gripper": np.random.rand(1, 1, 1).astype(np.float32),
        },
        "annotation": {"task": [[task]]},
    }


def create_omx_observation_idm() -> dict:
    """OMX 로봇 더미 관측 생성 (GR00T IDM 포맷).

    GR00T IDM의 get_action()에 전달할 수 있는 관측 딕셔너리를 생성합니다.
    비디오는 uint8 [0,255] 범위이며, 관절 이름은 해부학적 이름을 사용합니다.

    Returns:
        dict: video, state 키를 포함하는 관측 딕셔너리
            - video: (1, 2, 1, 256, 256, 3) uint8 (IDM은 2 프레임 필요)
            - state: IDM 관절 이름으로 구성
    """
    return {
        "video": np.random.randint(
            0, 255, (1, 2, 1, 256, 256, 3), dtype=np.uint8
        ),
        "state": {
            name: np.random.rand(1, 1, 1).astype(np.float32)
            for name in OMX_IDM_JOINT_NAMES
        },
    }


# =============================================================================
# 비디오 dtype 변환 함수
# =============================================================================


def convert_video_vla_to_idm(video_float32: np.ndarray) -> np.ndarray:
    """VLA float32 [0,1] → IDM uint8 [0,255] 변환.

    Args:
        video_float32: float32 배열, 값 범위 [0.0, 1.0]

    Returns:
        uint8 배열, 값 범위 [0, 255]
    """
    return np.clip(video_float32 * 255.0, 0, 255).astype(np.uint8)


def convert_video_idm_to_vla(video_uint8: np.ndarray) -> np.ndarray:
    """IDM uint8 [0,255] → VLA float32 [0,1] 변환.

    Args:
        video_uint8: uint8 배열, 값 범위 [0, 255]

    Returns:
        float32 배열, 값 범위 [0.0, 1.0]
    """
    return video_uint8.astype(np.float32) / 255.0
