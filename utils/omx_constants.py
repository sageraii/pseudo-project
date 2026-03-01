"""
OMX 로봇 공통 상수 및 유틸리티

4개 스크립트(week3, week4, week6_deploy, week6_eval)에 산재된 OMX 상수와
더미 관측 함수를 통합합니다. GR00T IDM과의 관절 이름 매핑 및
비디오 dtype 변환 유틸리티도 포함합니다.

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
