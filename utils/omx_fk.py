"""
OMX Forward Kinematics (FK) 유틸리티

URDF 파라미터를 기반으로 OMX 관절 위치 → End-Effector 포즈를 계산합니다.
Cosmos Predict2.5 Action-Conditioned 모델의 입력 데이터 변환에 사용됩니다.

Cosmos가 요구하는 EE 포즈 형식:
    state: [x, y, z, roll, pitch, yaw]  (6-dim)
    continuous_gripper_state: [0.0 ~ 1.0]  (1-dim)
    action: 연속 프레임 간 상대 변위 (7-dim)

URDF 소스:
    OMX-F: open_manipulator_description/urdf/omx_f/omx_f.urdf
    OMX-L: open_manipulator_description/urdf/omx_l/omx_l.urdf

Usage:
    from utils.omx_fk import OMXForwardKinematics

    fk = OMXForwardKinematics(robot="omx_f")
    ee_pose = fk.compute(joint_positions=[0.0, -0.5, 0.3, 0.2, 0.0])
    # ee_pose = {"position": [x, y, z], "rpy": [roll, pitch, yaw]}

    # Cosmos 형식으로 변환
    cosmos_state = fk.to_cosmos_state(joint_positions, gripper_value)
    cosmos_action = fk.compute_cosmos_action(states_t, states_t1)
"""

import numpy as np
from typing import Literal

from utils.omx_constants import (
    OMX_F_JOINT_PARAMS,
    OMX_F_EE_OFFSET,
    OMX_L_JOINT_PARAMS,
)


# =============================================================================
# 동차 변환 행렬 (Homogeneous Transformation Matrix)
# =============================================================================


def _rot_x(theta: float) -> np.ndarray:
    """X축 회전 4x4 동차 변환 행렬."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ])


def _rot_y(theta: float) -> np.ndarray:
    """Y축 회전 4x4 동차 변환 행렬."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ])


def _rot_z(theta: float) -> np.ndarray:
    """Z축 회전 4x4 동차 변환 행렬."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def _translation(x: float, y: float, z: float) -> np.ndarray:
    """평행 이동 4x4 동차 변환 행렬."""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


_ROT_FN = {"X": _rot_x, "Y": _rot_y, "Z": _rot_z}


def _rotation_matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """3x3 회전 행렬 → Roll-Pitch-Yaw (XYZ 오일러 각도) 변환.

    Returns:
        (roll, pitch, yaw) in radians
    """
    # pitch (Y축)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return (roll, pitch, yaw)


# =============================================================================
# OMX Forward Kinematics 클래스
# =============================================================================


class OMXForwardKinematics:
    """URDF 기반 OMX Forward Kinematics 계산기.

    URDF에서 추출한 관절 파라미터를 사용하여
    joint positions → EE pose (x, y, z, roll, pitch, yaw) 변환을 수행합니다.

    Args:
        robot: "omx_f" (follower, 매니퓰레이션) 또는 "omx_l" (leader, 텔레오퍼레이션)
    """

    def __init__(self, robot: Literal["omx_f", "omx_l"] = "omx_f"):
        if robot == "omx_f":
            self.joint_params = OMX_F_JOINT_PARAMS
            self.ee_offset = OMX_F_EE_OFFSET
        elif robot == "omx_l":
            self.joint_params = OMX_L_JOINT_PARAMS
            self.ee_offset = None  # OMX-L URDF에 EE 링크 미정의
        else:
            raise ValueError(f"Unknown robot: {robot}. Use 'omx_f' or 'omx_l'.")

        self.robot = robot
        self.n_joints = len(self.joint_params)  # 5

    def compute(self, joint_positions: list[float] | np.ndarray) -> dict:
        """관절 위치 → End-Effector 포즈 계산.

        Args:
            joint_positions: 5개 관절 각도 [joint1, ..., joint5] (radians)

        Returns:
            dict with:
                "position": [x, y, z] (meters)
                "rpy": [roll, pitch, yaw] (radians)
                "transform": 4x4 homogeneous transformation matrix
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.shape != (self.n_joints,):
            raise ValueError(
                f"Expected {self.n_joints} joint positions, got {joint_positions.shape}"
            )

        # 베이스부터 link5까지 변환 체인
        T = np.eye(4)
        for i, (axis, (ox, oy, oz)) in enumerate(self.joint_params):
            # URDF origin offset (부모→자식 평행이동)
            T = T @ _translation(ox, oy, oz)
            # 관절 회전
            T = T @ _ROT_FN[axis](joint_positions[i])

        # End-Effector 오프셋 (fixed joint)
        if self.ee_offset is not None:
            T = T @ _translation(*self.ee_offset)

        position = T[:3, 3].tolist()
        rpy = list(_rotation_matrix_to_rpy(T[:3, :3]))

        return {
            "position": position,
            "rpy": rpy,
            "transform": T,
        }

    def to_cosmos_state(
        self,
        joint_positions: list[float] | np.ndarray,
        gripper: float = 0.0,
    ) -> dict:
        """관절 위치 → Cosmos Predict2.5 Action-Conditioned 입력 state 변환.

        Cosmos가 요구하는 형식:
            state: [x, y, z, roll, pitch, yaw]
            continuous_gripper_state: float

        Args:
            joint_positions: 5개 관절 각도 (radians)
            gripper: 그리퍼 값 [0.0 (닫힘) ~ 1.0 (열림)]

        Returns:
            dict with "state" (6-dim list) and "continuous_gripper_state" (float)
        """
        result = self.compute(joint_positions)
        state = result["position"] + result["rpy"]  # [x, y, z, r, p, y]
        return {
            "state": state,
            "continuous_gripper_state": float(gripper),
        }

    def compute_cosmos_action(
        self,
        joint_positions_t: list[float] | np.ndarray,
        joint_positions_t1: list[float] | np.ndarray,
        gripper_t: float = 0.0,
        gripper_t1: float = 0.0,
    ) -> np.ndarray:
        """연속 두 프레임의 관절 위치 → Cosmos action (상대 변위) 계산.

        Cosmos action = EE_pose(t+1) - EE_pose(t) (7-dim)
            [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, delta_gripper]

        Args:
            joint_positions_t: t 시점 관절 각도 (5-dim)
            joint_positions_t1: t+1 시점 관절 각도 (5-dim)
            gripper_t: t 시점 그리퍼 값
            gripper_t1: t+1 시점 그리퍼 값

        Returns:
            np.ndarray: 7-dim 상대 변위 action
        """
        state_t = self.to_cosmos_state(joint_positions_t, gripper_t)
        state_t1 = self.to_cosmos_state(joint_positions_t1, gripper_t1)

        delta_state = np.array(state_t1["state"]) - np.array(state_t["state"])
        delta_gripper = state_t1["continuous_gripper_state"] - state_t["continuous_gripper_state"]

        return np.append(delta_state, delta_gripper)

    def batch_to_cosmos_states(
        self,
        joint_trajectory: np.ndarray,
        gripper_trajectory: np.ndarray,
    ) -> dict:
        """궤적 전체를 Cosmos state 시퀀스로 변환.

        Args:
            joint_trajectory: (T, 5) 관절 각도 궤적
            gripper_trajectory: (T,) 그리퍼 값 궤적

        Returns:
            dict with:
                "states": (T, 6) EE 포즈 시퀀스
                "continuous_gripper_states": (T,) 그리퍼 시퀀스
                "actions": (T-1, 7) 상대 변위 action 시퀀스
        """
        joint_trajectory = np.asarray(joint_trajectory)
        gripper_trajectory = np.asarray(gripper_trajectory)
        T = joint_trajectory.shape[0]

        states = []
        for t in range(T):
            cosmos = self.to_cosmos_state(joint_trajectory[t], gripper_trajectory[t])
            states.append(cosmos["state"])

        states = np.array(states)
        grippers = gripper_trajectory.copy()

        # action = 연속 프레임 간 상대 변위
        actions = []
        for t in range(T - 1):
            delta = self.compute_cosmos_action(
                joint_trajectory[t], joint_trajectory[t + 1],
                gripper_trajectory[t], gripper_trajectory[t + 1],
            )
            actions.append(delta)

        return {
            "states": states,
            "continuous_gripper_states": grippers,
            "actions": np.array(actions) if actions else np.empty((0, 7)),
        }
