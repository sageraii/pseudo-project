#!/usr/bin/env python3
"""
SO-100 → OMX 데이터 변환 스크립트 (URDF FK → IK 방법 B)

두 로봇의 URDF 기반 순기구학(FK)과 역기구학(IK)을 사용하여
SO-100 LeRobot 데이터셋을 OMX LeRobot 데이터셋으로 변환합니다.

Pipeline:
    SO-100 parquet (degrees)
      → deg2rad (calibration offset 적용)
      → SO-100 FK → 4x4 EE Transform
      → OMX IK (scipy numerical optimization, warm-start)
      → OMX joint angles (radians)
      → Write OMX LeRobot v2 format

URDF Sources:
    SO-100:  ~/claude/SO-ARM100/Simulation/SO100/so100.urdf
    OMX-F:   ~/claude/open_manipulator/open_manipulator_description/urdf/omx_f/omx_f.urdf

Prerequisites:
    numpy, scipy  (필수, 기본 설치)
    pandas, pyarrow  (데이터셋 변환 시 필요, --self-test는 불필요)

Usage:
    # FK/IK 자체 테스트 (pandas/pyarrow 불필요)
    python scripts/convert_so100_to_omx.py --self-test

    # 전체 변환 (pandas, pyarrow 필요)
    python scripts/convert_so100_to_omx.py \
        --so100-dataset /path/to/so100_lerobot_dataset \
        --output-dir /path/to/omx_output

    # 워크스페이스 검증만 (변환 없이)
    python scripts/convert_so100_to_omx.py \
        --so100-dataset /path/to/so100_lerobot_dataset \
        --output-dir /tmp/omx_test --validate-only

    # 일부 에피소드만 테스트
    python scripts/convert_so100_to_omx.py \
        --so100-dataset /path/to/so100_lerobot_dataset \
        --output-dir /tmp/omx_test --max-episodes 3
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


# =============================================================================
# URDF FK 유틸리티
# =============================================================================


def _make_transform(xyz, rpy=(0, 0, 0)):
    """(xyz, rpy) → 4x4 동차 변환 행렬. URDF 표준: rpy = (roll, pitch, yaw)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _rot_axis(axis_vec, angle):
    """축 벡터 + 각도 → 4x4 회전 행렬."""
    rotvec = np.array(axis_vec, dtype=float) * angle
    T = np.eye(4)
    if np.linalg.norm(rotvec) > 1e-12:
        T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return T


def forward_kinematics(chain, joint_angles, ee_offset=None):
    """URDF 체인 기반 순기구학.

    Args:
        chain: 관절 정의 리스트 [{origin_xyz, origin_rpy, axis}, ...]
        joint_angles: 관절 각도 (radians), len == len(chain)
        ee_offset: EE 고정 오프셋 {origin_xyz, origin_rpy}

    Returns:
        4x4 동차 변환 행렬 (base → EE)
    """
    T = np.eye(4)
    for i, jdef in enumerate(chain):
        T = T @ _make_transform(jdef["origin_xyz"], jdef["origin_rpy"])
        T = T @ _rot_axis(jdef["axis"], joint_angles[i])

    if ee_offset:
        T = T @ _make_transform(ee_offset["origin_xyz"], ee_offset["origin_rpy"])

    return T


# =============================================================================
# SO-100 URDF 체인 (so100.urdf 에서 추출)
# =============================================================================

SO100_CHAIN = [
    {  # shoulder_pan: base → shoulder
        "name": "shoulder_pan",
        "origin_xyz": (0, -0.0452, 0.0165),
        "origin_rpy": (1.57079, 0, 0),
        "axis": (0, 1, 0),
        "limits_rad": (-2.0, 2.0),
    },
    {  # shoulder_lift: shoulder → upper_arm
        "name": "shoulder_lift",
        "origin_xyz": (0, 0.1025, 0.0306),
        "origin_rpy": (-1.8, 0, 0),
        "axis": (1, 0, 0),
        "limits_rad": (0.0, 3.5),
    },
    {  # elbow_flex: upper_arm → lower_arm
        "name": "elbow_flex",
        "origin_xyz": (0, 0.11257, 0.028),
        "origin_rpy": (1.57079, 0, 0),
        "axis": (1, 0, 0),
        "limits_rad": (-3.14158, 0.0),
    },
    {  # wrist_flex: lower_arm → wrist
        "name": "wrist_flex",
        "origin_xyz": (0, 0.0052, 0.1349),
        "origin_rpy": (-1.0, 0, 0),
        "axis": (1, 0, 0),
        "limits_rad": (-2.5, 1.2),
    },
    {  # wrist_roll: wrist → gripper_base
        "name": "wrist_roll",
        "origin_xyz": (0, -0.0601, 0),
        "origin_rpy": (0, 1.57079, 0),
        "axis": (0, 1, 0),
        "limits_rad": (-3.14158, 3.14158),
    },
]

# gripper → jaw (EE 기준점)
SO100_EE_OFFSET = {
    "origin_xyz": (-0.0202, -0.0244, 0),
    "origin_rpy": (0, 3.14158, 0),
}


# =============================================================================
# OMX-F URDF 체인 (omx_f.urdf 에서 추출)
# =============================================================================

OMX_CHAIN = [
    {  # joint1: link0 → link1
        "name": "joint1",
        "origin_xyz": (-0.01125, 0, 0.034),
        "origin_rpy": (0, 0, 0),
        "axis": (0, 0, 1),
        "limits_rad": (-np.pi, np.pi),
    },
    {  # joint2: link1 → link2
        "name": "joint2",
        "origin_xyz": (0, 0, 0.0635),
        "origin_rpy": (0, 0, 0),
        "axis": (0, 1, 0),
        "limits_rad": (-np.pi, np.pi),
    },
    {  # joint3: link2 → link3
        "name": "joint3",
        "origin_xyz": (0.0415, 0, 0.11315),
        "origin_rpy": (0, 0, 0),
        "axis": (0, 1, 0),
        "limits_rad": (-np.pi, np.pi),
    },
    {  # joint4: link3 → link4
        "name": "joint4",
        "origin_xyz": (0.162, 0, 0),
        "origin_rpy": (0, 0, 0),
        "axis": (0, 1, 0),
        "limits_rad": (-np.pi, np.pi),
    },
    {  # joint5: link4 → link5
        "name": "joint5",
        "origin_xyz": (0.0287, 0, 0),
        "origin_rpy": (0, 0, 0),
        "axis": (1, 0, 0),
        "limits_rad": (-np.pi, np.pi),
    },
]

# link5 → end_effector_link (fixed)
OMX_EE_OFFSET = {
    "origin_xyz": (0.09193, -0.0016, 0),
    "origin_rpy": (0, 0, 0),
}


# =============================================================================
# SO-100 캘리브레이션: servo degrees → URDF radians
# =============================================================================

SO100_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]

# stats.json에서 추출한 데이터 범위의 중심점 (degrees)
SO100_ZERO_OFFSET_DEG = {
    "shoulder_pan": 123.0,
    "shoulder_lift": 116.0,
    "elbow_flex": 98.0,
    "wrist_flex": 54.0,
    "wrist_roll": -52.0,
}

SO100_GRIPPER_RANGE_DEG = (-7.3, 58.7)
OMX_GRIPPER_RANGE_RAD = (-0.5, 1.0)


def so100_deg_to_urdf_rad(joint_values_deg, offsets=SO100_ZERO_OFFSET_DEG):
    """SO-100 서보 degrees → URDF joint radians.

    Args:
        joint_values_deg: [6] 배열 (5 arm joints + 1 gripper, degrees)

    Returns:
        arm_rad: [5] ndarray, gripper_normalized: float [0,1]
    """
    arm_deg = joint_values_deg[:5]
    gripper_deg = joint_values_deg[5]

    arm_rad = np.zeros(5)
    for i, name in enumerate(SO100_JOINT_NAMES):
        arm_rad[i] = np.deg2rad(arm_deg[i] - offsets[name])

    g_min, g_max = SO100_GRIPPER_RANGE_DEG
    gripper_normalized = np.clip((gripper_deg - g_min) / (g_max - g_min), 0, 1)

    return arm_rad, gripper_normalized


# =============================================================================
# OMX Inverse Kinematics (수치 최적화)
# =============================================================================


def omx_ik(target_T, initial_guess=None, pos_weight=100.0, rot_weight=0.1,
           n_restarts=5):
    """수치 IK: 목표 4x4 transform → OMX 5-DOF joint angles.

    두 로봇의 kinematic 구조가 다르므로 (SO-100: rpy 오프셋 다수, OMX: 깔끔한 구조)
    orientation 완전 매칭은 불가능합니다. 위치(position) 우선 매칭 + 다중 시작점으로
    로컬 최소에 빠지는 것을 방지합니다.

    Args:
        target_T: 목표 EE 4x4 변환 행렬
        initial_guess: 초기 추정값 [5], None이면 zeros
        pos_weight: 위치 오차 가중치 (기본 100, 위치 우선)
        rot_weight: 방향 오차 가중치 (기본 0.1, 보조)
        n_restarts: 다중 시작점 수 (initial_guess 외 추가 랜덤 시도)
    """
    target_pos = target_T[:3, 3]
    target_rot = Rotation.from_matrix(target_T[:3, :3])

    bounds = [(jd["limits_rad"][0], jd["limits_rad"][1]) for jd in OMX_CHAIN]

    def cost_fn(q):
        T = forward_kinematics(OMX_CHAIN, q, OMX_EE_OFFSET)
        pos_err = np.sum((T[:3, 3] - target_pos) ** 2)
        cur_rot = Rotation.from_matrix(T[:3, :3])
        rot_err = (target_rot.inv() * cur_rot).magnitude() ** 2
        return pos_weight * pos_err + rot_weight * rot_err

    # 시작점 후보들: 주어진 초기값 + 랜덤 시작점들
    starts = []
    if initial_guess is not None:
        starts.append(np.array(initial_guess))
    else:
        starts.append(np.zeros(5))

    rng = np.random.RandomState(hash(tuple(target_pos.round(4))) % (2**31))
    for _ in range(n_restarts):
        q0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        starts.append(q0)

    best_q, best_cost, best_ok = starts[0], float("inf"), False

    for i, q0 in enumerate(starts):
        result = minimize(
            cost_fn, q0, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 300, "ftol": 1e-12},
        )
        if result.fun < best_cost:
            best_q = result.x
            best_cost = result.fun
            best_ok = result.success

        # Early exit: 초기값(warm-start)이 충분히 좋으면 restart 스킵
        if i == 0:
            T_check = forward_kinematics(OMX_CHAIN, best_q, OMX_EE_OFFSET)
            if np.linalg.norm(T_check[:3, 3] - target_pos) < 0.002:  # 2mm 이내
                break

    return best_q, best_cost, best_ok


def omx_gripper_from_normalized(gripper_01):
    """정규화된 그리퍼 [0,1] → OMX gripper radians."""
    g_min, g_max = OMX_GRIPPER_RANGE_RAD
    return g_min + gripper_01 * (g_max - g_min)


# =============================================================================
# --self-test: FK/IK 자체 검증 (pandas/pyarrow 불필요)
# =============================================================================


def run_self_test():
    """FK, IK, 단위변환 자체 테스트."""
    print("=" * 60)
    print("SO-100 → OMX 변환 자체 테스트 (URDF FK → IK)")
    print("=" * 60)

    # 1) SO-100 FK 홈 포지션
    print("\n[1] SO-100 FK 홈 포지션")
    T = forward_kinematics(SO100_CHAIN, np.zeros(5), SO100_EE_OFFSET)
    print(f"    EE 위치(m): [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]")

    # 2) OMX FK 홈 포지션
    print("\n[2] OMX-F FK 홈 포지션")
    T_omx = forward_kinematics(OMX_CHAIN, np.zeros(5), OMX_EE_OFFSET)
    print(f"    EE 위치(m): [{T_omx[0,3]:.4f}, {T_omx[1,3]:.4f}, {T_omx[2,3]:.4f}]")

    # 3) 단위 변환 테스트
    print("\n[3] 단위 변환 (중심값 → 0 rad)")
    test_deg = np.array([123.0, 116.0, 98.0, 54.0, -52.0, 16.0])
    arm_r, grip = so100_deg_to_urdf_rad(test_deg)
    print(f"    입력(deg): {test_deg}")
    print(f"    출력(rad): {arm_r.round(6)} (모두 ~0)")
    print(f"    그리퍼:    {grip:.3f}")
    assert np.allclose(arm_r, 0, atol=1e-6), "중심값이 0에 매핑되어야 함"
    print("    PASS")

    # 4) FK→IK 라운드트립 테스트
    print("\n[4] FK→IK 라운드트립 테스트")
    test_poses = [
        ("홈",    np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
        ("앞쪽",  np.array([0.3, 0.5, -0.8, 0.2, 0.1])),
        ("옆쪽",  np.array([-0.3, 0.8, -1.0, -0.3, -0.5])),
        ("위쪽",  np.array([1.0, 0.3, -0.5, 0.8, 1.0])),
    ]

    prev_q = np.zeros(5)
    all_pass = True
    for label, q_so100 in test_poses:
        T_target = forward_kinematics(SO100_CHAIN, q_so100, SO100_EE_OFFSET)
        target_pos = T_target[:3, 3]

        omx_q, cost, ok = omx_ik(T_target, initial_guess=prev_q)
        T_result = forward_kinematics(OMX_CHAIN, omx_q, OMX_EE_OFFSET)
        result_pos = T_result[:3, 3]
        err_mm = np.linalg.norm(target_pos - result_pos) * 1000

        status = "PASS" if err_mm < 10.0 else "WARN"
        if err_mm >= 10.0:
            all_pass = False
        print(f"    {label:4s}: 오차 {err_mm:6.2f}mm, "
              f"수렴={ok}, OMX q={omx_q.round(3)}")
        prev_q = omx_q

    # 5) SO-100 실제 데이터 범위 시뮬레이션
    print("\n[5] SO-100 실제 데이터 범위 변환 테스트")
    # stats.json 범위에서 랜덤 샘플
    np.random.seed(42)
    so100_ranges = {
        "shoulder_pan": (38.85, 207.25),
        "shoulder_lift": (37.7, 194.6),
        "elbow_flex": (20.9, 175.0),
        "wrist_flex": (-7.2, 115.9),
        "wrist_roll": (-167.0, 62.75),
        "gripper": (-7.3, 58.7),
    }

    n_samples = 50
    reachable = 0
    errors_mm = []

    prev_q = np.zeros(5)
    for _ in range(n_samples):
        sample_deg = np.array([
            np.random.uniform(*so100_ranges[name])
            for name in list(so100_ranges.keys())
        ])
        arm_rad, grip = so100_deg_to_urdf_rad(sample_deg)
        T_target = forward_kinematics(SO100_CHAIN, arm_rad, SO100_EE_OFFSET)

        omx_q, cost, ok = omx_ik(T_target, initial_guess=prev_q)
        T_result = forward_kinematics(OMX_CHAIN, omx_q, OMX_EE_OFFSET)
        err = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3]) * 1000
        errors_mm.append(err)

        if err < 10.0:
            reachable += 1
        if ok:
            prev_q = omx_q

    errors_mm = np.array(errors_mm)
    print(f"    샘플: {n_samples}")
    print(f"    도달 가능 (<10mm): {reachable}/{n_samples} ({reachable/n_samples:.0%})")
    print(f"    오차 평균: {errors_mm.mean():.2f}mm")
    print(f"    오차 최대: {errors_mm.max():.2f}mm")
    print(f"    오차 P95:  {np.percentile(errors_mm, 95):.2f}mm")

    # 6) warm-start 효과 테스트
    print("\n[6] Warm-start 효과 (궤적 연속성)")
    trajectory_deg = []
    base = np.array([123.0, 116.0, 98.0, 54.0, -52.0, 16.0])
    for t in range(20):
        delta = np.array([
            5.0 * np.sin(t * 0.3),
            3.0 * np.cos(t * 0.2),
            -4.0 * np.sin(t * 0.25),
            2.0 * np.cos(t * 0.35),
            6.0 * np.sin(t * 0.15),
            1.0 * np.sin(t * 0.4),
        ])
        trajectory_deg.append(base + delta * t)

    prev_q = np.zeros(5)
    traj_errors = []
    t_start = time.time()
    for sample in trajectory_deg:
        arm_rad, grip = so100_deg_to_urdf_rad(sample)
        T_target = forward_kinematics(SO100_CHAIN, arm_rad, SO100_EE_OFFSET)
        omx_q, _, ok = omx_ik(T_target, initial_guess=prev_q)
        T_result = forward_kinematics(OMX_CHAIN, omx_q, OMX_EE_OFFSET)
        err = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3]) * 1000
        traj_errors.append(err)
        if ok:
            prev_q = omx_q
    t_elapsed = time.time() - t_start

    traj_errors = np.array(traj_errors)
    fps = len(trajectory_deg) / max(t_elapsed, 0.001)
    print(f"    프레임: {len(trajectory_deg)}, 시간: {t_elapsed:.2f}s ({fps:.0f} fps)")
    print(f"    오차 평균: {traj_errors.mean():.2f}mm, 최대: {traj_errors.max():.2f}mm")

    print("\n" + "=" * 60)
    print("자체 테스트 완료")
    print("=" * 60)


# =============================================================================
# 데이터셋 변환 파이프라인 (pandas/pyarrow 필요)
# =============================================================================

OMX_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def _require_pandas():
    """pandas/pyarrow 임포트. 없으면 안내 메시지 출력."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        print("[ERROR] 데이터셋 변환에는 pandas와 pyarrow가 필요합니다.")
        print("  설치: pip install pandas pyarrow")
        print("  또는 --self-test 로 FK/IK 검증만 실행하세요.")
        sys.exit(1)


def convert_episode(pd, df_episode, prev_omx_q=None):
    """단일 에피소드의 SO-100 관절 데이터를 OMX로 변환."""
    n_frames = len(df_episode)
    omx_states = np.zeros((n_frames, 6))
    omx_actions = np.zeros((n_frames, 6))

    ik_stats = {"total": n_frames, "converged": 0, "pos_errors_mm": []}
    current_guess = prev_omx_q[:5].copy() if prev_omx_q is not None else np.zeros(5)

    for t in range(n_frames):
        row = df_episode.iloc[t]

        # --- state 변환 ---
        state_deg = np.array(row["observation.state"], dtype=float)
        arm_rad, grip_norm = so100_deg_to_urdf_rad(state_deg)
        T_target = forward_kinematics(SO100_CHAIN, arm_rad, SO100_EE_OFFSET)

        omx_q, cost, success = omx_ik(T_target, initial_guess=current_guess)

        T_check = forward_kinematics(OMX_CHAIN, omx_q, OMX_EE_OFFSET)
        pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
        ik_stats["pos_errors_mm"].append(pos_err * 1000)

        if success and pos_err < 0.01:
            ik_stats["converged"] += 1
            current_guess = omx_q.copy()

        omx_states[t, :5] = omx_q
        omx_states[t, 5] = omx_gripper_from_normalized(grip_norm)

        # --- action 변환 ---
        action_deg = np.array(row["action"], dtype=float)
        act_arm_rad, act_grip_norm = so100_deg_to_urdf_rad(action_deg)
        T_act_target = forward_kinematics(SO100_CHAIN, act_arm_rad, SO100_EE_OFFSET)
        omx_act_q, _, _ = omx_ik(T_act_target, initial_guess=omx_q)

        omx_actions[t, :5] = omx_act_q
        omx_actions[t, 5] = omx_gripper_from_normalized(act_grip_norm)

    errors = np.array(ik_stats["pos_errors_mm"])
    ik_stats["converge_ratio"] = ik_stats["converged"] / max(n_frames, 1)
    ik_stats["mean_error_mm"] = float(errors.mean())
    ik_stats["max_error_mm"] = float(errors.max())

    return omx_states, omx_actions, ik_stats


def build_omx_dataframe(pd, df_orig, omx_states, omx_actions):
    """변환된 OMX 데이터로 새 DataFrame 생성."""
    records = []
    for t in range(len(df_orig)):
        row = df_orig.iloc[t]
        record = {}
        for j, name in enumerate(OMX_JOINT_NAMES):
            record[f"state.{name}"] = float(omx_states[t, j])
        for j, name in enumerate(OMX_JOINT_NAMES):
            record[f"action.{name}"] = float(omx_actions[t, j])
        for meta_key in ["episode_index", "frame_index", "timestamp", "index", "task_index"]:
            if meta_key in row.index:
                record[meta_key] = row[meta_key]
        records.append(record)
    return pd.DataFrame(records)


def create_omx_metadata(so100_meta_dir, output_meta_dir, conversion_stats):
    """SO-100 메타데이터를 OMX용으로 변환."""
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    so100_info_path = so100_meta_dir / "info.json"
    if so100_info_path.exists():
        with open(so100_info_path) as f:
            info = json.load(f)
        info["robot_type"] = "omx"
        info["_conversion_source"] = "so100"
        info["_conversion_method"] = "URDF FK->IK (method B)"
        with open(output_meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    for fname in ["episodes.jsonl", "tasks.jsonl"]:
        src = so100_meta_dir / fname
        if src.exists():
            shutil.copy2(src, output_meta_dir / fname)

    with open(output_meta_dir / "conversion_stats.json", "w") as f:
        json.dump(conversion_stats, f, indent=2, default=str)


def create_omx_modality_json(output_dir):
    """OMX modality.json 생성 (IDM 학습용)."""
    modality = {
        "state": {},
        "action": {},
        "video": {"webcam": {"original_key": "observation.images.webcam"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    for i, name in enumerate(OMX_JOINT_NAMES):
        modality["state"][name] = {"start": i, "end": i + 1}
        modality["action"][name] = {
            "start": i, "end": i + 1,
            "absolute": name == "gripper",
        }
    out_path = output_dir / "modality.json"
    with open(out_path, "w") as f:
        json.dump(modality, f, indent=2)
    return out_path


def compute_omx_stats(all_states, all_actions):
    """변환된 OMX 데이터의 정규화 통계 계산."""
    states = np.array(all_states)
    actions = np.array(all_actions)

    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    return {"observation.state": _stats(states), "action": _stats(actions)}


def convert_dataset(args):
    """SO-100 데이터셋 전체를 OMX로 변환."""
    pd = _require_pandas()

    so100_dir = Path(args.so100_dataset)
    output_dir = Path(args.output_dir)
    data_dir = so100_dir / "data"
    meta_dir = so100_dir / "meta"
    video_dir = so100_dir / "videos"

    if not data_dir.exists():
        print(f"[ERROR] 데이터 디렉토리 없음: {data_dir}")
        sys.exit(1)

    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        print(f"[ERROR] parquet 파일 없음: {data_dir}")
        sys.exit(1)

    print(f"입력: {so100_dir}")
    print(f"출력: {output_dir}")
    print(f"parquet 파일: {len(parquet_files)}개")

    all_dfs = [pd.read_parquet(pf) for pf in parquet_files]
    df_all = pd.concat(all_dfs, ignore_index=True)

    episodes = sorted(df_all["episode_index"].unique())
    print(f"총 에피소드: {len(episodes)}, 총 프레임: {len(df_all)}")

    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
        print(f"변환 대상: {len(episodes)} 에피소드")

    # 워크스페이스 검증 (샘플링)
    print("\n--- 워크스페이스 검증 ---")
    sample_states = np.stack(df_all["observation.state"].values)
    n_val = min(200, len(sample_states))
    indices = np.random.choice(len(sample_states), n_val, replace=False)
    reachable, errors = 0, []
    prev_q = np.zeros(5)
    for idx in indices:
        arm_rad, _ = so100_deg_to_urdf_rad(sample_states[idx])
        T_t = forward_kinematics(SO100_CHAIN, arm_rad, SO100_EE_OFFSET)
        omx_q, _, ok = omx_ik(T_t, initial_guess=prev_q)
        T_r = forward_kinematics(OMX_CHAIN, omx_q, OMX_EE_OFFSET)
        e = np.linalg.norm(T_t[:3, 3] - T_r[:3, 3]) * 1000
        errors.append(e)
        if e < 10:
            reachable += 1
        if ok:
            prev_q = omx_q

    errors = np.array(errors)
    ratio = reachable / n_val
    print(f"  도달 가능 (<10mm): {reachable}/{n_val} ({ratio:.0%})")
    print(f"  위치 오차: 평균 {errors.mean():.2f}mm, 최대 {errors.max():.2f}mm")

    if ratio < 0.5:
        print("[WARNING] 도달 가능 비율이 50% 미만. 캘리브레이션 오프셋을 확인하세요.")

    if args.validate_only:
        print("\n검증 완료 (--validate-only).")
        return

    # 에피소드별 변환
    print(f"\n--- 변환 시작 ({len(episodes)} 에피소드) ---")
    output_data_dir = output_dir / "data" / "chunk-000"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    all_omx_states, all_omx_actions, all_converted_dfs = [], [], []
    conversion_stats = {"episodes": {}}
    prev_omx_q = None
    t_start = time.time()

    for ep_idx, ep in enumerate(episodes):
        df_ep = df_all[df_all["episode_index"] == ep].reset_index(drop=True)
        ep_start = time.time()
        omx_states, omx_actions, ik_stats = convert_episode(pd, df_ep, prev_omx_q)
        ep_elapsed = time.time() - ep_start

        prev_omx_q = omx_states[-1]
        all_omx_states.append(omx_states)
        all_omx_actions.append(omx_actions)

        df_omx = build_omx_dataframe(pd, df_ep, omx_states, omx_actions)
        all_converted_dfs.append(df_omx)

        conversion_stats["episodes"][int(ep)] = {
            "frames": len(df_ep),
            "converge_ratio": ik_stats["converge_ratio"],
            "mean_error_mm": ik_stats["mean_error_mm"],
            "max_error_mm": ik_stats["max_error_mm"],
            "time_sec": round(ep_elapsed, 1),
        }

        fps = len(df_ep) / max(ep_elapsed, 0.01)
        print(f"  에피소드 {ep:3d}: {len(df_ep):5d} frames, "
              f"수렴 {ik_stats['converge_ratio']:.0%}, "
              f"오차 {ik_stats['mean_error_mm']:.2f}mm, "
              f"{fps:.0f} fps ({ep_elapsed:.1f}s)")

    total_elapsed = time.time() - t_start
    print(f"\n변환 완료: {total_elapsed:.1f}초")

    # 저장
    print("\n--- 저장 ---")
    df_result = pd.concat(all_converted_dfs, ignore_index=True)
    out_parquet = output_data_dir / "episode_000000.parquet"
    df_result.to_parquet(out_parquet, index=False)
    print(f"  데이터: {out_parquet} ({len(df_result)} rows)")

    if video_dir.exists():
        output_video_dir = output_dir / "videos"
        if output_video_dir.exists():
            shutil.rmtree(output_video_dir)
        output_video_dir.symlink_to(video_dir.resolve())
        print(f"  비디오: symlink → {video_dir.resolve()}")

    output_meta_dir = output_dir / "meta"
    create_omx_metadata(meta_dir, output_meta_dir, conversion_stats)
    mod_path = create_omx_modality_json(output_dir)

    all_s = np.concatenate(all_omx_states, axis=0)
    all_a = np.concatenate(all_omx_actions, axis=0)
    stats = compute_omx_stats(all_s, all_a)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    mean_converge = np.mean([v["converge_ratio"] for v in conversion_stats["episodes"].values()])
    mean_error = np.mean([v["mean_error_mm"] for v in conversion_stats["episodes"].values()])
    print(f"\n=== 변환 요약 ===")
    print(f"  에피소드: {len(episodes)}, 프레임: {len(df_result)}")
    print(f"  평균 수렴률: {mean_converge:.0%}, 평균 오차: {mean_error:.2f}mm")
    print(f"  출력: {output_dir}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SO-100 → OMX LeRobot 데이터 변환 (URDF FK→IK)",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="FK/IK 자체 검증 (pandas/pyarrow 불필요)")
    parser.add_argument("--so100-dataset",
                        help="SO-100 LeRobot 데이터셋 경로")
    parser.add_argument("--output-dir",
                        help="OMX 데이터셋 출력 경로")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="변환할 최대 에피소드 수 (테스트용)")
    parser.add_argument("--validate-only", action="store_true",
                        help="워크스페이스 검증만 수행")

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    elif args.so100_dataset and args.output_dir:
        convert_dataset(args)
    else:
        parser.print_help()
        print("\n사용 예:")
        print("  python scripts/convert_so100_to_omx.py --self-test")
        print("  python scripts/convert_so100_to_omx.py --so100-dataset /path/to/data --output-dir /tmp/out")


if __name__ == "__main__":
    main()
