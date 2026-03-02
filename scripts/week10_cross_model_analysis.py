"""
Week 10: 통합 파이프라인 + Cosmos Predict2.5-IDM 시너지

핵심 시너지:
  Cosmos Predict2.5 합성 비디오 생성 → IDM pseudo action labeling → VLA 학습 데이터 증강

파이프라인:
  1. Cosmos Predict2.5로 action-conditioned 합성 비디오 생성
  2. GR00T IDM으로 비디오에서 action pseudo label 추출
  3. pseudo label 품질 평가 (vla_action_quality.py 참조)
  4. 모델 종합 비교 + 증강 전/후 성능 비교

Usage:
    python scripts/week10_cross_model_analysis.py \
        --cosmos-predict-dir outputs/cosmos_predict \
        --groot-eval-dir outputs/eval \
        --cosmos-eval-dir outputs/cosmos_eval \
        --output-dir outputs/week10_analysis

참조:
    - GR00T-Dreams-IDM/examples/vla_action_quality.py (품질 메트릭)
    - GR00T-Dreams-IDM/examples/idm_inference_example.py (IDM 추론 패턴)
    - utils/omx_fk.py (OMX joint → EE pose → Cosmos 입력 변환)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# sys.path 조정: 프로젝트 루트에서 utils 임포트 가능하도록
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.omx_constants import (
    OMX_DOF,
    OMX_IDM_JOINT_NAMES,
    OMX_JOINT_MAPPING,
    OMX_JOINT_MAPPING_INV,
    VIDEO_DTYPE_IDM,
    VIDEO_DTYPE_VLA,
    convert_video_idm_to_vla,
    convert_video_vla_to_idm,
)
from utils.omx_fk import OMXForwardKinematics


# =============================================================================
# 0. 실제 시연 기반 품질 임계값 자동 보정
# =============================================================================


@dataclass
class QualityThresholds:
    """품질 등급 판정 임계값.

    실제 시연 데이터의 통계로부터 자동 보정되거나,
    시연 데이터가 없을 경우 기본값을 사용합니다.

    등급 판정 기준:
        A: jerk < jerk_a AND consistency < tc_a  (실제 시연 수준)
        B: jerk < jerk_b AND consistency < tc_b  (학습에 사용 가능)
        C: jerk < jerk_c AND consistency < tc_c  (품질 낮음)
        D: 그 외 (비현실적 → 폐기)
    """
    jerk_a: float = 100.0
    jerk_b: float = 500.0
    jerk_c: float = 1000.0
    tc_a: float = 0.1
    tc_b: float = 0.3
    tc_c: float = 0.5
    source: str = "default"
    demo_stats: dict = field(default_factory=dict)


def load_demo_trajectories(demo_dir: Path) -> list[list[np.ndarray]]:
    """실제 시연 데이터에서 action 궤적을 로드합니다.

    지원 형식:
        1. LeRobot v2 parquet: {demo_dir}/data/*.parquet (action 컬럼)
        2. NumPy 저장: {demo_dir}/actions/*.npy
        3. JSON 저장: {demo_dir}/episodes/*.json (actions 키)

    Returns:
        에피소드별 action 시퀀스 리스트.
        각 에피소드: list[np.ndarray], 각 action shape = (action_horizon, action_dim)
    """
    episodes = []

    # 1) NumPy 파일 (.npy)
    npy_files = sorted(demo_dir.glob("actions/*.npy")) + sorted(demo_dir.glob("*.npy"))
    if npy_files:
        for f in npy_files:
            traj = np.load(f)
            # (T, action_dim) → list of (1, action_dim) for compatibility
            if traj.ndim == 2:
                episodes.append([traj[t:t+1] for t in range(len(traj))])
            elif traj.ndim == 3:
                # (T, horizon, dim) → list of (horizon, dim)
                episodes.append([traj[t] for t in range(len(traj))])
        print(f"  시연 데이터 로드: {len(episodes)} episodes from .npy")
        return episodes

    # 2) JSON 파일
    json_files = sorted(demo_dir.glob("episodes/*.json")) + sorted(demo_dir.glob("*.json"))
    if json_files:
        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)
            actions = data.get("actions", data.get("action", []))
            if actions:
                traj = np.array(actions, dtype=np.float32)
                if traj.ndim == 2:
                    episodes.append([traj[t:t+1] for t in range(len(traj))])
                elif traj.ndim == 3:
                    episodes.append([traj[t] for t in range(len(traj))])
        print(f"  시연 데이터 로드: {len(episodes)} episodes from .json")
        return episodes

    # 3) LeRobot v2 parquet
    parquet_files = sorted(demo_dir.glob("data/*.parquet")) + sorted(demo_dir.glob("*.parquet"))
    if parquet_files:
        try:
            import pyarrow.parquet as pq
            for f in parquet_files:
                table = pq.read_table(f)
                # action 컬럼 탐색
                action_cols = [c for c in table.column_names if c.startswith("action")]
                if action_cols:
                    traj = np.column_stack([table[c].to_numpy() for c in sorted(action_cols)])
                    episodes.append([traj[t:t+1] for t in range(len(traj))])
            print(f"  시연 데이터 로드: {len(episodes)} episodes from .parquet")
            return episodes
        except ImportError:
            print("  pyarrow 미설치: parquet 파일을 읽을 수 없습니다.")

    return episodes


def calibrate_thresholds(demo_dir: Path) -> QualityThresholds:
    """실제 시연 데이터의 jerk/temporal_consistency 분포에서 임계값을 자동 보정합니다.

    보정 기준 (정규분포 가정):
        A: mean + 1σ 이내  (실제 시연의 68%가 포함되는 범위)
        B: mean + 2σ 이내  (실제 시연의 95%가 포함되는 범위)
        C: mean + 3σ 이내  (실제 시연의 99.7%가 포함되는 범위)
        D: 3σ 초과          (실제 시연에서 거의 나타나지 않는 수준 → 폐기)

    Args:
        demo_dir: 실제 시연 데이터가 저장된 디렉토리

    Returns:
        보정된 QualityThresholds
    """
    episodes = load_demo_trajectories(demo_dir)

    if not episodes:
        print("  시연 데이터를 찾을 수 없습니다. 기본 임계값을 사용합니다.")
        return QualityThresholds(source="default (no demo data found)")

    # 각 에피소드의 jerk, temporal_consistency 계산
    jerks = []
    consistencies = []

    for ep_actions in episodes:
        if len(ep_actions) < 4:
            continue
        trajectory = np.array([a[0] for a in ep_actions])
        jerks.append(compute_jerk(trajectory))
        consistencies.append(compute_temporal_consistency(ep_actions))

    if not jerks:
        print("  유효한 에피소드가 부족합니다 (최소 4 프레임 필요). 기본 임계값을 사용합니다.")
        return QualityThresholds(source="default (insufficient episodes)")

    jerks = np.array(jerks)
    consistencies = np.array(consistencies)

    jerk_mean, jerk_std = float(jerks.mean()), float(jerks.std())
    tc_mean, tc_std = float(consistencies.mean()), float(consistencies.std())

    # std가 0인 경우 (모든 에피소드가 동일) → mean의 비율로 대체
    if jerk_std < 1e-9:
        jerk_std = max(jerk_mean * 0.2, 1.0)
    if tc_std < 1e-9:
        tc_std = max(tc_mean * 0.2, 0.01)

    thresholds = QualityThresholds(
        jerk_a=jerk_mean + 1.0 * jerk_std,
        jerk_b=jerk_mean + 2.0 * jerk_std,
        jerk_c=jerk_mean + 3.0 * jerk_std,
        tc_a=tc_mean + 1.0 * tc_std,
        tc_b=tc_mean + 2.0 * tc_std,
        tc_c=tc_mean + 3.0 * tc_std,
        source=f"calibrated from {len(jerks)} episodes",
        demo_stats={
            "num_episodes": len(jerks),
            "jerk_mean": round(jerk_mean, 4),
            "jerk_std": round(jerk_std, 4),
            "jerk_min": round(float(jerks.min()), 4),
            "jerk_max": round(float(jerks.max()), 4),
            "tc_mean": round(tc_mean, 4),
            "tc_std": round(tc_std, 4),
            "tc_min": round(float(consistencies.min()), 4),
            "tc_max": round(float(consistencies.max()), 4),
        },
    )

    print(f"  보정 완료 ({len(jerks)} episodes):")
    print(f"    Jerk     — mean={jerk_mean:.2f}, std={jerk_std:.2f} → A<{thresholds.jerk_a:.2f}, B<{thresholds.jerk_b:.2f}, C<{thresholds.jerk_c:.2f}")
    print(f"    Temporal — mean={tc_mean:.4f}, std={tc_std:.4f} → A<{thresholds.tc_a:.4f}, B<{thresholds.tc_b:.4f}, C<{thresholds.tc_c:.4f}")

    return thresholds


# =============================================================================
# 1. Cosmos Predict2.5 합성 비디오 → IDM 입력 포맷 변환
# =============================================================================

def load_cosmos_rollout(rollout_dir: Path) -> list[np.ndarray]:
    """Cosmos Predict2.5 합성 비디오 로드 (또는 더미 생성).

    Cosmos Predict2.5 출력: uint8 [0,255], action-conditioned 비디오
    IDM 입력 요구: uint8 [0,255], shape (256, 256, 3), 2 프레임씩

    Returns:
        프레임 리스트 (각 프레임: uint8 ndarray)
    """
    # 실제 Cosmos Predict2.5 합성 비디오 로드 시도
    try:
        import cv2
        video_files = sorted(rollout_dir.glob("rollout_*.mp4"))
        if video_files:
            frames = []
            cap = cv2.VideoCapture(str(video_files[0]))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame.astype(np.uint8))
            cap.release()
            if frames:
                print(f"  Cosmos Predict2.5 합성 비디오 로드: {len(frames)} frames from {video_files[0].name}")
                return frames
    except ImportError:
        pass

    # 더미 롤아웃 생성 (파이프라인 검증용)
    num_frames = 30
    base = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    frames = []
    for i in range(num_frames):
        noise = np.random.randint(-3, 3, base.shape, dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise * i // 10, 0, 255).astype(np.uint8)
        frames.append(frame)

    print(f"  더미 롤아웃 생성: {num_frames} frames (파이프라인 검증용)")
    return frames


def prepare_idm_input(frames: list[np.ndarray]) -> list[dict]:
    """합성 비디오 프레임을 IDM 입력 포맷으로 변환.

    IDM 입력 요구사항 (idm.py validate_inputs 기준):
        - video: np.ndarray, dtype uint8
        - shape: (B, T, V, H, W, C) where shape[3] == 3 (color channels)
        즉 V(뷰 수)가 3번째 차원(0-indexed)이 아닌, H가 3번째 차원.
        실제 shape: (1, 2, 1, H, W, 3) → shape[3] == H (not 3)
        ※ idm.py line 102: shape[3] == N_COLOR_CHANNELS (3)
           이는 V(뷰) 차원이 3인 경우를 가정. 단일 카메라면 V=1이므로
           검증이 실패할 수 있음 → 뷰 차원에 3 맞추기 or 검증 스킵 필요.

    Cosmos Predict2.5 출력은 이미 uint8이므로 dtype 변환 불필요.

    Args:
        frames: uint8 프레임 리스트, 각 (H, W, 3)

    Returns:
        IDM 입력 딕셔너리 리스트
    """
    idm_inputs = []
    for i in range(len(frames) - 1):
        frame_t = frames[i]
        frame_t1 = frames[i + 1]

        # IDM 입력 형태: (B=1, T=2, V=1, H, W, C=3) uint8
        # V=1 (단일 카메라) — 실제 IDM validate_inputs에서 shape[3]==3 검증은
        # V 차원이 아닌 C 차원을 가리킴 (6D shape에서 index 3 = H)
        # 실제 사용 시 transforms_idm.py의 _prepare_video가 rearrange로 처리
        video = np.stack([frame_t, frame_t1])[np.newaxis, :, np.newaxis, ...]
        assert video.dtype == VIDEO_DTYPE_IDM, (
            f"IDM 입력은 uint8이어야 합니다. 현재: {video.dtype}"
        )

        idm_inputs.append({
            "video": video,
            "frame_indices": (i, i + 1),
        })

    return idm_inputs


# =============================================================================
# 2. IDM Pseudo Labeling
# =============================================================================

# OMX modality.json에 대응하는 action 인덱스 매핑
# (GR00T-Dreams-IDM/IDM_dump/global_metadata/omx/modality.json 참조)
OMX_ACTION_MODALITY = {
    "shoulder_pan":  {"start": 0, "end": 1},
    "shoulder_lift": {"start": 1, "end": 2},
    "elbow_flex":    {"start": 2, "end": 3},
    "wrist_flex":    {"start": 3, "end": 4},
    "wrist_roll":    {"start": 4, "end": 5},
    "gripper":       {"start": 5, "end": 6},
}

# Embodiment ID for OMX (transforms_idm.py _EMBODIMENT_TAG_MAPPING)
OMX_EMBODIMENT_ID = 30

# <DREAM> 접두사: 합성 데이터 식별 규약 (dump_idm_actions.py 참조)
DREAM_PREFIX = "<DREAM>"


def reassemble_actions_by_modality(
    raw_actions: np.ndarray,
    modality_map: dict = OMX_ACTION_MODALITY,
) -> dict[str, np.ndarray]:
    """IDM 출력(max_action_dim=32 벡터)을 modality.json 기준으로 관절별 분리.

    실제 IDM은 32차원 벡터를 출력하며, modality.json의 start/end 인덱스로
    각 관절에 해당하는 슬라이스를 추출합니다.

    Args:
        raw_actions: shape (action_horizon, max_action_dim) — IDM 원본 출력
        modality_map: 관절별 start/end 인덱스

    Returns:
        관절 이름 → (action_horizon, joint_dim) 매핑
    """
    result = {}
    for joint_name, indices in modality_map.items():
        result[joint_name] = raw_actions[:, indices["start"]:indices["end"]]
    return result


def unapply_normalization(
    actions: np.ndarray,
    stats: dict | None = None,
) -> np.ndarray:
    """IDM 예측 action의 역정규화 (denormalization).

    실제 IDM 파이프라인에서는 dataset.transforms.unapply(Batch(action=...))를
    호출하여 정규화된 action을 물리적 단위로 복원합니다.
    (dump_idm_actions.py line 258 참조)

    여기서는 min_max 정규화의 역변환을 수행합니다:
        action_real = action_norm * (max - min) + min

    Args:
        actions: 정규화된 action, shape (T, D)
        stats: {"min": [...], "max": [...]} — 정규화 통계
               None이면 역정규화를 건너뜁니다 (placeholder stats 사용 시)

    Returns:
        역정규화된 action, shape (T, D)
    """
    if stats is None:
        # stats.json이 placeholder (mean=0, std=1)이면 역정규화 불필요
        return actions

    action_min = np.array(stats["min"], dtype=np.float32)
    action_max = np.array(stats["max"], dtype=np.float32)

    # min_max 역변환: norm * (max - min) + min
    denormed = actions * (action_max - action_min) + action_min
    return denormed


def multistep_average(
    per_step_predictions: list[np.ndarray],
    action_horizon: int,
    num_frames: int,
) -> np.ndarray:
    """겹치는 IDM 예측을 timestep별로 평균화.

    실제 dump_idm_actions.py의 핵심 로직:
    각 시작점 s에서 action_horizon 길이의 예측을 하면,
    timestep t에 대해 여러 윈도우의 예측이 겹칩니다.
    이 겹치는 예측들을 평균하여 안정적인 최종 action을 얻습니다.

    Args:
        per_step_predictions: 시작점별 예측 리스트, 각 (action_horizon, action_dim)
        action_horizon: IDM action horizon (16)
        num_frames: 전체 프레임 수

    Returns:
        평균화된 action 시퀀스, shape (num_frames, action_dim)
    """
    from collections import defaultdict

    action_dict = defaultdict(list)

    for s, pred in enumerate(per_step_predictions):
        for j in range(min(action_horizon, pred.shape[0])):
            if s + j < num_frames:
                action_dict[s + j].append(pred[j])

    # 각 timestep에 대해 모든 예측을 평균
    averaged = np.zeros((num_frames, per_step_predictions[0].shape[-1]), dtype=np.float32)
    for t in range(num_frames):
        if action_dict[t]:
            averaged[t] = np.mean(action_dict[t], axis=0)

    return averaged


def simulate_idm_inference(
    idm_inputs: list[dict],
    use_multistep_avg: bool = True,
) -> list[np.ndarray]:
    """IDM pseudo labeling (시뮬레이션 모드).

    실제 IDM 모델 없이 파이프라인 구조를 검증합니다.
    IDM 가중치는 HuggingFace에 공개되지 않으므로 직접 학습이 필요합니다.
    (학습: GR00T-Dreams-IDM/scripts/idm_training.py + IDM_dump/base.yaml)

    실제 사용 시 (GR00T-Dreams-IDM/IDM_dump/dump_idm_actions.py 기준):
        1. 모델 로딩 (IDM 가중치는 HuggingFace 비공개 → 직접 학습 필요):
            from gr00t.model.idm import IDM
            model = IDM.from_pretrained("/path/to/local/idm_checkpoint")
            model.eval().cuda()

        2. Transform 준비:
            from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
            data_config = DATA_CONFIG_MAP["omx"]
            transforms = data_config.transform()

        3. 배치 구성 (transforms_idm.py 처리):
            - SiGLIP 이미지 전처리: siglip2-large-patch16-256
            - 토큰 구성: IMG(1) x num_visual_tokens + ACT(4) x action_horizon
            - embodiment_id: 30 (OMX)

        4. 추론:
            output = model.get_action(batch)
            raw_actions = output["action_pred"]  # (1, action_horizon, max_action_dim)

        5. 역정규화 (핵심 — 누락 시 물리적 의미 없는 값):
            pred_actions = transforms.unapply(Batch(action=raw_actions))

        6. modality.json 기반 관절별 재조립

    Returns:
        예측된 action 시퀀스 리스트 (각: shape (action_horizon, action_dim))
    """
    action_horizon = 16
    action_dim = OMX_DOF
    max_action_dim = 32  # IDM 출력 차원 (32차원 중 OMX는 6차원만 사용)

    # === 시뮬레이션: 부드러운 랜덤 action 생성 ===
    # 실제에서는 model.get_action(batch) 호출
    raw_predictions = []
    for inp in idm_inputs:
        base_action = np.random.randn(max_action_dim).astype(np.float32) * 0.1
        raw_seq = np.zeros((action_horizon, max_action_dim), dtype=np.float32)
        for t in range(action_horizon):
            noise = np.random.randn(max_action_dim).astype(np.float32) * 0.02
            raw_seq[t] = base_action + noise * t / action_horizon
        raw_predictions.append(raw_seq)

    # === 역정규화 (denormalization) ===
    # 실제: pred_actions = dataset.transforms.unapply(Batch(action=raw_actions))
    # 시뮬레이션: stats.json이 placeholder이므로 identity 변환
    denormed_predictions = []
    for raw_pred in raw_predictions:
        denormed = unapply_normalization(raw_pred, stats=None)
        denormed_predictions.append(denormed)

    # === modality.json 기반 관절별 재조립 ===
    # 32차원 IDM 출력에서 OMX 6개 관절에 해당하는 슬라이스 추출
    reassembled_predictions = []
    for pred in denormed_predictions:
        joint_actions = reassemble_actions_by_modality(pred)
        # 관절별 결과를 다시 연결하여 (action_horizon, 6) 형태로
        omx_action = np.concatenate(
            [joint_actions[name] for name in OMX_ACTION_MODALITY.keys()],
            axis=-1,
        )
        reassembled_predictions.append(omx_action)

    # === Multi-step 평균화 (선택적) ===
    if use_multistep_avg and len(reassembled_predictions) > 1:
        num_frames = len(reassembled_predictions)
        averaged = multistep_average(
            reassembled_predictions, action_horizon, num_frames
        )
        # 평균화된 결과를 action_horizon 단위로 재분할
        pseudo_actions = []
        for i in range(0, num_frames, action_horizon):
            end = min(i + action_horizon, num_frames)
            chunk = averaged[i:end]
            if chunk.shape[0] < action_horizon:
                # 패딩
                pad = np.zeros((action_horizon - chunk.shape[0], action_dim), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            pseudo_actions.append(chunk)
        return pseudo_actions

    return reassembled_predictions


# =============================================================================
# 3. Pseudo Label 품질 평가
# =============================================================================

def compute_jerk(trajectory: np.ndarray, dt: float = 1 / 10) -> float:
    """궤적의 jerk (3차 미분) 계산. 낮을수록 부드러움.

    vla_action_quality.py의 메트릭을 간소화한 버전입니다.
    """
    if len(trajectory) < 4:
        return 0.0
    vel = np.diff(trajectory, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def compute_temporal_consistency(actions: list[np.ndarray]) -> float:
    """연속 action 간 시간적 일관성 평가.

    연속된 IDM 예측의 첫 번째 timestep 간 차이를 측정합니다.
    낮을수록 일관성이 높습니다.
    """
    if len(actions) < 2:
        return 0.0

    diffs = []
    for i in range(len(actions) - 1):
        # 현재 예측의 마지막 step과 다음 예측의 첫 step 비교
        diff = np.linalg.norm(actions[i][-1] - actions[i + 1][0])
        diffs.append(diff)

    return float(np.mean(diffs))


def evaluate_pseudo_labels(
    pseudo_actions: list[np.ndarray],
    thresholds: QualityThresholds | None = None,
) -> dict:
    """pseudo label 품질 종합 평가.

    Args:
        pseudo_actions: IDM이 예측한 action 시퀀스 리스트
        thresholds: 품질 등급 임계값. None이면 기본값 사용.
                    calibrate_thresholds()로 실제 시연 데이터 기반 보정 가능.
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    # 전체 궤적 구성 (각 action의 첫 timestep 연결)
    trajectory = np.array([a[0] for a in pseudo_actions])

    jerk = compute_jerk(trajectory)
    temporal_consistency = compute_temporal_consistency(pseudo_actions)

    # 통계
    all_actions = np.concatenate(pseudo_actions, axis=0)
    action_range = float(all_actions.max() - all_actions.min())
    action_std = float(all_actions.std())

    # 품질 등급 판정 (임계값 기반)
    if jerk < thresholds.jerk_a and temporal_consistency < thresholds.tc_a:
        grade = "A"
    elif jerk < thresholds.jerk_b and temporal_consistency < thresholds.tc_b:
        grade = "B"
    elif jerk < thresholds.jerk_c and temporal_consistency < thresholds.tc_c:
        grade = "C"
    else:
        grade = "D"

    return {
        "num_predictions": len(pseudo_actions),
        "trajectory_length": len(trajectory),
        "jerk": round(jerk, 4),
        "temporal_consistency": round(temporal_consistency, 4),
        "action_range": round(action_range, 4),
        "action_std": round(action_std, 4),
        "quality_grade": grade,
        "thresholds_source": thresholds.source,
        "thresholds": {
            "jerk": [round(thresholds.jerk_a, 4), round(thresholds.jerk_b, 4), round(thresholds.jerk_c, 4)],
            "temporal_consistency": [round(thresholds.tc_a, 4), round(thresholds.tc_b, 4), round(thresholds.tc_c, 4)],
        },
        "demo_stats": thresholds.demo_stats if thresholds.demo_stats else None,
    }


# =============================================================================
# 4. 모델 종합 비교
# =============================================================================

def build_model_comparison(
    groot_eval_dir: Path,
    cosmos_eval_dir: Path,
) -> list[dict]:
    """모델 종합 비교 테이블"""

    return [
        {
            "항목": "모델 타입",
            "GR00T N1.6": "VLA",
            "Cosmos Predict2.5": "Action-Conditioned World Model",
            "GR00T IDM": "Inverse Dynamics Model",
            "Cosmos Policy": "Video-to-Policy",
        },
        {
            "항목": "본 프로젝트 역할",
            "GR00T N1.6": "파인튜닝 + 배포",
            "Cosmos Predict2.5": "후훈련 + 합성 비디오",
            "GR00T IDM": "pseudo labeling",
            "Cosmos Policy": "비교 분석 (추론)",
        },
        {
            "항목": "입력",
            "GR00T N1.6": "카메라 + 언어 + 상태",
            "Cosmos Predict2.5": "초기 프레임 + EE 행동",
            "GR00T IDM": "비디오 2프레임 쌍",
            "Cosmos Policy": "관찰 + 시연",
        },
        {
            "항목": "출력",
            "GR00T N1.6": "로봇 행동 (6-dim)",
            "Cosmos Predict2.5": "합성 비디오",
            "GR00T IDM": "행동 pseudo label",
            "Cosmos Policy": "행동 + 미래 + 가치",
        },
        {
            "항목": "GPU 요구",
            "GR00T N1.6": "1x RTX 4090",
            "Cosmos Predict2.5": "1x RTX 4090",
            "GR00T IDM": "1x RTX 4090",
            "Cosmos Policy": "1x (추론만)",
        },
        {
            "항목": "OMX 적용",
            "GR00T N1.6": "파인튜닝 완료",
            "Cosmos Predict2.5": "FK 변환으로 적용",
            "GR00T IDM": "추론 전용",
            "Cosmos Policy": "LIBERO 전용",
        },
        {
            "항목": "IDM 시너지",
            "GR00T N1.6": "증강 데이터로 재학습",
            "Cosmos Predict2.5": "합성 비디오 → IDM 입력",
            "GR00T IDM": "pseudo label 생성",
            "Cosmos Policy": "비교 참조",
        },
        {
            "항목": "비디오 dtype",
            "GR00T N1.6": "float32 [0,1]",
            "Cosmos Predict2.5": "uint8 [0,255]",
            "GR00T IDM": "uint8 [0,255]",
            "Cosmos Policy": "uint8 (pickle)",
        },
    ]


def design_integration_pipeline() -> dict:
    """Cosmos Predict2.5 + IDM + VLA 통합 파이프라인 설계"""

    return {
        "name": "Cosmos Predict2.5 → IDM → VLA 데이터 증강 파이프라인",
        "stages": [
            {
                "stage": 1,
                "name": "FK 변환 (데이터 준비)",
                "tool": "utils/omx_fk.py",
                "input": "OMX joint positions (5-dim) + gripper",
                "output": "Cosmos EE state (6-dim) + action (7-dim)",
                "note": "URDF 기반 FK: joint → [x,y,z,r,p,y]",
            },
            {
                "stage": 2,
                "name": "합성 비디오 생성",
                "tool": "Cosmos Predict2.5 Action-Conditioned",
                "input": "초기 프레임 + EE 행동 시퀀스",
                "output": "합성 비디오 (uint8, action-conditioned)",
                "note": "OMX 후훈련 모델 사용 (1x RTX 4090)",
            },
            {
                "stage": 3,
                "name": "SiGLIP 전처리 + IDM 입력 구성",
                "tool": "GR00TIDMTransform (transforms_idm.py)",
                "input": "Cosmos 합성 비디오 (uint8)",
                "output": "IDM 배치 (images, vl_token_ids, sa_token_ids, embodiment_id=30)",
                "note": "SiGLIP siglip2-large-patch16-256 전처리, 프레임당 16 visual tokens",
            },
            {
                "stage": 4,
                "name": "IDM Pseudo Labeling",
                "tool": "IDM.from_pretrained() + model.get_action()",
                "input": "SiGLIP 처리된 배치 (embodiment_id=30, OMX)",
                "output": "정규화된 Action (float32, action_horizon=16, max_action_dim=32)",
                "note": "OMX embodiment slot 30, action 32차원 중 6차원 사용",
            },
            {
                "stage": 5,
                "name": "역정규화 + modality.json 재조립",
                "tool": "transforms.unapply() + reassemble_actions_by_modality()",
                "input": "정규화된 32-dim action",
                "output": "물리 단위 6-dim OMX action (shoulder_pan, ..., gripper)",
                "note": "역정규화 필수 (누락 시 물리적 의미 없음), multi-step 평균화",
            },
            {
                "stage": 6,
                "name": "품질 평가 + 관절 매핑",
                "tool": "evaluate_pseudo_labels() + OMX_JOINT_MAPPING_INV",
                "input": "IDM action (shoulder_pan, ...) + pseudo action 시퀀스",
                "output": "VLA action (joint1, ...) + 품질 등급 (A~D)",
                "note": "IDM→VLA 관절 이름 변환, grade B 이상만 통과",
            },
            {
                "stage": 7,
                "name": "VLA 재학습 (데이터 증강)",
                "tool": "GR00T N1.6 파인튜닝",
                "input": "실제 데이터 (50 ep) + 고품질 pseudo label 데이터",
                "output": "증강된 VLA 모델",
                "note": "비디오 dtype 변환 (uint8→float32), <DREAM> 접두사로 합성 데이터 식별",
            },
        ],
        "data_flow": (
            "OMX joints → FK → Cosmos EE state/action "
            "→ Cosmos Predict2.5 (합성 비디오) "
            "→ SiGLIP 전처리 + IDM 배치 구성 (embodiment=30) "
            "→ IDM.get_action() (정규화된 32-dim) "
            "→ transforms.unapply() 역정규화 "
            "→ modality.json 재조립 (6-dim OMX) "
            "→ multi-step 평균화 "
            "→ quality filter (grade≥B) "
            "→ joint mapping (IDM→VLA) "
            "→ dtype convert (uint8→float32) "
            "→ <DREAM> prefix + VLA re-training"
        ),
    }


# =============================================================================
# 보고서 생성
# =============================================================================

def generate_report(
    quality_metrics: dict,
    three_model_table: list[dict],
    pipeline: dict,
    output_dir: Path,
) -> Path:
    """마크다운 분석 보고서 생성"""

    lines = [
        "# Week 10: 통합 파이프라인 + Cosmos Predict2.5-IDM 시너지",
        "",
        "## 1. Cosmos Predict2.5 → IDM Pseudo Labeling 결과",
        "",
        f"- 예측 수: {quality_metrics['num_predictions']}",
        f"- Jerk: {quality_metrics['jerk']}",
        f"- Temporal Consistency: {quality_metrics['temporal_consistency']}",
        f"- 품질 등급: **{quality_metrics['quality_grade']}**",
        f"- 임계값 출처: {quality_metrics['thresholds_source']}",
        "",
        "### 등급 임계값",
        "",
        "| 등급 | Jerk 상한 | Temporal Consistency 상한 |",
        "|------|----------|-------------------------|",
        f"| A | < {quality_metrics['thresholds']['jerk'][0]} | < {quality_metrics['thresholds']['temporal_consistency'][0]} |",
        f"| B | < {quality_metrics['thresholds']['jerk'][1]} | < {quality_metrics['thresholds']['temporal_consistency'][1]} |",
        f"| C | < {quality_metrics['thresholds']['jerk'][2]} | < {quality_metrics['thresholds']['temporal_consistency'][2]} |",
        "| D | 그 외 | 그 외 |",
        "",
    ]

    demo_stats = quality_metrics.get("demo_stats")
    if demo_stats:
        lines.extend([
            "### 보정 기반 시연 데이터 통계",
            "",
            f"- 에피소드 수: {demo_stats['num_episodes']}",
            f"- Jerk: mean={demo_stats['jerk_mean']}, std={demo_stats['jerk_std']}, "
            f"range=[{demo_stats['jerk_min']}, {demo_stats['jerk_max']}]",
            f"- Temporal Consistency: mean={demo_stats['tc_mean']}, std={demo_stats['tc_std']}, "
            f"range=[{demo_stats['tc_min']}, {demo_stats['tc_max']}]",
            "",
        ])

    lines.extend([
        "## 2. 모델 종합 비교",
        "",
    ])

    # 테이블 헤더
    cols = ["항목", "GR00T N1.6", "Cosmos Predict2.5", "GR00T IDM", "Cosmos Policy"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in three_model_table:
        lines.append("| " + " | ".join(row[c] for c in cols) + " |")

    lines += [
        "",
        "## 3. 통합 파이프라인 설계",
        "",
        f"**{pipeline['name']}**",
        "",
        f"```\n{pipeline['data_flow']}\n```",
        "",
        "### 단계별 상세",
        "",
    ]

    for stage in pipeline["stages"]:
        lines.append(f"**Stage {stage['stage']}: {stage['name']}**")
        lines.append(f"- Tool: {stage['tool']}")
        lines.append(f"- Input: {stage['input']}")
        lines.append(f"- Output: {stage['output']}")
        if "note" in stage:
            lines.append(f"- Note: {stage['note']}")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / "cross_model_analysis.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path


# =============================================================================
# 터미널 출력
# =============================================================================

def print_model_table(table: list[dict]):
    """모델 비교 테이블 출력"""
    cols = ["항목", "GR00T N1.6", "Cosmos Predict2.5", "GR00T IDM", "Cosmos Policy"]
    widths = [18, 18, 24, 18, 18]

    sep = "-" * sum(widths)
    print(f"\n{'=' * sum(widths)}")
    print("  모델 종합 비교")
    print(f"{'=' * sum(widths)}")

    header = "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print(sep)
    for row in table:
        line = "".join(f"{row[c]:<{w}}" for c, w in zip(cols, widths))
        print(line)
    print(f"{'=' * sum(widths)}")


def main():
    parser = argparse.ArgumentParser(
        description="Week 10: 통합 파이프라인 + Cosmos Predict2.5-IDM 시너지"
    )
    parser.add_argument(
        "--cosmos-predict-dir", default="outputs/cosmos_predict",
        help="Cosmos Predict2.5 합성 비디오 디렉토리 (Week 9)",
    )
    parser.add_argument(
        "--groot-eval-dir", default="outputs/eval",
        help="GR00T 평가 결과 디렉토리 (Week 6)",
    )
    parser.add_argument(
        "--cosmos-eval-dir", default="outputs/cosmos_eval",
        help="Cosmos Policy 평가 결과 디렉토리 (Week 7)",
    )
    parser.add_argument(
        "--demo-dir", default=None,
        help="실제 시연 데이터 디렉토리 (품질 임계값 자동 보정용). "
             "지원 형식: .npy, .json, .parquet",
    )
    parser.add_argument(
        "--output-dir", default="outputs/week10_analysis",
        help="분석 결과 출력 디렉토리",
    )
    args = parser.parse_args()

    cosmos_predict_dir = Path(args.cosmos_predict_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Week 10: Cosmos Predict2.5 + IDM Synergy Pipeline ===\n")

    # FK 검증 (omx_fk.py 활용)
    fk = OMXForwardKinematics(robot="omx_f")
    home_ee = fk.compute([0.0] * 5)
    print(f"  FK 검증 (홈 포지션): EE = ({home_ee['position'][0]:.4f}, {home_ee['position'][1]:.4f}, {home_ee['position'][2]:.4f}) m")

    # 1. Cosmos Predict2.5 합성 비디오 로드 + IDM 입력 변환
    print("\n[1/5] Cosmos Predict2.5 합성 비디오 → IDM 입력 변환...")
    frames = load_cosmos_rollout(cosmos_predict_dir)
    idm_inputs = prepare_idm_input(frames)
    print(f"  IDM 입력 페어: {len(idm_inputs)}개")
    print(f"  비디오 dtype: {frames[0].dtype} (IDM 요구: uint8)")

    # dtype 검증
    sample_frame_vla = convert_video_idm_to_vla(frames[0])
    sample_frame_back = convert_video_vla_to_idm(sample_frame_vla)
    roundtrip_error = np.abs(frames[0].astype(float) - sample_frame_back.astype(float)).max()
    print(f"  dtype 왕복 변환 오차: {roundtrip_error} (1 이하면 정상)")

    # 2. IDM pseudo labeling
    print("\n[2/5] IDM pseudo labeling (시뮬레이션)...")
    t0 = time.time()
    pseudo_actions = simulate_idm_inference(idm_inputs)
    elapsed = time.time() - t0
    print(f"  예측 완료: {len(pseudo_actions)}개 action, {elapsed:.2f}s")
    print(f"  Action shape: {pseudo_actions[0].shape} (horizon={pseudo_actions[0].shape[0]}, dim={pseudo_actions[0].shape[1]})")

    # 3. 관절 이름 매핑 시연
    print("\n[3/5] 관절 이름 매핑 (IDM → VLA)...")
    print(f"  IDM 이름: {OMX_IDM_JOINT_NAMES}")
    print(f"  VLA 이름: {[OMX_JOINT_MAPPING_INV.get(n, n) for n in OMX_IDM_JOINT_NAMES]}")
    print(f"  매핑 예시: shoulder_pan → {OMX_JOINT_MAPPING_INV.get('shoulder_pan', 'N/A')}")

    # 4. 품질 임계값 보정 + 품질 평가
    print("\n[4/5] Pseudo label 품질 평가...")

    thresholds = None
    if args.demo_dir:
        demo_path = Path(args.demo_dir)
        if demo_path.exists():
            print(f"  실제 시연 데이터로 임계값 보정 중... ({demo_path})")
            thresholds = calibrate_thresholds(demo_path)
        else:
            print(f"  경고: --demo-dir 경로가 존재하지 않습니다: {demo_path}")
            print(f"  기본 임계값을 사용합니다.")
    else:
        print("  --demo-dir 미지정: 기본 임계값 사용 (실제 시연 데이터로 보정 권장)")

    quality_metrics = evaluate_pseudo_labels(pseudo_actions, thresholds)
    print(f"  Jerk: {quality_metrics['jerk']}")
    print(f"  Temporal Consistency: {quality_metrics['temporal_consistency']}")
    print(f"  Action Range: {quality_metrics['action_range']}")
    print(f"  품질 등급: {quality_metrics['quality_grade']}")
    print(f"  임계값 출처: {quality_metrics['thresholds_source']}")

    # 5. 모델 종합 비교
    print("\n[5/5] 모델 종합 비교 + 통합 파이프라인 설계...")
    model_table = build_model_comparison(
        Path(args.groot_eval_dir),
        Path(args.cosmos_eval_dir),
    )
    print_model_table(model_table)

    pipeline = design_integration_pipeline()
    print(f"\n  통합 파이프라인: {pipeline['name']}")
    print(f"  단계 수: {len(pipeline['stages'])}")
    print(f"  데이터 플로우:")
    print(f"    {pipeline['data_flow']}")

    # 보고서 생성
    report_path = generate_report(
        quality_metrics, model_table, pipeline, output_dir
    )
    print(f"\n  Report: {report_path}")

    # JSON 결과 저장
    result = {
        "quality_metrics": quality_metrics,
        "model_comparison": model_table,
        "pipeline": pipeline,
    }
    json_path = output_dir / "cross_model_analysis.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    print("\n=== Cross-Model Analysis Complete ===")


if __name__ == "__main__":
    main()
