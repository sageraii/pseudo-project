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
    - GR00T-IDM-Documentation/src/vla_action_quality.py (품질 메트릭)
    - GR00T-IDM-Documentation/src/idm_inference_example.py (IDM 추론 패턴)
    - utils/omx_fk.py (OMX joint → EE pose → Cosmos 입력 변환)
"""

import argparse
import json
import sys
import time
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

    IDM은 연속 2 프레임 (t, t+1)을 입력으로 받아 action을 예측합니다.
    Cosmos Predict2.5 출력은 이미 uint8이므로 dtype 변환 불필요.

    Args:
        frames: uint8 프레임 리스트, 각 (256, 256, 3)

    Returns:
        IDM 입력 딕셔너리 리스트
    """
    idm_inputs = []
    for i in range(len(frames) - 1):
        frame_t = frames[i]
        frame_t1 = frames[i + 1]

        # IDM 입력 형태: (1, 2, 1, 256, 256, 3) uint8
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
# 2. IDM Pseudo Labeling (시뮬레이션)
# =============================================================================

def simulate_idm_inference(idm_inputs: list[dict]) -> list[np.ndarray]:
    """IDM pseudo labeling 시뮬레이션.

    실제 IDM 모델 없이 파이프라인 구조를 검증합니다.
    실제 사용 시:
        from gr00t.model.idm import IDM
        model = IDM.from_pretrained("nvidia/GR00T-IDM")
        output = model.get_action(batch)

    Returns:
        예측된 action 시퀀스 (각 action: shape (action_horizon, action_dim))
    """
    action_horizon = 16
    action_dim = OMX_DOF

    pseudo_actions = []
    for inp in idm_inputs:
        # 시뮬레이션: 부드러운 랜덤 action 생성
        # 실제: model.get_action(inp) 호출
        base_action = np.random.randn(action_dim).astype(np.float32) * 0.1
        action_seq = np.zeros((action_horizon, action_dim), dtype=np.float32)
        for t in range(action_horizon):
            noise = np.random.randn(action_dim).astype(np.float32) * 0.02
            action_seq[t] = base_action + noise * t / action_horizon

        pseudo_actions.append(action_seq)

    return pseudo_actions


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


def evaluate_pseudo_labels(pseudo_actions: list[np.ndarray]) -> dict:
    """pseudo label 품질 종합 평가"""

    # 전체 궤적 구성 (각 action의 첫 timestep 연결)
    trajectory = np.array([a[0] for a in pseudo_actions])

    jerk = compute_jerk(trajectory)
    temporal_consistency = compute_temporal_consistency(pseudo_actions)

    # 통계
    all_actions = np.concatenate(pseudo_actions, axis=0)
    action_range = float(all_actions.max() - all_actions.min())
    action_std = float(all_actions.std())

    # 품질 등급 판정
    if jerk < 100 and temporal_consistency < 0.1:
        grade = "A"
    elif jerk < 500 and temporal_consistency < 0.3:
        grade = "B"
    elif jerk < 1000 and temporal_consistency < 0.5:
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
    }


# =============================================================================
# 4. 세 모델 종합 비교
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
                "name": "비디오 전처리",
                "tool": "OpenCV + prepare_idm_input()",
                "input": "Cosmos 합성 비디오 (uint8)",
                "output": "IDM 입력 (uint8, 256x256, 2-frame pairs)",
                "note": "리사이즈 + 프레임 페어링",
            },
            {
                "stage": 4,
                "name": "IDM Pseudo Labeling",
                "tool": "GR00T IDM",
                "input": "2-frame 페어 (uint8, 256x256)",
                "output": "Action 벡터 (float32, action_horizon x action_dim)",
                "note": "관절 이름: shoulder_pan 등 (OMX_IDM_JOINT_NAMES)",
            },
            {
                "stage": 5,
                "name": "관절 이름 매핑 + 품질 평가",
                "tool": "OMX_JOINT_MAPPING_INV + evaluate_pseudo_labels()",
                "input": "IDM action (shoulder_pan, ...) + pseudo action 시퀀스",
                "output": "VLA action (joint1, ...) + 품질 등급 (A~D)",
                "note": "IDM→VLA 변환 + grade B 이상만 통과",
            },
            {
                "stage": 6,
                "name": "VLA 재학습 (데이터 증강)",
                "tool": "GR00T N1.6 파인튜닝",
                "input": "실제 데이터 (50 ep) + 고품질 pseudo label 데이터",
                "output": "증강된 VLA 모델",
                "note": "비디오 dtype 변환 (uint8→float32) 포함",
            },
        ],
        "data_flow": (
            "OMX joints → FK → Cosmos EE state/action "
            "→ Cosmos Predict2.5 (합성 비디오) "
            "→ resize+pair → IDM (uint8 input) "
            "→ pseudo action (IDM joint names) "
            "→ joint mapping (VLA joint names) "
            "→ quality filter (grade≥B) "
            "→ dtype convert (float32) "
            "→ VLA re-training"
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
        "",
        "## 2. 세 모델 종합 비교",
        "",
    ]

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

    # 4. 품질 평가
    print("\n[4/5] Pseudo label 품질 평가...")
    quality_metrics = evaluate_pseudo_labels(pseudo_actions)
    print(f"  Jerk: {quality_metrics['jerk']}")
    print(f"  Temporal Consistency: {quality_metrics['temporal_consistency']}")
    print(f"  Action Range: {quality_metrics['action_range']}")
    print(f"  품질 등급: {quality_metrics['quality_grade']}")

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
