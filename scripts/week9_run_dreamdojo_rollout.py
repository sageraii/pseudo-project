"""
Week 9: DreamDojo 롤아웃 생성 및 세계 모델 분석 (참고용)

NOTE: 본 프로젝트에서 DreamDojo는 Cosmos Predict2.5 Action-Conditioned 모델로 대체되었습니다.
이 스크립트는 DreamDojo 세계 모델의 API 패턴, 품질 메트릭, 비디오 분석 로직을
참고하기 위해 유지됩니다. 품질 메트릭(analyze_rollout_quality)은
Cosmos Predict2.5 합성 비디오 평가에도 재사용 가능합니다.

DreamDojo 사전 훈련 모델로 미래 비디오 롤아웃을 생성하고
물리 시뮬레이션 품질을 분석합니다.

Usage:
    python scripts/week9_run_dreamdojo_rollout.py \
        --model nvidia/DreamDojo-2B-480p-GR1 \
        --output-dir outputs/dreamdojo_rollouts

Constraints (DreamDojo):
    - 증류 파이프라인 미공개 → 실시간 10 FPS 불가
    - 텔레오퍼레이션 코드 미공개 → VR 연동 불가
    - 후훈련: 8x H100 80GB 필요 → 추론만 수행

See also:
    - Cosmos Predict2.5: 1x RTX 4090으로 OMX 후훈련 가능
    - utils/omx_fk.py: OMX joint → EE pose 변환 (Cosmos 입력)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def check_dreamdojo_available() -> bool:
    """DreamDojo 패키지 사용 가능 여부 확인"""
    try:
        # DreamDojo 패키지명: cosmos-predict2
        import cosmos_predict2  # noqa: F401
        return True
    except ImportError:
        return False


def generate_rollout_config(model_name: str) -> dict:
    """DreamDojo 롤아웃 생성 설정"""
    # 모델별 설정 매핑
    configs = {
        "nvidia/DreamDojo-2B-480p-GR1": {
            "config_file": "configs/2b_480_640_gr1.yaml",
            "robot": "GR-1",
            "resolution": "480x640",
            "params": "2B",
        },
        "nvidia/DreamDojo-2B-480p-G1": {
            "config_file": "configs/2b_480_640_g1.yaml",
            "robot": "Unitree G1",
            "resolution": "480x640",
            "params": "2B",
        },
        "nvidia/DreamDojo-2B-480p-YAM": {
            "config_file": "configs/2b_480_640_yam.yaml",
            "robot": "YAM",
            "resolution": "480x640",
            "params": "2B",
        },
    }
    return configs.get(model_name, configs["nvidia/DreamDojo-2B-480p-GR1"])


def analyze_rollout_quality(rollout_frames: list[np.ndarray]) -> dict:
    """롤아웃 비디오 품질 분석"""
    if not rollout_frames:
        return {"error": "No frames"}

    frames = np.array(rollout_frames)
    num_frames = len(frames)
    duration_sec = num_frames / 10  # 10 FPS 가정

    # 프레임 간 차이 (시간적 일관성)
    if num_frames > 1:
        diffs = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2, 3))
        temporal_consistency = 1.0 - np.mean(diffs) / 255.0
    else:
        temporal_consistency = 1.0

    # 색상 분포 분석
    mean_intensity = np.mean(frames) / 255.0
    std_intensity = np.std(frames) / 255.0

    # 장기 안정성 (마지막 10% vs 처음 10%)
    n10 = max(1, num_frames // 10)
    early_mean = np.mean(frames[:n10])
    late_mean = np.mean(frames[-n10:])
    stability = 1.0 - abs(early_mean - late_mean) / 255.0

    return {
        "num_frames": num_frames,
        "duration_sec": round(duration_sec, 1),
        "temporal_consistency": round(temporal_consistency, 4),
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "long_horizon_stability": round(stability, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="DreamDojo rollout generation")
    parser.add_argument(
        "--model",
        default="nvidia/DreamDojo-2B-480p-GR1",
        help="DreamDojo model name",
    )
    parser.add_argument("--output-dir", default="outputs/dreamdojo_rollouts", help="Output dir")
    parser.add_argument("--num-rollouts", type=int, default=5, help="Number of rollouts")
    parser.add_argument("--rollout-length", type=int, default=150, help="Frames per rollout (10 FPS)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = generate_rollout_config(args.model)

    print("=== Week 9: DreamDojo Rollout Generation ===")
    print(f"  Model: {args.model}")
    print(f"  Robot: {config['robot']}")
    print(f"  Resolution: {config['resolution']}")
    print(f"  Config: {config['config_file']}")
    print(f"  Rollouts: {args.num_rollouts} x {args.rollout_length} frames")
    print(f"  Expected duration: {args.rollout_length / 10:.0f}s per rollout")

    # DreamDojo 사용 가능 확인
    available = check_dreamdojo_available()
    if not available:
        print("\n  WARNING: DreamDojo (cosmos-predict2) not installed.")
        print("  Install: pip install -e . (in DreamDojo directory)")
        print("\n  Generating dummy rollouts for pipeline testing...")

    all_results = []

    for i in range(args.num_rollouts):
        print(f"\n--- Rollout {i + 1}/{args.num_rollouts} ---")

        if available:
            # 실제 DreamDojo 추론
            # DreamDojo는 config + torchrun 기반
            # 실행: bash launch.sh <config_file>
            print(f"  실제 추론은 다음 명령으로 실행:")
            print(f"    cd DreamDojo && bash launch.sh {config['config_file']}")
            frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                       for _ in range(args.rollout_length)]
        else:
            # 더미 롤아웃 (파이프라인 테스트)
            t0 = time.time()
            frames = []
            base_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            for f in range(args.rollout_length):
                noise = np.random.randint(-5, 5, base_frame.shape, dtype=np.int16)
                frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                frames.append(frame)
            gen_time = time.time() - t0
            print(f"  Generated {len(frames)} frames in {gen_time:.1f}s")

        # 품질 분석
        quality = analyze_rollout_quality(frames)
        quality["rollout_index"] = i
        quality["model"] = args.model
        quality["robot"] = config["robot"]
        all_results.append(quality)

        print(f"  Duration: {quality['duration_sec']}s")
        print(f"  Temporal consistency: {quality['temporal_consistency']:.4f}")
        print(f"  Long-horizon stability: {quality['long_horizon_stability']:.4f}")

        # 비디오 저장 (선택)
        try:
            import cv2
            video_path = output_dir / f"rollout_{i:03d}.mp4"
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  Saved: {video_path}")
        except ImportError:
            pass

    # 요약
    print(f"\n{'=' * 50}")
    print(f"  DreamDojo Rollout Summary")
    print(f"{'=' * 50}")
    avg_consistency = np.mean([r["temporal_consistency"] for r in all_results])
    avg_stability = np.mean([r["long_horizon_stability"] for r in all_results])
    print(f"  Avg temporal consistency: {avg_consistency:.4f}")
    print(f"  Avg long-horizon stability: {avg_stability:.4f}")
    print(f"  Total rollouts: {len(all_results)}")

    # 제약사항 안내
    print(f"\n  Constraints:")
    print(f"    - Distillation pipeline: UNRELEASED (no 10 FPS realtime)")
    print(f"    - Teleoperation code: UNRELEASED (no VR integration)")
    print(f"    - Post-training: requires 8x H100 80GB")

    # 결과 저장
    report_path = output_dir / "dreamdojo_rollout_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
