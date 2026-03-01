"""
Week 7: Cosmos Policy LIBERO 벤치마크 추론

사전 훈련된 Cosmos Policy 모델로 LIBERO 4개 서브스위트를 평가합니다.
Docker 내부에서 실행해야 합니다.

Usage (Docker 내부):
    python scripts/week7_eval_cosmos_libero.py \
        --suites spatial object goal long \
        --num-episodes 20

Requirements:
    - cosmos-policy Docker 컨테이너 내부에서 실행
    - GPU with 6-10 GB VRAM
"""

import argparse
import json
import sys
import time
from pathlib import Path

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
PAPER_RESULTS = {
    "libero_spatial": 98.1,
    "libero_object": 100.0,
    "libero_goal": 98.2,
    "libero_long": 97.6,
}


def eval_suite(suite: str, num_episodes: int, ckpt_path: str) -> dict:
    """단일 LIBERO 서브스위트 평가"""
    print(f"\n--- Evaluating: {suite} ({num_episodes} episodes/task) ---")

    try:
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action,
            get_model,
            load_dataset_stats,
        )
        from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

        cfg = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=ckpt_path,
            libero_task_suite=suite,
            num_episodes_per_task=num_episodes,
        )

        model, cosmos_config = get_model(cfg)
        dataset_stats = load_dataset_stats(cfg)

        # 실제 평가 실행
        # 여기서는 구조만 보여줌 - 실제로는 LIBERO 환경 필요
        print(f"  Model loaded: {ckpt_path}")
        print(f"  Config: {cfg.config}")
        print(f"  Suite: {suite}")
        print(f"  Episodes per task: {num_episodes}")

        # 실제 LIBERO 평가는 eval.sh 스크립트로 실행:
        # bash eval.sh cosmos_predict2_2b_480p_libero__inference_only \
        #     nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        #     libero_spatial 20
        result = {
            "suite": suite,
            "num_episodes": num_episodes,
            "status": "model_loaded",
            "paper_result": PAPER_RESULTS.get(suite, 0),
            "note": "Run eval.sh for full LIBERO evaluation",
        }
        return result

    except ImportError as e:
        print(f"  WARNING: Cosmos Policy not available: {e}")
        print(f"  Must run inside cosmos-policy Docker container.")
        return {
            "suite": suite,
            "num_episodes": num_episodes,
            "status": "import_error",
            "error": str(e),
            "paper_result": PAPER_RESULTS.get(suite, 0),
        }


def main():
    parser = argparse.ArgumentParser(description="Cosmos Policy LIBERO evaluation")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["spatial"],
        choices=["spatial", "object", "goal", "long", "all"],
        help="LIBERO suites to evaluate",
    )
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument(
        "--ckpt-path",
        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        help="Model checkpoint path",
    )
    parser.add_argument("--output-dir", default="outputs/cosmos_eval", help="Output dir")
    args = parser.parse_args()

    # 'all' 확장
    if "all" in args.suites:
        suites = LIBERO_SUITES
    else:
        suites = [f"libero_{s}" for s in args.suites]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Week 7: Cosmos Policy LIBERO Evaluation ===")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  Suites: {', '.join(suites)}")
    print(f"  Episodes/task: {args.num_episodes}")

    # 평가 실행
    all_results = {}
    for suite in suites:
        result = eval_suite(suite, args.num_episodes, args.ckpt_path)
        all_results[suite] = result

    # 결과 요약
    print(f"\n{'=' * 55}")
    print(f"  Cosmos Policy LIBERO Results Summary")
    print(f"{'=' * 55}")
    print(f"{'Suite':<20} {'Paper':>10} {'Status':>15}")
    print(f"{'-' * 55}")
    for suite, result in all_results.items():
        paper = f"{result['paper_result']:.1f}%"
        status = result.get("status", "unknown")
        print(f"{suite:<20} {paper:>10} {status:>15}")
    print(f"{'=' * 55}")

    # 저장
    report_path = output_dir / "cosmos_libero_eval.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {report_path}")

    # eval.sh 안내
    print(f"\nFor full evaluation, run inside Docker:")
    for suite in suites:
        short = suite.replace("libero_", "")
        print(f"  bash eval.sh cosmos_predict2_2b_480p_libero__inference_only \\")
        print(f"      {args.ckpt_path} {short} {args.num_episodes}")


if __name__ == "__main__":
    main()
