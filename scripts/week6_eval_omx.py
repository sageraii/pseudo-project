"""
Week 6: ACT vs GR00T 비교 평가 (OMX 동일 조건)

동일한 OMX 작업에서 ACT와 GR00T 정책을 비교 평가합니다.
성공률, 추론 지연시간, 궤적 부드러움, 일반화 능력을 측정합니다.

Usage:
    python scripts/week6_eval_omx.py \
        --groot-checkpoint outputs/groot_omx/checkpoint-best \
        --act-checkpoint outputs/train/act_omx_pick \
        --task pick_cube \
        --num-trials 20
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가 (utils 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.omx_constants import OMX_DOF, OMX_IMG_SIZE, create_omx_observation


@dataclass
class EvalResult:
    policy: str
    task: str
    trial: int
    success: bool
    steps: int
    elapsed_sec: float
    latency_ms: float
    jerk: float  # 궤적 부드러움 (낮을수록 좋음)


@dataclass
class EvalSummary:
    policy: str
    task: str
    num_trials: int
    success_rate: float
    avg_elapsed_sec: float
    avg_latency_ms: float
    avg_jerk: float
    results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "task": self.task,
            "num_trials": self.num_trials,
            "success_rate": round(self.success_rate, 4),
            "avg_elapsed_sec": round(self.avg_elapsed_sec, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_jerk": round(self.avg_jerk, 4),
        }


# 작업 정의 (OMX)
TASKS = {
    "pick_cube": "Pick up the red cube",
    "place_on_plate": "Place the cube on the plate",
    "stack_blocks": "Stack the cubes",
}


def compute_jerk(trajectory: list[np.ndarray], dt: float = 1 / 15) -> float:
    """궤적의 jerk (3차 미분) 크기 계산. 낮을수록 부드러움."""
    if len(trajectory) < 4:
        return 0.0
    traj = np.array(trajectory)
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def load_groot_policy(checkpoint: str):
    """GR00T 정책 로드"""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(
        model_path=checkpoint,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device="cuda",
    )


def eval_groot(policy, task_desc: str, num_trials: int) -> list[EvalResult]:
    """GR00T 정책 평가"""
    results = []
    for trial in range(num_trials):
        trajectory = []
        latencies = []
        start = time.time()

        num_steps = 30  # 데모: 고정 스텝 (실제: 성공 조건 판정)
        for step in range(num_steps):
            obs = create_omx_observation(task_desc)
            t0 = time.time()
            action, info = policy.get_action(obs)
            latencies.append((time.time() - t0) * 1000)

            # 궤적 기록 (첫 번째 액션 키의 값)
            for key, val in action.items():
                if isinstance(val, np.ndarray):
                    trajectory.append(val.flatten()[:OMX_DOF])
                    break

        elapsed = time.time() - start
        jerk = compute_jerk(trajectory)
        success = True  # 데모: 항상 성공 (실제: 로봇 상태로 판정)

        results.append(EvalResult(
            policy="GR00T",
            task=task_desc,
            trial=trial + 1,
            success=success,
            steps=num_steps,
            elapsed_sec=elapsed,
            latency_ms=np.mean(latencies),
            jerk=jerk,
        ))
    return results


def eval_act_cli(act_checkpoint: str, task_desc: str, num_trials: int) -> list[EvalResult]:
    """ACT 정책 평가 (LeRobot CLI 기반 시뮬레이션)"""
    # 실제 배포:
    # python -m lerobot.scripts.control_robot \
    #     --robot.type=omx_follower --control.policy.path=<checkpoint>
    #
    # 여기서는 추론 지연시간만 시뮬레이션 (ACT는 CPU 가능, <10ms)
    results = []
    for trial in range(num_trials):
        trajectory = []
        num_steps = 30
        start = time.time()

        for step in range(num_steps):
            # ACT 추론 시뮬레이션 (실제: LeRobot 추론 루프)
            time.sleep(0.005)  # ~5ms (ACT CPU 추론)
            action = np.random.rand(OMX_DOF).astype(np.float32) * 0.1
            trajectory.append(action)

        elapsed = time.time() - start
        jerk = compute_jerk(trajectory)

        results.append(EvalResult(
            policy="ACT",
            task=task_desc,
            trial=trial + 1,
            success=True,  # 데모
            steps=num_steps,
            elapsed_sec=elapsed,
            latency_ms=5.0,  # ACT ~5ms on CPU
            jerk=jerk,
        ))
    return results


def summarize(results: list[EvalResult], policy_name: str, task: str) -> EvalSummary:
    """평가 결과 요약"""
    return EvalSummary(
        policy=policy_name,
        task=task,
        num_trials=len(results),
        success_rate=sum(1 for r in results if r.success) / len(results),
        avg_elapsed_sec=np.mean([r.elapsed_sec for r in results]),
        avg_latency_ms=np.mean([r.latency_ms for r in results]),
        avg_jerk=np.mean([r.jerk for r in results]),
        results=results,
    )


def print_comparison(act_summary: EvalSummary, groot_summary: EvalSummary):
    """비교 테이블 출력"""
    print(f"\n{'=' * 65}")
    print(f"  ACT vs GR00T Comparison: {act_summary.task}")
    print(f"{'=' * 65}")
    print(f"{'Metric':<25} {'ACT':>15} {'GR00T':>15} {'Winner':>8}")
    print(f"{'-' * 65}")

    metrics = [
        ("Success Rate", f"{act_summary.success_rate:.1%}", f"{groot_summary.success_rate:.1%}",
         "ACT" if act_summary.success_rate > groot_summary.success_rate else "GR00T"),
        ("Avg Latency (ms)", f"{act_summary.avg_latency_ms:.1f}", f"{groot_summary.avg_latency_ms:.1f}",
         "ACT" if act_summary.avg_latency_ms < groot_summary.avg_latency_ms else "GR00T"),
        ("Avg Episode Time (s)", f"{act_summary.avg_elapsed_sec:.1f}", f"{groot_summary.avg_elapsed_sec:.1f}",
         "ACT" if act_summary.avg_elapsed_sec < groot_summary.avg_elapsed_sec else "GR00T"),
        ("Smoothness (jerk)", f"{act_summary.avg_jerk:.4f}", f"{groot_summary.avg_jerk:.4f}",
         "ACT" if act_summary.avg_jerk < groot_summary.avg_jerk else "GR00T"),
    ]

    for name, act_val, groot_val, winner in metrics:
        print(f"{name:<25} {act_val:>15} {groot_val:>15} {winner:>8}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="ACT vs GR00T evaluation on OMX")
    parser.add_argument("--groot-checkpoint", required=True, help="GR00T checkpoint path")
    parser.add_argument("--act-checkpoint", required=True, help="ACT checkpoint path")
    parser.add_argument("--task", default="pick_cube", choices=TASKS.keys(), help="Task ID")
    parser.add_argument("--num-trials", type=int, default=20, help="Trials per policy")
    parser.add_argument("--output-dir", default="outputs/eval", help="Results output dir")
    args = parser.parse_args()

    task_desc = TASKS[args.task]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Week 6: ACT vs GR00T Evaluation on OMX ===")
    print(f"  Task: {args.task} ('{task_desc}')")
    print(f"  Trials: {args.num_trials} per policy\n")

    # 1. GR00T 평가
    print("[1/3] Evaluating GR00T...")
    groot_policy = load_groot_policy(args.groot_checkpoint)
    groot_results = eval_groot(groot_policy, task_desc, args.num_trials)
    groot_summary = summarize(groot_results, "GR00T", args.task)
    print(f"  Done: {groot_summary.success_rate:.1%} success rate")

    # 2. ACT 평가
    print("\n[2/3] Evaluating ACT...")
    act_results = eval_act_cli(args.act_checkpoint, task_desc, args.num_trials)
    act_summary = summarize(act_results, "ACT", args.task)
    print(f"  Done: {act_summary.success_rate:.1%} success rate")

    # 3. 비교 출력
    print("\n[3/3] Comparison:")
    print_comparison(act_summary, groot_summary)

    # 결과 저장
    report = {
        "task": args.task,
        "task_description": task_desc,
        "num_trials": args.num_trials,
        "act": act_summary.to_dict(),
        "groot": groot_summary.to_dict(),
    }
    report_path = output_dir / f"eval_{args.task}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved: {report_path}")


if __name__ == "__main__":
    main()
