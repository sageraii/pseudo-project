"""
Week 3: GR00T N1.6 추론 테스트 (OMX 6-dim)

OMX 매니퓰레이터의 6차원 액션/상태 공간으로 GR00T 사전훈련 모델을 테스트합니다.
실제 로봇 없이 더미 데이터로 추론 파이프라인이 동작하는지 검증합니다.

Usage:
    cd Isaac-GR00T
    uv run python ../pseudo-project/scripts/week3_test_groot_inference.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가 (utils 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from utils.omx_constants import OMX_DOF, OMX_IMG_SIZE, create_omx_observation

NUM_TEST_STEPS = 10


def main():
    print("=== Week 3: GR00T N1.6 Inference Test (OMX 6-dim) ===\n")

    # 1. 모델 로드
    print("[1/4] Loading GR00T N1.6 model...")
    t0 = time.time()
    policy = Gr00tPolicy(
        model_path="nvidia/GR00T-N1.6-3B",
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device="cuda",
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s\n")

    # 2. 단일 추론 테스트
    print("[2/4] Single inference test...")
    obs = create_omx_observation("pick up the red cube")
    action, info = policy.get_action(obs)

    print(f"  Action keys: {list(action.keys())}")
    for key, val in action.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
    print()

    # 3. 추론 속도 벤치마크
    print(f"[3/4] Inference speed benchmark ({NUM_TEST_STEPS} steps)...")
    latencies = []
    for i in range(NUM_TEST_STEPS):
        obs = create_omx_observation("pick up the red cube")
        t0 = time.time()
        action, info = policy.get_action(obs)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    freq = 1000 / avg_latency

    print(f"  Average latency: {avg_latency:.1f} +/- {std_latency:.1f} ms")
    print(f"  Inference frequency: {freq:.1f} Hz")
    print(f"  Min/Max: {min(latencies):.1f} / {max(latencies):.1f} ms\n")

    # 4. 다양한 작업 테스트
    print("[4/4] Multi-task inference test...")
    tasks = [
        "pick up the red cube",
        "place the cube on the plate",
        "stack the cubes",
    ]
    for task in tasks:
        obs = create_omx_observation(task)
        action, info = policy.get_action(obs)
        action_norm = np.mean([np.linalg.norm(v) for v in action.values() if isinstance(v, np.ndarray)])
        print(f"  Task: '{task}' -> action norm: {action_norm:.4f}")

    print("\n=== GR00T Inference Test Complete ===")
    print(f"Summary:")
    print(f"  - Action space: {OMX_DOF}-dim (OMX 5-DOF + gripper)")
    print(f"  - Avg latency: {avg_latency:.1f} ms ({freq:.1f} Hz)")
    print(f"  - OMX ros2_control: 100 Hz -> {'OK' if freq > 10 else 'WARNING: too slow'}")


if __name__ == "__main__":
    main()
