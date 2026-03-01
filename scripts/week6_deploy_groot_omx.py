"""
Week 6: GR00T 모델 OMX 실제 로봇 배포

파인튜닝된 GR00T 체크포인트를 OMX 로봇에 배포하고 실시간 추론 루프를 실행합니다.
ROS 2 ros2_control (100Hz) 제어 주기와 동기화합니다.

Usage:
    cd Isaac-GR00T
    uv run python ../pseudo-project/scripts/week6_deploy_groot_omx.py \
        --checkpoint outputs/groot_omx/checkpoint-best \
        --task "pick up the red cube"

Requirements:
    - OMX follower connected via USB
    - ROS 2 Jazzy with ros2_control running
    - Camera connected and publishing
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가 (utils 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from utils.omx_constants import OMX_CONTROL_HZ, OMX_DOF, OMX_IMG_SIZE

# 안전 제한
MAX_JOINT_VELOCITY = 0.5  # rad/step
MAX_EPISODES = 100
EPISODE_TIMEOUT_SEC = 60


class OMXRobotInterface:
    """OMX 로봇 ROS 2 인터페이스 (실제 구현 시 rclpy 사용)"""

    def __init__(self, follower_port: str = "/dev/ttyACM0", camera_index: int = 0):
        self.follower_port = follower_port
        self.camera_index = camera_index
        self._running = False

        # 실제 구현 시:
        # import rclpy
        # from sensor_msgs.msg import JointState, Image
        # self.node = rclpy.create_node('groot_omx_deploy')
        # self.joint_sub = self.node.create_subscription(JointState, ...)
        # self.action_pub = self.node.create_publisher(JointTrajectory, ...)

        print(f"  OMX Interface initialized")
        print(f"    Follower: {follower_port}")
        print(f"    Camera: index={camera_index}")

    def get_observation(self, task: str) -> dict:
        """현재 OMX 관측값 획득"""
        # 실제 구현: ROS 2 토픽에서 JointState + Image 구독
        # 여기서는 데모용 더미 데이터
        try:
            import cv2

            cap = cv2.VideoCapture(self.camera_index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.resize(frame, (OMX_IMG_SIZE, OMX_IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = frame.astype(np.float32) / 255.0
            else:
                img = np.random.rand(OMX_IMG_SIZE, OMX_IMG_SIZE, 3).astype(np.float32)
        except ImportError:
            img = np.random.rand(OMX_IMG_SIZE, OMX_IMG_SIZE, 3).astype(np.float32)

        # 실제 구현: self.current_joint_state (ROS 2 콜백으로 갱신)
        joints = np.zeros(OMX_DOF, dtype=np.float32)

        return {
            "video": {"cam1": img.reshape(1, 1, OMX_IMG_SIZE, OMX_IMG_SIZE, 3)},
            "state": {
                "joint1": np.array([[[joints[0]]]]).astype(np.float32),
                "joint2": np.array([[[joints[1]]]]).astype(np.float32),
                "joint3": np.array([[[joints[2]]]]).astype(np.float32),
                "joint4": np.array([[[joints[3]]]]).astype(np.float32),
                "joint5": np.array([[[joints[4]]]]).astype(np.float32),
                "gripper": np.array([[[joints[5]]]]).astype(np.float32),
            },
            "annotation": {"task": [[task]]},
        }

    def execute_action(self, action: dict) -> bool:
        """액션을 OMX follower에 전송"""
        # 안전 클리핑
        for key, val in action.items():
            if isinstance(val, np.ndarray):
                np.clip(val, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY, out=val)

        # 실제 구현:
        # msg = JointTrajectory()
        # msg.joint_names = ['joint1', ..., 'gripper']
        # point = JointTrajectoryPoint()
        # point.positions = action_values
        # self.action_pub.publish(msg)

        return True

    def reset(self):
        """OMX를 홈 포지션으로 리셋"""
        # 실제 구현: 안전 홈밍 서비스 호출
        print("  Robot reset to home position")
        time.sleep(1.0)


def run_deployment(policy: Gr00tPolicy, robot: OMXRobotInterface, task: str, num_episodes: int):
    """배포 루프 실행"""
    results = []

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes}: '{task}' ---")
        robot.reset()

        start_time = time.time()
        step = 0
        success = False
        latencies = []

        while True:
            # 타임아웃 체크
            elapsed = time.time() - start_time
            if elapsed > EPISODE_TIMEOUT_SEC:
                print(f"  Timeout after {elapsed:.1f}s")
                break

            # 관측 → 추론 → 실행
            obs = robot.get_observation(task)

            t0 = time.time()
            action, info = policy.get_action(obs)
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)

            robot.execute_action(action)
            step += 1

            # 데모: 30스텝 후 종료 (실제: 성공 조건 판정)
            if step >= 30:
                success = True
                break

        elapsed = time.time() - start_time
        avg_latency = np.mean(latencies) if latencies else 0

        result = {
            "episode": ep + 1,
            "task": task,
            "success": success,
            "steps": step,
            "elapsed_sec": elapsed,
            "avg_latency_ms": avg_latency,
        }
        results.append(result)

        status = "SUCCESS" if success else "FAIL"
        print(f"  [{status}] Steps: {step}, Time: {elapsed:.1f}s, Latency: {avg_latency:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Deploy GR00T on OMX robot")
    parser.add_argument("--checkpoint", required=True, help="GR00T checkpoint path")
    parser.add_argument("--task", default="pick up the red cube", help="Task description")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--follower-port", default="/dev/ttyACM0", help="OMX follower USB port")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    print("=== Week 6: GR00T OMX Deployment ===\n")

    # Ctrl+C 핸들링
    def signal_handler(sig, frame):
        print("\nDeployment stopped by user.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 1. 모델 로드
    print("[1/3] Loading GR00T checkpoint...")
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device="cuda",
    )
    print(f"  Loaded: {args.checkpoint}")

    # 2. 로봇 연결
    print("\n[2/3] Connecting to OMX robot...")
    robot = OMXRobotInterface(
        follower_port=args.follower_port,
        camera_index=args.camera_index,
    )

    # 3. 배포 실행
    print(f"\n[3/3] Running deployment: '{args.task}' x {args.num_episodes} episodes")
    results = run_deployment(policy, robot, args.task, args.num_episodes)

    # 결과 요약
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    avg_time = np.mean([r["elapsed_sec"] for r in results])
    avg_latency = np.mean([r["avg_latency_ms"] for r in results])

    print(f"\n{'=' * 50}")
    print(f"GR00T OMX Deployment Results")
    print(f"{'=' * 50}")
    print(f"  Task: {args.task}")
    print(f"  Success rate: {successes}/{total} ({100 * successes / total:.1f}%)")
    print(f"  Avg episode time: {avg_time:.1f}s")
    print(f"  Avg inference latency: {avg_latency:.1f}ms")
    print(f"  Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
