# 실전 프로젝트 커리큘럼: 12주 로드맵

> 관련 문서: [Project.md](Project.md) — 모델 분석 및 전략 | [PIPELINE.md](PIPELINE.md) — IDM/Cosmos 데이터 파이프라인

## 8. 실전 프로젝트 커리큘럼: 12주 로드맵

### 프로젝트 개요

**목표**: ROBOTIS OMX 매니퓰레이터로 모방학습 데이터를 수집하고, ACT 베이스라인 및 GR00T N1.6 VLA 파인튜닝을 수행하며, Cosmos Predict2.5 세계 모델과 IDM 시너지 파이프라인을 통해 합성 데이터 증강까지 달성

**접근 방식**: Option C+ 하이브리드 (OMX 기반 + Cosmos Predict2.5 통합)
- **Week 1-2**: OMX 하드웨어 조립 + Physical AI Tools/LeRobot 환경 구축
- **Week 3-4**: GR00T N1.6 환경 설정 + OMX로 모방학습 데이터 수집
- **Week 5-6**: ACT 베이스라인 + GR00T 파인튜닝 + OMX 배포 비교
- **Week 7-8**: Cosmos Policy LIBERO 평가 + **Cosmos Predict2.5 Action-Conditioned 환경 구축** (병렬)
- **Week 9-10**: Cosmos 데이터 증강 + **IDM 시너지 파이프라인 구축** (병렬) ※실증 결과 반영: [PIPELINE.md](PIPELINE.md) 참조
- **Week 11-12**: VLA 데이터 증강 재학습 + 5개 모델 비교 분석 + 프로젝트 마무리

**환경 및 도구:**

- **하드웨어**: ROBOTIS OMX-AI (OMX-L 리더 + OMX-F 팔로워, 5-DOF + 1 Gripper)
- **소프트웨어**: ROS 2 Jazzy + Physical AI Tools (Docker) + ROBOTIS 포크 LeRobot
- **시뮬레이션**: Gazebo/RViz (URDF), MuJoCo (옵션)
- **AI 모델**: ACT (베이스라인), GR00T N1.6 (파인튜닝), Cosmos Policy (추론), Cosmos Predict2.5 (후훈련+추론), GR00T IDM (pseudo labeling)
- **프로젝트 목표 작업**: 객체 조작 및 배치 작업 (Pick and Place++)
- **데이터 수집**: 15 FPS, HuggingFace datasets 포맷, 6차원 액션 (5관절 + 그리퍼)
- **GPU 요구사항**: GR00T 파인튜닝/Cosmos Predict2.5 후훈련/추론 1x RTX 4090+, ACT 학습용 1x RTX 4090+

**OMX 하드웨어 사양:**

| 구분 | OMX-L (리더) | OMX-F (팔로워) |
|------|-------------|---------------|
| DOF | 5 + 1 Gripper | 5 + 1 Gripper |
| 작업 반경 | 335mm | 400mm |
| 무게 | 360g | 560g |
| 페이로드 | - | 100~250g |
| 전원 | 5V USB-C | 12VDC |
| 모터 | XL330 | XL430 + XL330 |
| 통신 | TTL, 1Mbps | TTL, 1Mbps |

---

### Week 1-2: 환경 설정 및 기초 인프라 구축

#### Week 1: OMX 하드웨어 조립 및 Physical AI Tools 환경 구축

**학습 목표:**

- OMX-AI (리더 + 팔로워) 하드웨어 조립 및 캘리브레이션
- Physical AI Tools Docker 환경 구축
- ROS 2 Jazzy 기반 제어 아키텍처 이해

**실습 내용:**

1. **OMX 하드웨어 조립**

   - OMX-L (리더) 및 OMX-F (팔로워) 조립 (공장 캘리브레이션 완료)
   - USB-C 연결 및 DYNAMIXEL 모터 ID 확인
   - 안전 홈밍 기능 테스트 (초기 자세 복귀)

2. **Physical AI Tools Docker 환경 구축**

   ```bash
   # 사전 요구사항: Docker Engine + NVIDIA Container Toolkit
   sudo usermod -aG docker $USER
   sudo systemctl enable docker

   # Open Manipulator 컨테이너
   git clone https://github.com/ROBOTIS-GIT/open_manipulator
   cd open_manipulator/docker && ./container.sh start

   # Physical AI Tools 컨테이너
   git clone --recurse-submodules https://github.com/ROBOTIS-GIT/physical_ai_tools.git
   cd physical_ai_tools/docker && ./container.sh start
   ```

3. **USB 포트 및 카메라 설정**

   ```bash
   # 컨테이너 진입 후 USB 디바이스 확인
   ./container.sh enter
   ls -al /dev/serial/by-id/

   # Leader/Follower 포트 설정
   # omx_l_leader_ai.launch.py → Leader serial ID 입력
   # omx_f_follower_ai.launch.py → Follower serial ID 입력

   # 카메라 토픽 설정 (compressed 필수)
   sudo nano ~/ros2_ws/src/physical_ai_tools/physical_ai_server/config/omx_f_config.yaml
   # 토픽 예: camera1/image_raw/compressed
   ```

4. **ROS 2 제어 아키텍처 확인**

   - `ros2_control` 프레임워크 (100Hz 조인트 제어)
   - 파이프라인: 입력 소스 → ROS 2 JointTrajectory → controller_manager → DynamixelHardwareInterface → TTL
   - arm_controller (5-DOF), gpio_command_controller (그리퍼)

**결과물:**

- ✅ OMX-AI 하드웨어 조립 및 동작 확인
- ✅ Physical AI Tools + Open Manipulator Docker 컨테이너 실행
- ✅ 카메라 스트리밍 및 ROS 2 토픽 확인

**학습 자료:**

- https://ai.robotis.com/omx/assembly_guide_omx.html
- https://ai.robotis.com/omx/setup_guide_physical_ai_tools.html
- https://ai.robotis.com/omx/hardware_omx.html

---

#### Week 2: OMX 텔레오퍼레이션 및 LeRobot 설정

**학습 목표:**

- OMX 리더-팔로워 텔레오퍼레이션 실습
- ROBOTIS 포크 LeRobot 설치 및 CLI 워크플로우 이해
- 데이터 수집 파이프라인 이해 (Physical AI Tools + LeRobot 두 경로)

**실습 내용:**

1. **ROBOTIS 포크 LeRobot 설치**

   ```bash
   # ⚠️ 원본 LeRobot이 아닌 ROBOTIS 포크 사용 필수
   conda create -y -n lerobot python=3.10
   conda activate lerobot
   conda install -c conda-forge ffmpeg=6.1.1 -y

   git clone https://github.com/ROBOTIS-GIT/lerobot.git
   cd lerobot
   pip install -e ".[dynamixel]"
   ```

2. **OMX 텔레오퍼레이션 테스트 (LeRobot CLI)**

   ```bash
   # USB 포트 확인
   lerobot-find-port
   # 예: 팔로워 /dev/ttyACM0, 리더 /dev/ttyACM1

   # 텔레오퍼레이션 실행
   python -m lerobot.teleoperate \
       --robot.type=omx_follower \
       --robot.port=/dev/ttyACM0 \
       --teleop.type=omx_leader \
       --teleop.port=/dev/ttyACM1

   # 카메라 통합 텔레오퍼레이션 (OpenCV, 640x480, 30fps)
   python -m lerobot.teleoperate \
       --robot.type=omx_follower \
       --robot.port=/dev/ttyACM0 \
       --teleop.type=omx_leader \
       --teleop.port=/dev/ttyACM1 \
       --robot.cameras='[{type=opencv, key=cam1, index=0, width=640, height=480, fps=30}]'
   ```

3. **Physical AI Tools 웹 UI 텔레오퍼레이션**

   ```bash
   # Physical AI Server 실행 (Docker 내부)
   cd physical_ai_tools/docker
   ./container.sh enter
   ai_server

   # OMX Follower 노드 실행 (별도 터미널)
   cd open_manipulator/docker && ./container.sh enter
   ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py
   ```
   - 브라우저에서 `http://localhost` 접속 → 로봇 타입 선택 → 텔레오퍼레이션

4. **샘플 데이터 녹화 테스트 (LeRobot CLI)**

   ```bash
   # HuggingFace 인증
   huggingface-cli login
   export HF_USER=$(huggingface-cli whoami | head -1)

   # 5개 에피소드 샘플 녹화
   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-test \
       --dataset.num_episodes=5 \
       --dataset.single_task="Pick up the red cube"
   # 제어키: → 에피소드 완료, ← 재녹화, ESC 종료
   ```

**데이터 스키마 (OMX 6차원):**

```
action: List[float32]              # 리더 상태 (5관절 + 그리퍼 = 6차원)
observation.state: List[float32]   # 팔로워 상태 (6차원)
observation.images.camera1: Image  # RGB 이미지
timestamp: float32
frame_index: int64
episode_index: int64
```

**결과물:**

- ✅ OMX 리더-팔로워 텔레오퍼레이션 동작 확인
- ✅ ROBOTIS 포크 LeRobot CLI 워크플로우 숙지
- ✅ Physical AI Tools 웹 UI 데이터 수집 경로 확인
- ✅ 샘플 시연 데이터 (5-10개)

**과제:**

- 3가지 간단한 작업에 대해 각 5개씩 시연 데이터 수집 (총 15개)
  - 작업 1: 객체 잡기 (Pick)
  - 작업 2: 객체를 목표 위치로 이동 (Place)
  - 작업 3: 객체 쌓기 (Stack)

**학습 자료:**

- https://ai.robotis.com/omx/setup_guide_lerobot.html
- https://ai.robotis.com/omx/operation_omx.html
- https://ai.robotis.com/omx/lerobot_imitation_learning_omx.html

---

### Week 3-4: GR00T N1.6 기초 및 데이터 수집

#### Week 3: GR00T N1.6 이해 및 OMX 연동 설정

**학습 목표:**

- GR00T N1.6 아키텍처 이해
- 사전 훈련된 모델 로드 및 추론
- OMX용 `modality_config.json` 작성 (5-DOF + gripper = 6차원 액션 공간)

**실습 내용:**

1. **GR00T N1.6 설치**

   ```bash
   # GitHub 클론
   git clone https://github.com/NVIDIA/Isaac-GR00T
   cd Isaac-GR00T

   # uv 기반 환경 설정 (conda가 아닌 uv 사용)
   pip install uv
   uv venv .venv --python 3.10
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```
2. **사전 훈련 모델 테스트**

   ```python
   from gr00t.policy.gr00t_policy import Gr00tPolicy
   from gr00t.data.embodiment_tags import EmbodimentTag
   import numpy as np

   # 실제 API: Gr00tPolicy 생성자 사용
   policy = Gr00tPolicy(
       model_path="nvidia/GR00T-N1.6-3B",
       embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
       device="cuda"
   )

   # Observation은 중첩 딕셔너리 형태
   observation = {
       "video": {
           "cam": np.random.rand(1, 1, 224, 224, 3).astype(np.float32)
       },
       "state": {
           "joints": np.random.rand(1, 1, 6).astype(np.float32)  # OMX: 5-DOF + 1 gripper = 6-dim
       },
       "annotation": {
           "task": [["pick up the red cube"]]
       }
   }

   # 추론: get_action() 메서드 사용
   action = policy.get_action(observation)
   ```
3. **OMX용 modality_config.json 작성 (핵심)**

   ```json
   // configs/omx_modality_config.json
   // OMX: 5-DOF + 1 gripper = 6차원 액션/상태 공간
   {
     "video": {
       "cam1": {"resolution": [224, 224], "num_frames": 1}
     },
     "state": {
       "joints": {"dim": 6}
     },
     "action": {
       "joints": {"dim": 6, "action_horizon": 16}
     },
     "annotation": {
       "task": {"type": "string"}
     }
   }
   ```
   - `EmbodimentTag.NEW_EMBODIMENT`로 OMX 등록
   - OMX의 6차원(5관절 + 그리퍼) → GR00T 액션 공간 매핑
   - 상대 행동(relative action chunk) 방식 이해
   - ROBOTIS 포크 LeRobot 데이터 → GR00T LeRobot v2 포맷 변환 확인

**결과물:**

- ✅ 사전 훈련 GR00T 모델 실행 (RTX 4090: ~44ms, H100: ~38ms)
- ✅ OMX용 modality_config.json 완성 (6차원 액션)
- ✅ 추론 성능 벤치마크

**이론 학습:**

- VLA 모델 아키텍처 논문 리뷰
- Diffusion Policy vs VLA 비교
- 상대 행동 vs 절대 행동 이해

---

#### Week 4: OMX로 체계적 데이터 수집

**학습 목표:**

- OMX Physical AI Tools 웹 UI와 LeRobot CLI 두 경로로 대량 데이터 수집
- 데이터 다양성 확보 전략
- 데이터 품질 관리 및 GR00T 포맷 변환

**실습 내용:**

1. **Physical AI Tools 웹 UI로 대량 수집 (경로 1)**

   - 브라우저에서 Record 페이지 접속
   - Task Info 설정:
     - **FPS**: 15 (OMX 권장)
     - **Episode Time**: 30-60초
     - **Reset Time**: 15-30초
     - **Num Episodes**: 50개/작업
   - Start 클릭 → 워밍업 → 녹화 → 리셋 자동 반복
   - 제어: Stop(저장 후 중단), Retry(재시작), Next(다음으로)

2. **LeRobot CLI로 대량 수집 (경로 2)**

   ```bash
   # 작업별 50개 에피소드 수집
   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-pick \
       --dataset.num_episodes=50 \
       --dataset.single_task="Pick up the object" \
       --dataset.fps=15

   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-place \
       --dataset.num_episodes=50 \
       --dataset.single_task="Place the object on the target"

   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-stack \
       --dataset.num_episodes=50 \
       --dataset.single_task="Stack the cubes"
   ```

3. **데이터 다양성 확보**

   - 객체 위치/종류 변경 (OMX 페이로드 100~250g 이내)
   - 조명 조건 변경
   - 카메라 각도 변경
   - 배경 변경

4. **데이터 품질 관리**

   - Physical AI Tools 웹 UI에서 에피소드별 시각화 검증
   - 불량 에피소드 제거 (Data Tools → 에피소드 삭제, 자동 재색인)
   - 데이터셋 병합 (여러 세션의 데이터를 하나로 통합)

5. **GR00T LeRobot v2 포맷 변환**

   ```bash
   # OMX HuggingFace 데이터 → GR00T-flavored LeRobot v2 변환
   # parquet (상태/액션) + mp4 (비디오) + meta/modality.json
   python scripts/convert_to_lerobot_v2.py \
       --input_dir data/omx_recordings \
       --output_dir data/omx_groot_v2 \
       --modality_config configs/omx_modality_config.json
   ```

**결과물:**

- ✅ 150+ 고품질 시연 데이터 (작업당 50개 × 3작업)
- ✅ HuggingFace Hub에 데이터셋 업로드
- ✅ GR00T LeRobot v2 포맷 변환 완료
- ✅ 데이터 품질 검증 보고서

**학습 자료:**

- https://ai.robotis.com/omx/dataset_preparation_recording_omx.html
- https://ai.robotis.com/omx/data_tools_omx.html
- https://ai.robotis.com/omx/dataset_preparation_visualization_omx.html

---

### Week 5-6: ACT 베이스라인 & GR00T 파인튜닝 및 OMX 배포

#### Week 5: ACT 베이스라인 학습 + GR00T 파인튜닝

**학습 목표:**

- ACT (Action Chunking Transformers) 베이스라인 학습으로 OMX 데이터 검증
- GR00T N1.6 후훈련(post-training) 수행
- 두 모델의 학습 과정 비교 (학습 곡선, 수렴 속도)

**실습 내용:**

1. **ACT 베이스라인 학습 (ROBOTIS fork LeRobot)**

   ACT는 OMX 공식 지원 정책으로, GR00T 비교의 베이스라인 역할을 합니다.

   ```bash
   # ACT 정책 학습 (Physical AI Tools Docker 내부 또는 LeRobot CLI)
   # 방법 1: Physical AI Tools Web UI
   # http://localhost:7860 → Training 탭 → ACT 선택 → 학습 시작

   # 방법 2: ROBOTIS fork LeRobot CLI
   cd ~/lerobot  # ROBOTIS-GIT/lerobot
   lerobot-train \
       --dataset.repo_id=${HF_USER}/omx_pick_cube \
       --policy.type=act \
       --policy.device=cuda \
       --training.num_epochs=100 \
       --training.batch_size=64
   ```

   ACT 학습 결과물:
   - `outputs/train/` 디렉토리에 `config.json` + `model.safetensors` 생성
   - 학습 로그 및 loss 추이 기록

2. **OMX용 modality_config 작성 (GR00T용)**

   ```json
   // configs/omx_modality_config.json
   // OMX: 5-DOF + 1 gripper = 6-dim action/state
   {
     "video": {"cam1": {"resolution": [224, 224], "num_frames": 1}},
     "state": {"joints": {"dim": 6}},
     "action": {"joints": {"dim": 6, "action_horizon": 16}},
     "annotation": {"task": {"type": "string"}}
   }
   ```

3. **OMX 데이터를 GR00T LeRobot v2 포맷으로 변환**

   ```bash
   # Week 4에서 수집한 OMX HuggingFace 데이터셋을
   # GR00T-flavored LeRobot v2 포맷으로 변환
   # parquet (상태/액션) + mp4 (비디오) + meta/modality.json 필수
   python scripts/convert_to_lerobot_v2.py \
       --input_dir data/omx_hf_dataset \
       --output_dir data/omx_lerobot_v2 \
       --modality_config configs/omx_modality_config.json
   ```

   > **주의**: OMX LeRobot 데이터는 HuggingFace datasets 포맷이며,
   > GR00T는 LeRobot v2 (parquet + mp4 + modality.json) 포맷을 요구합니다.
   > 6-dim action space가 modality_config와 일치하는지 반드시 확인하세요.

4. **GR00T N1.6 파인튜닝 실행**

   ```bash
   # 실제 launch_finetune.py 사용 (OMX 6-dim 설정)
   CUDA_VISIBLE_DEVICES=0 uv run python \
       gr00t/experiment/launch_finetune.py \
       --base-model-path nvidia/GR00T-N1.6-3B \
       --dataset-path data/omx_lerobot_v2 \
       --embodiment-tag NEW_EMBODIMENT \
       --modality-config-path configs/omx_modality_config.json \
       --num-gpus 1 \
       --max-steps 10000 \
       --batch-size 32
   ```

5. **학습 모니터링 및 비교**

   - TensorBoard 로그 분석 (ACT vs GR00T 동시 비교)
   - 손실 함수 수렴 속도 비교
   - ACT: epoch 기반 학습 곡선 / GR00T: step 기반 학습 곡선
   - 하이퍼파라미터 최적화 (learning rate, batch size, augmentation)

**결과물:**

- ✅ ACT 베이스라인 체크포인트 (`config.json` + `model.safetensors`)
- ✅ 파인튜닝된 GR00T 체크포인트
- ✅ ACT vs GR00T 학습 과정 비교 리포트
- ✅ 최적 하이퍼파라미터 문서

**예상 GPU 시간:**

- ACT 학습: 1x RTX 3090/4090 충분, 100 epoch 기준 약 1-3시간
- GR00T 파인튜닝: 1x H100 또는 L40 권장, 10K 스텝 기준 약 4-8시간
- 다중 GPU 사용 시 GR00T `--num-gpus` 조정

---

#### Week 6: OMX 실제 로봇 배포 및 ACT vs GR00T 비교 평가

**학습 목표:**

- ACT 및 GR00T 모델을 OMX 실제 로봇에 배포
- 동일 작업에서 두 정책의 성능 비교
- 정량적/정성적 평가 및 실패 분석

**실습 내용:**

1. **ACT 모델 OMX 배포 (Physical AI Tools / LeRobot)**

   ```bash
   # 방법 1: Physical AI Tools Web UI로 배포
   # http://localhost:7860 → Inference 탭 → 학습된 ACT 모델 선택 → 실행

   # 방법 2: LeRobot CLI로 배포
   cd ~/lerobot
   python -m lerobot.scripts.control_robot \
       --robot.type=omx_follower \
       --robot.port=/dev/ttyACM0 \
       --control.type=record \
       --control.policy.path=outputs/train/act_omx_pick \
       --control.single_task="Pick up the red cube"
   ```

2. **GR00T 모델 OMX 배포**

   ```python
   import numpy as np
   from gr00t.policy.gr00t_policy import Gr00tPolicy
   from gr00t.data.embodiment_tags import EmbodimentTag

   # 파인튜닝된 GR00T 체크포인트 로드
   policy = Gr00tPolicy(
       model_path="checkpoints/groot_omx/best_model",
       embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
       device="cuda"
   )

   # OMX 실시간 추론 루프 (ROS 2 연동)
   # ros2_control 100Hz 제어 주기와 동기화 필요
   while True:
       obs = get_omx_observation()  # 카메라 224x224 + 6-dim joints
       action = policy.get_action(obs)  # 6-dim action 출력
       omx_robot.execute_action(action)  # OMX follower에 명령 전송
   ```

   > **참고**: GR00T 추론은 GPU 필요 (CUDA), ACT는 CPU에서도 동작 가능.
   > OMX의 `ros2_control`은 100Hz로 동작하므로, 추론 지연시간(latency) 측정 필수.

3. **동일 조건 비교 평가 프로토콜**

   | 평가 항목 | ACT | GR00T | 비고 |
   |----------|-----|-------|------|
   | 작업 성공률 | - | - | 각 20회 시행 |
   | 평균 실행 시간 | - | - | 초 단위 |
   | 추론 지연시간 | - | - | ms 단위 |
   | Smoothness | - | - | 궤적 jerk 측정 |
   | 일반화 능력 | - | - | 물체 위치 변경 시 |

   ```bash
   # 각 모델별 동일 작업 20회 반복 실험
   # 작업 예시: pick_cube, place_on_plate, stack_blocks
   for task in pick_cube place_on_plate stack_blocks; do
       echo "=== Evaluating ACT on ${task} ==="
       python eval_omx.py --policy=act --task=${task} --num_trials=20
       echo "=== Evaluating GR00T on ${task} ==="
       python eval_omx.py --policy=groot --task=${task} --num_trials=20
   done
   ```

4. **실패 케이스 분석 및 개선**

   - 실패 에피소드 영상 녹화 (LeRobot `--control.type=record`)
   - ACT vs GR00T 실패 패턴 비교 (grasping 실패, 위치 오차, 충돌 등)
   - 필요시 추가 데이터 수집 → 재학습 반복

**결과물:**

- ✅ ACT 베이스라인 OMX 배포 완료
- ✅ GR00T OMX 배포 완료
- ✅ ACT vs GR00T 성능 비교 보고서 (표 + 그래프)
- ✅ 실패 케이스 분석 및 개선 방안 문서

**평가 메트릭:**

- 작업 성공률 (목표: ACT ≥70%, GR00T ≥80%)
- 추론 지연시간 (목표: <100ms per step)
- 궤적 부드러움 (Smoothness/Jerk 지표)
- 일반화 점수 (물체 위치 변경 시 성공률 변화)

---

### Week 7-8: Cosmos Policy 평가 + Cosmos Predict2.5 Action-Conditioned 구축 (병렬)

> **병렬 처리**: Week 7-8은 두 트랙을 병렬로 진행합니다.
> - **트랙 A**: Cosmos Policy LIBERO 벤치마크 평가 (기존)
> - **트랙 B**: Cosmos Predict2.5 Action-Conditioned 환경 구축 + OMX 데이터 변환 (신규)

#### Week 7: Cosmos Policy 환경 구축 및 추론

**학습 목표:**

- Cosmos Policy 아키텍처 이해 (Latent Action Encoding, 통합 생성 프로세스)
- Docker 기반 환경 구축
- 사전 훈련 모델로 LIBERO 벤치마크 추론

**실습 내용:**

1. **Cosmos Policy 설치 (Docker 필수)**

   ```bash
   git clone https://github.com/NVlabs/cosmos-policy
   cd cosmos-policy

   # Docker 환경 빌드 및 실행 (pip 설치 미지원)
   docker build -t cosmos-policy -f docker/Dockerfile .
   docker run --gpus all -it --rm \
       -v $(pwd):/workspace \
       cosmos-policy bash
   ```
2. **LIBERO 벤치마크 추론 (사전 훈련 모델)**

   ```python
   from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
   from cosmos_policy.experiments.robot.cosmos_utils import (
       get_action, get_model, load_dataset_stats
   )

   # Config 기반 모델 로딩 (from_pretrained 패턴 아님)
   cfg = PolicyEvalConfig(
       config="cosmos_predict2_2b_480p_libero__inference_only",
       ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
       libero_task_suite="libero_spatial",
       num_episodes_per_task=20,
   )

   model, cosmos_config = get_model(cfg)
   dataset_stats = load_dataset_stats(cfg)

   # 추론 실행 (6-10 GB VRAM, 단일 GPU 충분)
   action_return_dict = get_action(
       cfg, model, dataset_stats,
       observation, task_description,
       cosmos_config=cosmos_config
   )
   # 반환: 행동 + 미래 상태 예측 + 가치 추정
   ```
3. **LIBERO 4개 서브스위트 전체 평가**

   ```bash
   # 실제 평가 스크립트 (Docker 내부)
   bash eval.sh cosmos_predict2_2b_480p_libero__inference_only \
       nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
       libero_spatial 20

   # 4개 서브스위트: spatial, object, goal, long
   ```
4. **미래 예측 시각화**

   - Cosmos Policy의 latent diffusion 기반 미래 상태 예측 시각화
   - 행동 + 미래 프레임 + 가치 함수를 동시 분석
   - 다양한 작업에서의 예측 품질 비교

**결과물:**

- ✅ Docker 기반 Cosmos Policy 환경 구축
- ✅ LIBERO 벤치마크 추론 결과 (목표: 논문 재현 ~98.5%)
- ✅ 미래 예측 시각화 자료

**이론 학습:**

- Cosmos-Predict2 아키텍처 논문 리뷰
- Latent Action Encoding 원리 이해
- Video-to-Policy 전환 메커니즘

---

#### Week 8: Cosmos Policy 심화 분석 및 GR00T 비교

**학습 목표:**

- Cosmos Policy의 데이터 효율성 분석
- GR00T N1.6 파인튜닝 결과와 정량적 비교
- Test-time Planning 메커니즘 이해

**실습 내용:**

1. **데이터 포맷 차이 분석**

   ```python
   # Cosmos Policy 데이터 포맷 (GR00T LeRobot v2와 다름)
   # Cosmos: pickle observation dict
   cosmos_obs = {
       "primary_image": np.array(...),   # [H, W, 3]
       "wrist_image": np.array(...),     # [H, W, 3]
       "proprio": np.array(...),         # [D]
   }
   # + T5 text embeddings 필요

   # GR00T: LeRobot v2 포맷
   groot_obs = {
       "video": {"cam": np.array(...)},  # [B, T, H, W, C]
       "state": {"joints": np.array(...)},
       "annotation": {"task": [["..."]]}
   }
   ```
2. **GR00T vs Cosmos Policy 정량적 비교**

   | 메트릭 | GR00T N1.6 (파인튜닝) | Cosmos Policy (추론) |
   |--------|----------------------|---------------------|
   | 성공률 (단순 작업) | Week 6 결과 | LIBERO 결과 |
   | 추론 속도 | ~44ms (RTX 4090) | 측정 |
   | 데이터 요구량 | 50-100 demos | 50-200 demos |
   | 미래 예측 | 불가 | 가능 |
   | 언어 이해 | 자연어 명령 | 제한적 |

3. **Cosmos Policy 강점/약점 실증**

   - 강점: 데이터 효율성, SOTA 벤치마크 성능, 미래 예측
   - 약점: Docker 전용, 커스텀 작업 파인튜닝 시 8x H100 필요
   - GR00T 대비: 언어 이해 제한, 크로스 플랫폼 호환성 부재

4. **비교 분석 보고서 작성**

   - 두 모델의 아키텍처적 차이 (VLA vs Video-to-Policy)
   - 각 모델이 우수한 작업 유형 분석
   - 실용적 배포 관점에서의 트레이드오프

**결과물:**

- ✅ GR00T vs Cosmos Policy 비교 보고서
- ✅ 데이터 포맷 변환 가이드
- ✅ 각 모델 적합 시나리오 분석

---

#### Week 7-8 트랙 B: Cosmos Predict2.5 Action-Conditioned 환경 구축 (병렬)

> **트랙 A(Cosmos Policy 평가)와 동시 진행**. Cosmos Predict2.5를 OMX 세계 모델로 활용하기 위한 기반을 구축합니다.

**학습 목표:**

- Cosmos Predict2.5 Action-Conditioned 아키텍처 이해
- OMX joint space → EE space 변환 (Forward Kinematics)
- OMX 데이터를 Cosmos 입력 포맷으로 변환

**실습 내용:**

1. **Cosmos Predict2.5 설치 및 환경 구축**

   ```bash
   cd ~/cosmos-predict2.5
   # Docker 환경 또는 uv 기반 설치
   uv sync

   # 사전 훈련 모델 다운로드
   # nvidia/Cosmos-Predict2.5-2B/robot/action-cond
   ```

2. **Action-Conditioned 추론 테스트 (Bridge 데이터)**

   ```bash
   # 기본 예제로 동작 확인 (1x GPU)
   python examples/action_conditioned.py \
       -i assets/action_conditioned/basic/inference_params.json \
       -o outputs/action_conditioned/basic
   ```

3. **OMX FK 변환 유틸리티 구현** (`utils/omx_fk.py`)

   ```python
   from utils.omx_fk import OMXForwardKinematics

   fk = OMXForwardKinematics(robot="omx_f")  # Follower 로봇

   # OMX-F URDF 기반 kinematic chain (관절축: Z, Y, Y, Y, X):
   # joint1: Z축, offset (-0.01125, 0, 0.034)   ← 베이스 수평 회전
   # joint2: Y축, offset (0, 0, 0.0635)          ← 어깨 수직 회전
   # joint3: Y축, offset (0.0415, 0, 0.11315)    ← 팔꿈치 굴곡
   # joint4: Y축, offset (0.162, 0, 0)            ← 손목 굴곡
   # joint5: X축, offset (0.0287, 0, 0)           ← 손목 회전
   # EE:     fixed, offset (0.09193, -0.0016, 0)  ← 엔드이펙터

   # 관절 각도 → EE 포즈 (Cosmos 입력 형식)
   cosmos_state = fk.to_cosmos_state(
       joint_positions=[0.0, -0.5, 0.3, 0.2, 0.0],  # 5-DOF (radians)
       gripper=0.5                                     # [0.0 ~ 1.0]
   )
   # → {"state": [x, y, z, roll, pitch, yaw], "continuous_gripper_state": 0.5}

   # 궤적 전체 변환 (T timesteps)
   batch = fk.batch_to_cosmos_states(joint_trajectory, gripper_trajectory)
   # → states: (T, 6), actions: (T-1, 7), continuous_gripper_states: (T,)
   ```

4. **OMX 데이터 → Cosmos 입력 포맷 변환**

   ```python
   from utils.omx_fk import OMXForwardKinematics
   import json

   fk = OMXForwardKinematics(robot="omx_f")

   # Week 4에서 수집한 OMX 데이터 (joint space)를 Cosmos 형식으로 변환
   # joint_trajectory: (T, 5), gripper_trajectory: (T,)
   batch = fk.batch_to_cosmos_states(joint_trajectory, gripper_trajectory)

   # annotations/*.json 구조 생성:
   annotation = {
       "state": batch["states"].tolist(),                    # (T, 6) EE 포즈
       "continuous_gripper_state": batch["continuous_gripper_states"].tolist(),
       "videos": ["videos/episode_000.mp4"],
   }
   with open("datasets/omx/annotations/episode_000.json", "w") as f:
       json.dump(annotation, f)
   ```

**결과물:**

- ✅ Cosmos Predict2.5 Action-Conditioned 추론 환경
- ✅ OMX FK 변환 유틸리티 (`utils/omx_fk.py`) - URDF 기반 joint→EE 변환 + Cosmos state/action 생성
- ✅ OMX URDF kinematic 파라미터 (`utils/omx_constants.py`) - OMX-F/OMX-L 관절 오프셋, 축, EE 오프셋
- ✅ OMX → Cosmos 데이터 변환 스크립트

---

### Week 9-10: Cosmos 데이터 증강 + IDM 시너지 파이프라인 (병렬)

> **실증 반영 (v4)**: Cosmos Predict2.5 실험 결과, post-training 없이는 새 로봇에 사용 불가 확인.
> 실제 OMX 로봇 데이터 유무에 따라 방안 선택. 상세: [PIPELINE.md](PIPELINE.md) "Cosmos 합성 데이터 파이프라인 (v4)" 참조.
>
> - **방안 A** (OMX 데이터 있을 때): Cosmos Predict2.5 Base 후훈련 → 합성 비디오 → IDM pseudo label
> - **방안 B** (OMX 데이터 없을 때): Cosmos Transfer 2.5로 기존 시연 영상 시각 증강 (action 유지)
> - **트랙 B**: GR00T IDM pseudo labeling + VLA 데이터 증강 파이프라인 구축

#### Week 9 트랙 A: Cosmos Predict2.5 OMX 후훈련 + 합성 비디오 생성

**학습 목표:**

- Cosmos Predict2.5를 OMX 데이터로 후훈련하여 OMX 전용 세계 모델 확보
- 후훈련된 모델로 합성 비디오 생성

**실습 내용:**

1. **OMX 데이터로 Action-Conditioned 후훈련**

   ```bash
   cd ~/cosmos-predict2.5

   # OMX 데이터셋 구조 확인
   # datasets/omx/
   # ├── annotations/*.json  (FK 변환된 EE 포즈 + 그리퍼)
   # └── videos/*.mp4        (카메라 영상)

   # 후훈련 실행 (1x GPU, RTX 4090 가능)
   torchrun --nproc_per_node=1 --master_port=12341 \
       -m scripts.train \
       --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
       -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320 \
       ~dataloader_train.dataloaders
   ```

2. **체크포인트 변환**

   ```bash
   # DCP → PyTorch 형식 변환
   python scripts/convert_distcp_to_pt.py \
       $CHECKPOINT_DIR/model $CHECKPOINT_DIR
   # → model_ema_bf16.pt 생성 (추론 권장)
   ```

3. **OMX 합성 비디오 생성**

   ```bash
   # 후훈련 체크포인트로 합성 비디오 생성
   python examples/action_conditioned.py \
       -i datasets/omx/annotations/inference_params.json \
       -o outputs/omx_synthetic_videos \
       --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
       --experiment ac_reason_embeddings_rectified_flow_2b_256_320
   ```

4. **합성 비디오 품질 검증**

   - 생성된 비디오와 실제 OMX 비디오의 시각적 비교
   - 물리적 일관성 (물체 상호작용, 그리퍼 동작)
   - 장기 안정성 (연속 프레임의 시간적 일관성)

**결과물:**

- ✅ OMX 후훈련된 Cosmos Predict2.5 체크포인트
- ✅ 합성 비디오 세트 (다양한 행동 시퀀스)
- ✅ 합성 비디오 품질 평가 보고서

---

#### Week 9 트랙 B: GR00T IDM pseudo labeling 파이프라인 구축 (병렬)

**학습 목표:**

- GR00T IDM 추론 환경 구축
- 합성 비디오에서 action pseudo label 추출
- 품질 평가 메트릭 구현

**실습 내용:**

1. **GR00T IDM 추론 환경**

   ```python
   # IDM 모델 로드
   from gr00t.model.idm import IDM
   model = IDM.from_pretrained("/path/to/local/idm_checkpoint")
   ```

2. **합성 비디오 → IDM 입력 변환**

   ```python
   from utils.omx_constants import VIDEO_DTYPE_IDM, OMX_IDM_JOINT_NAMES

   # Cosmos 출력 (uint8, 가변 해상도) → IDM 입력 (uint8, 256x256, 2-frame pairs)
   # 리사이즈 + 프레임 페어링
   ```

3. **Pseudo label 품질 평가 구현**

   - Jerk (3차 미분): 낮을수록 부드러운 궤적
   - Temporal consistency: 연속 예측 간 일관성
   - Grade B 이상만 VLA 학습에 사용

**결과물:**

- ✅ IDM 추론 파이프라인
- ✅ 품질 평가 모듈
- ✅ 품질 필터링된 pseudo label 데이터셋

---

#### Week 10: 통합 파이프라인 실행 + 모델 크로스 분석

> Week 9의 두 트랙 결과를 결합하여 전체 통합 파이프라인을 실행합니다.

**실습 내용:**

1. **전체 통합 파이프라인 실행**

   ```
   [Week 4 데이터] OMX 텔레오퍼레이션 → 실제 데이터 (50회)
                          ↓
   [Week 7-8 트랙B] FK 변환 → Cosmos 입력 포맷
                          ↓
   [Week 9 트랙A] Cosmos Predict2.5 추론 → 합성 비디오 (수백 회)
                          ↓
   [Week 9 트랙B] IDM pseudo labeling → action label
                          ↓ 관절 매핑 (IDM→VLA) + dtype 변환
                          ↓ 품질 필터 (grade ≥ B)
                          ↓
   [Week 10] 실제 데이터 + 합성 데이터 → GR00T VLA 재파인튜닝
   ```

2. **VLA 데이터 증강 재학습**

   ```bash
   # 실제 데이터 + 고품질 pseudo label 데이터 결합
   # GR00T 재파인튜닝
   bash scripts/week5_finetune_groot_omx.sh
   # DATASET_DIR에 증강 데이터 포함
   ```

3. **증강 전 vs 증강 후 비교**

   | 메트릭 | 증강 전 (50 demos) | 증강 후 (50 + 합성) |
   |--------|-------------------|---------------------|
   | 단순 작업 성공률 | Week 6 결과 | 재평가 |
   | 복잡 작업 성공률 | Week 6 결과 | 재평가 |
   | 일반화 능력 | - | 새 환경/물체 테스트 |

**결과물:**

- ✅ 증강 데이터로 재학습된 GR00T VLA 체크포인트
- ✅ 증강 전/후 성능 비교 보고서
- ✅ 5개 모델 크로스 분석 보고서

---

#### Week 9-10 (기존 참고): DreamDojo 추론 및 세계 모델 체험

> **참고**: DreamDojo는 Cosmos Predict2.5로 대체되었으나, GPU 리소스 확보 시 추가 비교를 위해 기존 내용을 유지합니다.

#### Week 9 (참고): DreamDojo 환경 구축 및 롤아웃 생성

**학습 목표:**

- DreamDojo 세계 파운데이션 모델 아키텍처 이해
- 사전 훈련 모델 로드 및 롤아웃 생성
- 물리 시뮬레이션 품질 평가

**실습 내용:**

1. **DreamDojo 설치**

   ```bash
   git clone https://github.com/dreamdojo-world/dreamdojo
   cd dreamdojo

   # 주의: 패키지명이 cosmos-predict2 (DreamDojo가 아님)
   pip install -e .

   # launch.sh의 내부 경로 하드코딩 수정 필요
   # /mnt/amlfs-01/shared/... → 로컬 경로로 변경
   ```
2. **사전 훈련 모델 로드 및 추론**

   ```python
   # DreamDojo는 Python 클래스가 아닌 config + torchrun 기반
   # 추론은 scripts 디렉토리의 스크립트 활용

   # 사전 훈련 모델 다운로드
   # nvidia/DreamDojo-2B-480p-GR1 등 HuggingFace 체크포인트

   # 추론 실행 (config 기반)
   # configs/2b_480_640_gr1.yaml 참조
   ```
3. **롤아웃 생성 및 시각화**

   - 초기 상태 + 행동 시퀀스 입력 → 미래 픽셀 시퀀스 생성
   - 다양한 로봇 플랫폼(GR-1, Unitree G1, YAM)에서의 롤아웃 비교
   - 물리적 일관성 평가 (중력, 충돌, 마찰)

4. **세계 모델 품질 분석**

   - 생성된 비디오의 시각적 품질 (FID, SSIM 등)
   - 장기 롤아웃 안정성 (1분+ 연속 생성)
   - 물리 법칙 준수 여부 정성적 평가

**결과물:**

- ✅ DreamDojo 추론 환경 구축
- ✅ 다양한 시나리오의 롤아웃 비디오
- ✅ 세계 모델 품질 평가 보고서

**이론 학습:**

- World Foundation Model 논문 리뷰
- Latent Action Pre-training 메커니즘
- Autoregressive Generation vs Diffusion 비교

**제약사항:**

- 증류 파이프라인 미공개 → 실시간 10 FPS 생성 불가
- 텔레오퍼레이션 코드 미공개 → VR 연동 불가
- 후훈련(Post-training)은 8 GPU 노드 필요 → 추론만 수행

---

#### Week 10: 통합 파이프라인 + Cosmos Predict2.5-IDM 시너지 크로스 분석

**학습 목표:**

- Cosmos Predict2.5 합성 비디오 + IDM pseudo labeling 시너지 파이프라인 구축
- 5개 모델의 아키텍처적 차이와 상호 보완성 분석
- 증강 전/후 성능 비교

**실습 내용:**

1. **Cosmos Predict2.5 + IDM 시너지 파이프라인**

   - OMX joint → FK 변환 → Cosmos EE state/action
   - Cosmos Predict2.5 합성 비디오 생성
   - GR00T IDM pseudo labeling → 관절 매핑 (IDM→VLA) → 품질 필터 (grade ≥ B)
   - GR00T N1.6 VLA 재학습

2. **5개 모델 종합 비교 분석**

   ```
   ┌─────────────────┬──────────┬──────────────┬───────────────────┬───────────┬──────────────┐
   │ 비교 항목       │ ACT      │ GR00T N1.6   │ Cosmos Predict2.5 │ GR00T IDM │ Cosmos Policy│
   ├─────────────────┼──────────┼──────────────┼───────────────────┼───────────┼──────────────┤
   │ 모델 타입       │ IL Policy│ VLA          │ World Model       │ IDM       │ Video-Policy │
   │ 본 프로젝트     │ 학습+배포│ 파인튜닝+배포│ 후훈련+추론       │ 추론      │ 추론 전용    │
   │ GPU 요구        │ CPU 가능 │ 1x RTX 4090  │ 1x RTX 4090       │ 1x RTX4090│ 1x (추론)    │
   │ 추론 속도       │ <10ms    │ ~44ms        │ 비실시간          │ ~20ms     │ 측정필요     │
   │ 언어 이해       │ ❌        │ ✅            │ ❌                 │ ❌         │ 제한적       │
   │ IDM 시너지      │ -        │ 증강 수신    │ 합성 비디오 생성  │ pseudo lbl│ 참조         │
   └─────────────────┴──────────┴──────────────┴───────────────────┴───────────┴──────────────┘
   ```

3. **통합 파이프라인 설계**

   ```
   [Option C+ 실제 구현 파이프라인]

   OMX 시연 → GR00T N1.6 파인튜닝 (실시간 제어)
                      ↓ joint positions
   FK 변환 → Cosmos EE state/action (utils/omx_fk.py)
                      ↓
   합성 비디오 → Cosmos Predict2.5 (미래 비디오 생성)
                      ↓
   pseudo label → GR00T IDM (행동 역추정)
                      ↓ 관절 매핑 + 품질 필터
   재학습 → GR00T N1.6 (증강 데이터로 성능 향상)
   ```

4. **프로젝트 인사이트 정리**

   - VLA vs World Model vs IDM 패러다임 비교
   - FK 변환 (joint-space → EE-space)의 중요성
   - 향후 통합 연구 방향 제안

**결과물:**

- ✅ 5개 모델 크로스 분석 보고서
- ✅ Cosmos Predict2.5-IDM 시너지 파이프라인
- ✅ 데이터 포맷 변환 명세서 (FK, 관절 매핑, dtype 변환)

---

### Week 11-12: 비교 분석 보고서 및 프로젝트 마무리

#### Week 11: 종합 벤치마킹 및 분석 보고서

**학습 목표:**

- 12주간의 실험 결과 종합 분석
- 각 모델의 실용적 배포 시나리오 도출
- 재현 가능한 실험 환경 문서화

**실습 내용:**

1. **종합 벤치마킹 정리 (OMX 실측 기준)**

   ```python
   # ACT 베이스라인 (OMX 공식 정책) - 실측 데이터
   act_results = {
       "simple_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "complex_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "training_data": "50-100 demos (OMX 수집)",
       "training_gpu": "1x RTX 4090, ~1-3h",
       "inference_speed": "CPU 가능, GPU 시 <10ms",
       "note": "OMX 기본 지원, Physical AI Tools 통합",
   }

   # GR00T N1.6 (파인튜닝) - OMX 실측 데이터
   groot_results = {
       "simple_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "complex_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "training_data": "150+ demos (OMX 수집, LeRobot v2 변환)",
       "training_gpu": "1x H100, ~8h",
       "inference_speed": "~44ms (RTX 4090)",
       "action_dim": 6,  # OMX 5-DOF + 1 gripper
       "note": "ACT 대비 성능 차이 분석",
   }

   # Cosmos Policy (추론 전용) - LIBERO 벤치마크
   cosmos_results = {
       "libero_spatial": ...,
       "libero_object": ...,
       "libero_goal": ...,
       "libero_long": ...,
       "inference_gpu": "1x (6-10GB VRAM)",
       "note": "파인튜닝 미수행 (8x H100 필요), OMX 직접 배포 불가",
   }

   # Cosmos Predict2.5 + IDM 시너지 - 합성 데이터 품질
   synergy_results = {
       "num_synthetic_episodes": ...,
       "idm_pseudo_label_quality": ...,  # jerk, temporal consistency
       "quality_grade": ...,  # A/B/C/D
       "augmentation_gpu": "1x RTX 4090",
       "note": "FK 변환 + 합성 비디오 + pseudo labeling",
   }
   ```

2. **실용적 배포 시나리오 매트릭스 (OMX 기준)**

   | 시나리오 | 권장 모델 | 이유 |
   |---------|----------|------|
   | OMX 빠른 프로토타이핑 | ACT | OMX 공식 지원, 빠른 학습, CPU 추론 가능 |
   | OMX 고성능 조작 | GR00T N1.6 | 언어 이해, 크로스 플랫폼, 대규모 사전학습 |
   | 범용 로봇 제어 (언어 명령) | GR00T N1.6 | 유일하게 NL 명령 직접 지원 |
   | 정밀 조작 (소량 데이터) | Cosmos Policy | 50개 시연으로 SOTA, 데이터 효율적 |
   | 데이터 증강 (합성) | Cosmos Predict2.5 + IDM | 합성 비디오 → pseudo label → VLA 재학습 |
   | 저비용/교육용 배포 | ACT | GPU 불필요, 학습 곡선 낮음 |

3. **학습된 교훈 (Lessons Learned)**

   - 논문 의사코드 vs 실제 API의 괴리
   - GPU 요구사항의 현실적 제약
   - 데이터 포맷 표준화의 중요성
   - Docker 기반 ML 환경의 장단점

4. **코드 정리 및 재현 가능성 확보**

   - 전체 실험 스크립트 정리
   - Docker/환경 설정 문서화
   - 데이터 변환 유틸리티 코드

**결과물:**

- ✅ 종합 비교 분석 보고서 (30+ 페이지)
- ✅ 재현 가능한 실험 코드 (GitHub)
- ✅ 실용적 배포 가이드

---

#### Week 12: 최종 발표 및 프로젝트 마무리

**학습 목표:**

- 최종 발표 자료 준비
- 향후 연구 방향 제안
- 프로젝트 회고

**실습 내용:**

1. **OMX 라이브 데모 준비 (ACT vs GR00T)**

   ```python
   from gr00t.policy.gr00t_policy import Gr00tPolicy
   from gr00t.data.embodiment_tags import EmbodimentTag

   # GR00T 최적 체크포인트 로드
   groot_policy = Gr00tPolicy(
       model_path="checkpoints/groot_omx/best_model",
       embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
       device="cuda"
   )

   # ACT 모델은 Physical AI Tools 또는 LeRobot CLI로 로드
   # lerobot-record --control.policy.path=outputs/train/act_omx_best

   # 라이브 데모: OMX에서 동일 작업으로 ACT vs GR00T 비교
   demo_tasks = [
       "pick up the red cube",        # 단순 Pick
       "place the cube on the plate",  # Pick & Place
       "stack the cubes"               # 다단계 조작
   ]
   ```

2. **데모 비디오 제작**

   - ACT vs GR00T: OMX 동일 작업 나란히 비교 영상
   - GR00T 파인튜닝 전/후 성능 비교 영상
   - Cosmos Policy LIBERO 추론 시연
   - Cosmos Predict2.5 합성 비디오 시연
   - 5개 모델(ACT, GR00T, Cosmos Predict2.5, IDM, Cosmos Policy)의 출력 비교 영상

3. **발표 자료 준비**

   - 프로젝트 개요: OMX 하드웨어 + Option C+ 하이브리드 접근법
   - OMX 데이터 수집 파이프라인 (Physical AI Tools + LeRobot)
   - ACT vs GR00T 비교: 학습 효율, 성공률, 추론 속도, 일반화 능력
   - 5개 모델의 기술적 차이 (아키텍처 다이어그램)
   - Cosmos Policy / Cosmos Predict2.5 추론 분석
   - 실용적 배포 시나리오 및 향후 방향

4. **향후 연구 방향 제안**

   - OMX 멀티태스크 학습 확장 (10+ 작업)
   - ACT의 다른 정책 (Diffusion Policy, VQ-BeT) 대체 실험
   - 8x H100 접근 시 Cosmos Policy OMX 데이터 파인튜닝
   - Cosmos Predict2.5 증류 (DMD2) 파이프라인으로 실시간 추론
   - GR00T + ACT 앙상블 정책 연구
   - OMX 듀얼암(Bimanual) 확장

**최종 결과물:**

- ✅ 최종 발표 슬라이드 (30분)
- ✅ 데모 비디오 (OMX ACT vs GR00T 라이브 + 비교 분석)
- ✅ 완전한 코드베이스 (GitHub, 재현 가능)
- ✅ 기술 보고서 패키지
- ✅ 프로젝트 회고 문서

**발표 구성:**

1. **도입** (5분): OMX 하드웨어 소개 + Option C+ 접근법 선택 이유
2. **데이터 파이프라인** (5분): OMX 텔레오퍼레이션 → 데이터 수집 → 포맷 변환
3. **ACT vs GR00T** (10분): 학습/배포/성능 비교 실험 결과 (OMX 실측)
4. **추론 비교** (5분): Cosmos Policy LIBERO + Cosmos Predict2.5 합성 비디오 + IDM 시너지
5. **라이브 데모** (5분): OMX에서 ACT와 GR00T 동일 작업 실시간 시연
6. **결론** (5분): 배운 점, 향후 방향, 통합 파이프라인 비전

---

## 9. 프로젝트 성공을 위한 팁

### 리소스 관리

**GPU 리소스 (Option C+ 하이브리드 + OMX 기준):**

- ACT 베이스라인 학습: 1x RTX 3090/4090 충분 (~1-3시간)
- ACT 추론: CPU 가능 (GPU 불필요), GPU 시 <10ms
- GR00T N1.6 파인튜닝: 1x H100 또는 L40 권장 (A6000도 가능하나 느림)
- GR00T N1.6 추론: RTX 4090 (44ms, 22.8 Hz) 또는 H100 (38ms, 26.3 Hz)
- Cosmos Policy 추론: 1x GPU, 6-10 GB VRAM (RTX 3090 이상)
- Cosmos Predict2.5 후훈련/추론: 1x RTX 4090 (본 프로젝트 범위 내)
- GR00T IDM 추론: 1x RTX 4090
- ⚠️ Cosmos Policy 파인튜닝: 8x H100 80GB (본 프로젝트 범위 외)
- 클라우드 GPU 옵션: Lambda Labs, RunPod, AWS p4d/p5

**시간 관리:**

- 훈련 시간을 고려한 스케줄링
- 주말에 장시간 훈련 작업 배치
- 병렬 실험으로 시간 단축

### 일반적인 함정 및 해결책

**문제 1: Sim-to-Real 갭**

- 해결: 도메인 랜덤화, 실제 데이터 혼합, 점진적 적응

**문제 2: 데이터 품질 불량**

- 해결: 체계적 데이터 검증, 이상치 제거, 재수집

**문제 3: 과적합**

- 해결: 데이터 증강, 정규화, 조기 종료

**문제 4: 훈련 불안정**

- 해결: Learning rate 조정, 그래디언트 클리핑, 배치 크기 증가

### 확장 아이디어 (Option C+ 완료 후)

**GPU 리소스 확보 시:**

1. **Cosmos Policy 파인튜닝**: 8x H100 클라우드 인스턴스로 LeRobot 데이터 직접 파인튜닝
2. **Cosmos Predict2.5 증류**: DMD2 가이드로 실시간 추론 최적화
3. **모델 I/O 통합 어댑터**: GR00T LeRobot v2 ↔ Cosmos EE-space ↔ IDM 자동 변환

**기본 프로젝트 완료 후:**

4. **멀티모달 입력 추가**: 촉각, 힘 센서 통합
5. **언어 명령 확장**: 복잡한 자연어 지시 처리
6. **다중 로봇 협업**: 여러 LeRobot 협력 작업
7. **온라인 학습**: 실시간 피드백 기반 개선
8. **실제 응용**: 특정 산업 작업 적용 (분류, 포장 등)

### 학습 리소스

**논문:**

- GR00T N1: "An Open Foundation Model for Generalist Humanoid Robots"
- Cosmos Policy: "Fine-Tuning Video Models for Visuomotor Control"
- DreamDojo: "A Generalist Robot World Model from Large-Scale Human Videos"

**코드 저장소:**

- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/NVlabs/cosmos-policy
- https://github.com/dreamdojo-world/dreamdojo
- https://github.com/huggingface/lerobot

**커뮤니티:**

- NVIDIA Developer Forums
- Isaac Sim Discord
- LeRobot Community
- Robotics StackExchange

---

## 10. 예상 프로젝트 결과

### 정량적 목표 (Option C+ 하이브리드)

| 메트릭 | 목표 | 우수 | 비고 |
| ------ | ---- | ---- | ---- |
| GR00T 단순 작업 성공률 | >80% | >90% | 파인튜닝 후 실측 |
| GR00T 복잡 작업 성공률 | >60% | >75% | 파인튜닝 후 실측 |
| Cosmos LIBERO 재현율 | >95% | >98% | 논문 결과 재현 |
| Cosmos Predict2.5 합성 품질 | grade B+ | grade A | IDM pseudo label 품질 |
| GR00T 추론 속도 | <100ms | <50ms | RTX 4090 기준 |
| 비교 분석 보고서 | 20+ 페이지 | 30+ 페이지 | 5개 모델 종합 |

### 학습 성과

**기술적 역량:**

- ✅ VLA 모델 파인튜닝 및 배포 (GR00T N1.6)
- ✅ Video-to-Policy 모델 추론 및 평가 (Cosmos Policy)
- ✅ 세계 모델 후훈련 및 합성 비디오 생성 (Cosmos Predict2.5)
- ✅ IDM pseudo labeling 및 시너지 파이프라인 (GR00T IDM)
- ✅ 실제 API vs 논문 의사코드 갭 분석 능력
- ✅ Docker 기반 ML 환경 구축

**실무 경험:**

- ✅ 대규모 딥러닝 파인튜닝 (GR00T)
- ✅ 로봇 하드웨어 제어 및 데이터 수집
- ✅ 다양한 데이터 포맷 변환 (LeRobot v2, pickle, MP4)
- ✅ 체계적 벤치마킹 및 비교 분석
- ✅ 실험 설계 및 재현 가능한 연구

**Option C+ 접근법의 장점:**

- 12주 내 현실적으로 달성 가능한 목표 설정
- GR00T 파인튜닝을 통한 심화 실습 경험
- GPU 비용 최적화 (추론만으로도 모델 특성 충분히 이해)
- 향후 GPU 리소스 확보 시 Cosmos Policy 파인튜닝으로 확장 가능

이 12주 커리큘럼을 통해 NVIDIA의 로봇 AI 모델을 실제 소스코드 수준에서 이해하고, GR00T N1.6 파인튜닝의 실전 경험과 함께 Cosmos Predict2.5 + IDM 시너지 파이프라인을 통한 합성 데이터 증강 역량을 확보할 수 있습니다.

Option C+ 하이브리드 접근법을 통해 1x RTX 4090 환경에서도 VLA 파인튜닝, 세계 모델 후훈련, IDM pseudo labeling을 모두 수행하며, 향후 GPU 리소스 확보 시 Cosmos Policy 파인튜닝으로 확장할 수 있는 기반을 마련합니다.

---

