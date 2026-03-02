# 소스코드 설명 문서

**프로젝트**: NVIDIA 로봇 AI 모델 종합 벤치마킹 (12주)
**플랫폼**: ROBOTIS OMX (5-DOF + 1 Gripper, 6-dim action space)
**접근법**: Option C+ 하이브리드 (ACT 베이스라인 + GR00T 파인튜닝 + Cosmos Predict2.5 후훈련/IDM 시너지)

---

## 파일 구조 요약

```
pseudo-project/
├── configs/
│   └── omx_modality_config.py            # GR00T 파인튜닝용 OMX 모달리티 설정
├── utils/                                 # OMX 공통 유틸리티
│   ├── __init__.py
│   ├── omx_constants.py                   # OMX 상수, URDF kinematic 파라미터, 관절 매핑, dtype 변환
│   └── omx_fk.py                          # URDF 기반 FK (joint → EE pose → Cosmos state/action)
├── scripts/
│   ├── week1_setup_omx_env.sh             # OMX 하드웨어 환경 구축
│   ├── week2_setup_lerobot.sh             # LeRobot 설치 + 텔레오퍼레이션
│   ├── week3_test_groot_inference.py      # GR00T 추론 파이프라인 검증
│   ├── week4_collect_omx_data.sh          # OMX 데이터 수집 (50 ep x 3 tasks)
│   ├── week4_convert_omx_to_groot.py      # HF datasets → GR00T LeRobot v2 변환
│   ├── week5_train_act_baseline.sh        # ACT 베이스라인 학습
│   ├── week5_finetune_groot_omx.sh        # GR00T N1.6 파인튜닝
│   ├── week6_deploy_groot_omx.py          # GR00T OMX 실제 로봇 배포
│   ├── week6_eval_omx.py                 # ACT vs GR00T 비교 평가
│   ├── week7_eval_cosmos_libero.py        # Cosmos Policy LIBERO 벤치마크
│   ├── week8_cosmos_groot_comparison.py   # Cosmos Policy vs GR00T 비교 분석
│   ├── week9_run_dreamdojo_rollout.py     # DreamDojo 롤아웃 생성 + 분석 (참고)
│   ├── week10_cross_model_analysis.py     # 통합 파이프라인 + IDM 시너지
│   └── week11_benchmark_report.py         # 종합 벤치마킹 보고서 생성
└── SCRIPTS.md                             # 이 문서
```

---

## 1. configs/omx_modality_config.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 4-5 (데이터 변환 ~ 파인튜닝) |
| **목적** | GR00T N1.6 파인튜닝 시 OMX 로봇의 6차원 액션/상태 공간을 정의 |
| **실행 위치** | Isaac-GR00T 디렉토리 내에서 참조 |
| **라인 수** | 72 |

### 핵심 내용

GR00T의 `launch_finetune.py`는 `--modality-config-path` 플래그로 Python 파일을 로드합니다. 이 파일은 `register_modality_config()` API를 사용하여 OMX의 모달리티 구성을 등록합니다.

**4개 모달리티 정의:**

| 모달리티 | 키 | delta_indices | 설명 |
|----------|-----|---------------|------|
| `video` | `cam1` | `[0]` | 단일 카메라 (224x224) |
| `state` | `joint1~5, gripper` | `[0]` | 6-dim 현재 상태 |
| `action` | `joint1~5, gripper` | `[0..15]` | 6-dim, action_horizon=16 |
| `language` | `annotation.human.action.task_description` | `[0]` | 자연어 작업 지시 |

**ActionConfig 구분:**
- `joint1~joint5`: `ActionRepresentation.RELATIVE` + `ActionType.NON_EEF` (상대 관절 위치)
- `gripper`: `ActionRepresentation.ABSOLUTE` + `ActionType.GRIPPER` (절대 그리퍼 위치)

### 사용법

```bash
# launch_finetune.py에서 자동 로드
uv run python gr00t/experiment/launch_finetune.py \
    --modality-config-path configs/omx_modality_config.py \
    --embodiment-tag NEW_EMBODIMENT ...
```

### 의존성

- `gr00t.configs.data.embodiment_configs` → `register_modality_config()`
- `gr00t.data.types` → `ModalityConfig`, `ActionConfig`, `ActionRepresentation`, `ActionType`, `ActionFormat`
- `gr00t.data.embodiment_tags` → `EmbodimentTag.NEW_EMBODIMENT`

### 설계 결정

JSON이 아닌 Python 파일을 사용하는 이유: GR00T의 `load_modality_config()` 함수가 `importlib`로 Python 모듈을 동적 임포트하여 `register_modality_config()` 호출을 실행합니다. JSON 형식은 지원하지 않습니다.

---

## 2. scripts/week1_setup_omx_env.sh

| 항목 | 내용 |
|------|------|
| **Week** | Week 1 |
| **목적** | Docker, NVIDIA Container Toolkit, OMX 컨테이너 환경 구축 |
| **실행 위치** | 호스트 머신 |
| **라인 수** | 75 |

### 수행 단계

1. **Docker + NVIDIA GPU 검증**: `docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
2. **Open Manipulator 컨테이너**: `github.com/ROBOTIS-GIT/open_manipulator` 클론 → `container.sh start`
3. **Physical AI Tools 컨테이너**: `github.com/ROBOTIS-GIT/physical_ai_tools.git` 클론 → `container.sh start`
4. **USB 포트 확인**: `/dev/serial/by-id/` 검색 (Leader/Follower 시리얼 ID)
5. **카메라 확인**: `/dev/video*` 디바이스 검색

### 사전 요구사항

- Ubuntu 22.04+
- Docker Engine
- NVIDIA Container Toolkit
- OMX-L (Leader) + OMX-F (Follower) 하드웨어

### 출력

- `${HOME}/omx_workspace/` 디렉토리에 두 개 컨테이너 실행
- USB 시리얼 및 카메라 디바이스 확인 결과

---

## 3. scripts/week2_setup_lerobot.sh

| 항목 | 내용 |
|------|------|
| **Week** | Week 2 |
| **목적** | ROBOTIS fork LeRobot 설치, 텔레오퍼레이션 테스트, 샘플 녹화 |
| **실행 위치** | 호스트 머신 (conda 환경) |
| **라인 수** | 76 |

### 수행 단계

1. **ROBOTIS fork LeRobot 설치**: conda 환경(`lerobot`) 생성 → `github.com/ROBOTIS-GIT/lerobot` 클론 → `pip install -e ".[dynamixel]"`
2. **USB 포트 검색**: `lerobot-find-port` 명령어 안내
3. **텔레오퍼레이션 명령어**: 카메라 유/무 버전의 `python -m lerobot.teleoperate` 예시
4. **샘플 녹화**: `lerobot-record`로 5개 에피소드 녹화 예시

### 주의사항

- 반드시 `ROBOTIS-GIT/lerobot`을 사용해야 합니다 (원본 `huggingface/lerobot` 사용 불가)
- `ffmpeg=6.1.1`을 conda-forge에서 설치 필요
- HuggingFace 인증(`huggingface-cli login`) 필요

### 의존성

- conda (Miniconda/Anaconda)
- `dynamixel-sdk` (pip install 시 자동 포함)

---

## 4. scripts/week3_test_groot_inference.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 3 |
| **목적** | GR00T N1.6 사전훈련 모델의 OMX 6-dim 추론 파이프라인 검증 |
| **실행 위치** | Isaac-GR00T 디렉토리 (`uv run`) |
| **라인 수** | 110 |

### 수행 단계

1. **모델 로드**: `Gr00tPolicy(model_path="nvidia/GR00T-N1.6-3B", embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device="cuda")`
2. **단일 추론**: 더미 관측값으로 `get_action(obs)` 호출, 액션 shape/range 확인
3. **속도 벤치마크**: 10회 추론의 평균/표준편차 레이턴시 측정 (목표: >10 Hz)
4. **멀티태스크**: 3개 작업(pick/place/stack)으로 언어 조건부 추론 동작 확인

### 핵심 함수

- `create_omx_observation(task)`: OMX 6-dim 더미 관측 딕셔너리 생성
  - `video.cam1`: `(1, 1, 224, 224, 3)` float32
  - `state.joint1~gripper`: 각각 `(1, 1, 1)` float32
  - `annotation.task`: `[[task_string]]`

### API 사용 패턴

```python
action, info = policy.get_action(obs)  # tuple[dict, dict] 반환
```

### 성공 기준

- OMX ros2_control은 100Hz → GR00T 추론이 >10Hz이면 실시간 제어 가능
- RTX 4090 기준 약 ~44ms (≈22Hz) 예상

---

## 5. scripts/week4_collect_omx_data.sh

| 항목 | 내용 |
|------|------|
| **Week** | Week 4 |
| **목적** | OMX 텔레오퍼레이션으로 3개 작업 × 50 에피소드 데이터 수집 |
| **실행 위치** | 호스트 머신 (LeRobot conda 환경) |
| **라인 수** | 61 |

### 수집 작업

| Task ID | 작업 설명 | 에피소드 |
|---------|----------|---------|
| `omx_pick` | Pick up the object | 50 |
| `omx_place` | Place the object on the target | 50 |
| `omx_stack` | Stack the cubes | 50 |

### 사용법

```bash
# 환경변수 설정 후 실행
export HF_USER=$(huggingface-cli whoami | head -1)
bash scripts/week4_collect_omx_data.sh
```

### 출력

- HuggingFace Hub에 3개 데이터셋 업로드:
  - `${HF_USER}/omx_pick`
  - `${HF_USER}/omx_place`
  - `${HF_USER}/omx_stack`
- 포맷: HuggingFace datasets (6-dim state/action, 15 FPS)

### 조작 키

- `→`: 에피소드 완료
- `←`: 현재 에피소드 재녹화
- `ESC`: 중지

---

## 6. scripts/week4_convert_omx_to_groot.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 4 |
| **목적** | HuggingFace datasets → GR00T LeRobot v2 포맷 변환 |
| **실행 위치** | 호스트 머신 |
| **라인 수** | 189 |

### 변환 파이프라인

```
HuggingFace datasets (6-dim flat)
    ↓
GR00T LeRobot v2:
    ├── data/episode_*.parquet    (state/action을 개별 joint 컬럼으로 분리)
    ├── videos/cam1_episode_*.mp4 (이미지 컬럼에서 추출)
    └── meta/
        ├── modality.json         (video/state/action/annotation 구조)
        └── info.json             (소스, DOF, FPS 메타데이터)
```

### 핵심 함수

| 함수 | 역할 |
|------|------|
| `create_modality_json()` | `meta/modality.json` 생성 (video, state, action, annotation) |
| `convert_hf_to_groot_v2()` | 전체 변환 파이프라인 실행 |

### 상태/액션 분리 로직

OMX의 6-dim flat 벡터를 GR00T가 요구하는 개별 joint 컬럼으로 분리:

```python
OMX_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]
# state[0] → state.joint1, state[5] → state.gripper
# action[0] → action.joint1, action[5] → action.gripper
```

### 사용법

```bash
python scripts/week4_convert_omx_to_groot.py \
    --hf-repo-id $HF_USER/omx_pick \
    --output-dir data/omx_groot_v2/pick \
    --task-description "Pick up the object"
```

### 의존성

- `datasets` (HuggingFace)
- `pandas`, `pyarrow` (Parquet 변환)
- `opencv-python` (비디오 추출, 선택사항)

---

## 7. scripts/week5_train_act_baseline.sh

| 항목 | 내용 |
|------|------|
| **Week** | Week 5 |
| **목적** | ACT (Action Chunking Transformers) 베이스라인 학습 |
| **실행 위치** | 호스트 머신 (LeRobot conda 환경) |
| **라인 수** | 58 |

### 역할

ACT는 OMX 공식 지원 정책으로, GR00T와의 비교 베이스라인 역할을 합니다. CPU에서도 추론 가능하며 학습이 빠릅니다.

### 학습 설정

| 파라미터 | 기본값 | 환경변수 |
|---------|--------|----------|
| Epochs | 100 | `NUM_EPOCHS` |
| Batch size | 64 | `BATCH_SIZE` |
| Device | cuda | `DEVICE` |

### 사용법

```bash
# 3개 작업에 대해 순차 학습
bash scripts/week5_train_act_baseline.sh
```

### 출력

각 작업별 두 파일:
- `outputs/train/act_omx_pick/config.json`
- `outputs/train/act_omx_pick/model.safetensors`

### ACT vs GR00T 비교 포인트

| 항목 | ACT | GR00T |
|------|-----|-------|
| GPU 요구 | CPU 가능 | 1x RTX 4090 |
| 학습 시간 | ~1시간 | ~4시간+ |
| 추론 속도 | <10ms | ~44ms |
| 언어 이해 | 불가 | 가능 |

---

## 8. scripts/week5_finetune_groot_omx.sh

| 항목 | 내용 |
|------|------|
| **Week** | Week 5 |
| **목적** | GR00T N1.6을 OMX 6-dim 데이터로 파인튜닝 |
| **실행 위치** | Isaac-GR00T 디렉토리 |
| **라인 수** | 83 |

### 사전 검증

1. Isaac-GR00T 디렉토리 존재 확인
2. `omx_modality_config.py` 존재 확인
3. 변환된 데이터셋 디렉토리 존재 확인
4. GPU VRAM 확인 (`nvidia-smi`)

### 파인튜닝 설정

| 파라미터 | 기본값 | 환경변수 |
|---------|--------|----------|
| Base model | `nvidia/GR00T-N1.6-3B` | - |
| Embodiment | `NEW_EMBODIMENT` | - |
| Max steps | 10,000 | `MAX_STEPS` |
| Batch size | 32 | `BATCH_SIZE` |
| Learning rate | 1e-4 | `LEARNING_RATE` |
| GPUs | 1 | `NUM_GPUS` |

### 튜닝 전략

```
--tune-llm false           # LLM 동결 (언어 능력 보존)
--tune-visual false        # 비전 인코더 동결
--tune-projector true      # 프로젝터 학습 (모달리티 정렬)
--tune-diffusion-model true # 디퓨전 모델 학습 (액션 생성)
```

### 사용법

```bash
cd Isaac-GR00T
bash ../pseudo-project/scripts/week5_finetune_groot_omx.sh
```

### 출력

- `outputs/groot_omx/checkpoint-{step}/` (최대 3개 보존)

---

## 9. scripts/week6_deploy_groot_omx.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 6 |
| **목적** | 파인튜닝된 GR00T를 OMX 로봇에 실시간 배포 |
| **실행 위치** | Isaac-GR00T 디렉토리 (`uv run`) |
| **라인 수** | 228 |

### 클래스/함수 구조

| 이름 | 역할 |
|------|------|
| `OMXRobotInterface` | OMX ROS 2 인터페이스 (관측 획득, 액션 전송, 리셋) |
| `run_deployment()` | 배포 루프 (에피소드 반복, 레이턴시 측정, 타임아웃 처리) |

### OMXRobotInterface 메서드

| 메서드 | 기능 |
|--------|------|
| `__init__()` | USB 포트 + 카메라 인덱스 초기화 |
| `get_observation(task)` | 카메라 이미지 + 관절 상태 → GR00T 관측 딕셔너리 |
| `execute_action(action)` | 안전 클리핑 후 관절 명령 전송 |
| `reset()` | 홈 포지션 복귀 |

### 안전 제한

```python
MAX_JOINT_VELOCITY = 0.5   # rad/step
MAX_EPISODES = 100
EPISODE_TIMEOUT_SEC = 60
```

### 사용법

```bash
cd Isaac-GR00T
uv run python ../pseudo-project/scripts/week6_deploy_groot_omx.py \
    --checkpoint outputs/groot_omx/checkpoint-best \
    --task "pick up the red cube"
```

### 실제 배포 시 필요한 작업

현재 더미 데이터를 사용하는 부분을 실제 ROS 2 구현으로 교체:
- `rclpy` 노드 생성, `JointState` 구독, `JointTrajectory` 발행
- 성공 조건 판정 로직 (현재 30스텝 후 항상 성공)

---

## 10. scripts/week6_eval_omx.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 6 |
| **목적** | ACT vs GR00T 동일 조건 비교 평가 |
| **실행 위치** | Isaac-GR00T 디렉토리 (`uv run`) |
| **라인 수** | 272 |

### 데이터 클래스

| 클래스 | 필드 |
|--------|------|
| `EvalResult` | policy, task, trial, success, steps, elapsed_sec, latency_ms, jerk |
| `EvalSummary` | policy, task, num_trials, success_rate, avg_elapsed_sec, avg_latency_ms, avg_jerk |

### 핵심 함수

| 함수 | 역할 |
|------|------|
| `compute_jerk()` | 궤적의 3차 미분(jerk) 계산 → 낮을수록 부드러운 동작 |
| `eval_groot()` | GR00T 정책 실제 추론 평가 |
| `eval_act_cli()` | ACT 정책 시뮬레이션 평가 (~5ms CPU 추론) |
| `summarize()` | 평가 결과 통계 요약 |
| `print_comparison()` | ACT vs GR00T 비교 테이블 출력 |

### 평가 메트릭

| 메트릭 | 설명 | 승리 기준 |
|--------|------|----------|
| Success Rate | 작업 성공률 | 높을수록 좋음 |
| Latency (ms) | 추론 지연시간 | 낮을수록 좋음 |
| Smoothness (jerk) | 궤적 부드러움 | 낮을수록 좋음 |

### 사용법

```bash
cd Isaac-GR00T
uv run python ../pseudo-project/scripts/week6_eval_omx.py \
    --groot-checkpoint outputs/groot_omx/checkpoint-best \
    --act-checkpoint outputs/train/act_omx_pick \
    --task pick_cube --num-trials 20
```

### 출력

- `outputs/eval/eval_pick_cube.json`
- 비교 테이블 (터미널 출력)

---

## 11. scripts/week7_eval_cosmos_libero.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 7-8 |
| **목적** | Cosmos Policy 사전훈련 모델로 LIBERO 벤치마크 평가 |
| **실행 위치** | cosmos-policy Docker 컨테이너 내부 |
| **라인 수** | 152 |

### LIBERO 벤치마크

| Suite | 논문 결과 | 설명 |
|-------|----------|------|
| `libero_spatial` | 98.1% | 공간 관계 이해 |
| `libero_object` | 100.0% | 객체 인식 |
| `libero_goal` | 98.2% | 목표 조건 달성 |
| `libero_long` | 97.6% | 장기 시퀀스 |

### Cosmos Policy API 패턴

```python
from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, load_dataset_stats
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    libero_task_suite=suite,
    num_episodes_per_task=num_episodes,
)
model, cosmos_config = get_model(cfg)
```

### 사용법

```bash
# Docker 내부에서 실행
python scripts/week7_eval_cosmos_libero.py --suites all --num-episodes 20

# 또는 공식 eval.sh 사용
bash eval.sh cosmos_predict2_2b_480p_libero__inference_only \
    nvidia/Cosmos-Policy-LIBERO-Predict2-2B spatial 20
```

### 의존성

- cosmos-policy Docker 컨테이너
- GPU 6-10 GB VRAM

### 출력

- `outputs/cosmos_eval/cosmos_libero_eval.json`

---

## 12. scripts/week9_run_dreamdojo_rollout.py (참고)

> **참고**: DreamDojo는 본 프로젝트에서 Cosmos Predict2.5 Action-Conditioned 모델로 대체되었습니다.
> 이 스크립트는 DreamDojo 세계 모델의 API 패턴과 품질 메트릭 참고용으로 유지됩니다.

| 항목 | 내용 |
|------|------|
| **Week** | Week 9 (참고용) |
| **목적** | DreamDojo 세계 모델 롤아웃 생성 및 품질 분석 (참고) |
| **실행 위치** | DreamDojo 환경 |
| **라인 수** | 204 |
| **대체** | Cosmos Predict2.5 Action-Conditioned (Week 9 트랙A) |

### DreamDojo vs Cosmos Predict2.5 비교

| 항목 | DreamDojo | Cosmos Predict2.5 |
|------|-----------|-------------------|
| 유형 | World Foundation Model | Action-Conditioned World Model |
| 입력 | 초기 프레임 + 텍스트 | 초기 프레임 + EE 행동 |
| 후훈련 | 8x H100 80GB 필요 | 1x RTX 4090 가능 |
| OMX 적용 | 불가 (로봇별 체크포인트 없음) | FK 변환으로 적용 가능 |

### 지원 모델 (DreamDojo)

| 모델 | 로봇 | 해상도 | 파라미터 |
|------|------|--------|---------|
| `nvidia/DreamDojo-2B-480p-GR1` | GR-1 | 480x640 | 2B |
| `nvidia/DreamDojo-2B-480p-G1` | Unitree G1 | 480x640 | 2B |
| `nvidia/DreamDojo-2B-480p-YAM` | YAM | 480x640 | 2B |

### 품질 메트릭 (Cosmos Predict2.5 평가에도 재사용)

| 메트릭 | 계산 방법 | 의미 |
|--------|----------|------|
| temporal_consistency | `1 - mean(frame_diff) / 255` | 프레임 간 일관성 (높을수록 좋음) |
| long_horizon_stability | `1 - abs(early_mean - late_mean) / 255` | 장기 안정성 (높을수록 좋음) |
| mean_intensity | `mean(frames) / 255` | 평균 밝기 |
| std_intensity | `std(frames) / 255` | 밝기 분산 |

### 사용법

```bash
python scripts/week9_run_dreamdojo_rollout.py \
    --model nvidia/DreamDojo-2B-480p-GR1 \
    --num-rollouts 5 --rollout-length 150
```

---

## 13. scripts/week11_benchmark_report.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 11 |
| **목적** | 12주간 실험 결과를 종합하여 마크다운 보고서 생성 |
| **실행 위치** | 프로젝트 루트 |
| **라인 수** | 264 |

### 보고서 섹션

| 섹션 | 내용 | 데이터 소스 |
|------|------|------------|
| 1. 실험 개요 | 4개 모델 역할 정리 | 정적 |
| 2. ACT vs GR00T 비교 | Success Rate, Latency, Smoothness | `outputs/eval/eval_*.json` |
| 3. Cosmos Policy LIBERO | 논문 결과 vs 재현 결과 | `outputs/cosmos_eval/cosmos_libero_eval.json` |
| 4. Cosmos Predict2.5 합성 비디오 | Temporal Consistency, Stability | `outputs/cosmos_predict/cosmos_predict_report.json` |
| 5. IDM 시너지 파이프라인 | 증강 전/후 성능 비교 | `outputs/week10_analysis/cross_model_analysis.json` |
| 6. 종합 비교 매트릭스 | 모델 타입, GPU, 속도, 데이터 효율 | 정적 |
| 7. 배포 시나리오 | 사용 사례별 권장 모델 | 정적 |
| 8. 학습된 교훈 | API 괴리, GPU 제약, 데이터 포맷 | 정적 |

### 핵심 함수

| 함수 | 역할 |
|------|------|
| `load_json()` | JSON 파일 안전 로드 (파일 없으면 None) |
| `load_eval_results()` | `eval_*.json` 패턴으로 Week 6 결과 일괄 로드 |
| `generate_report()` | 마크다운 보고서 생성 (데이터 없으면 플레이스홀더) |

### 사용법

```bash
python scripts/week11_benchmark_report.py \
    --eval-dir outputs/eval \
    --cosmos-dir outputs/cosmos_eval \
    --dreamdojo-dir outputs/dreamdojo_rollouts \
    --output outputs/final_report.md
```

### 출력

- `outputs/final_report.md` (마크다운 보고서)

---

## 14. utils/omx_constants.py

| 항목 | 내용 |
|------|------|
| **목적** | OMX 로봇 공통 상수, URDF kinematic 파라미터, IDM↔VLA 관절 매핑, 비디오 dtype 변환 |
| **참조 스크립트** | week3, week4, week6_deploy, week6_eval, week8, week10, `omx_fk.py` |
| **라인 수** | 209 |

### OMX 기본 상수

| 상수 | 값 | 설명 |
|------|-----|------|
| `OMX_DOF` | `6` | 5 joints + 1 gripper |
| `OMX_IMG_SIZE` | `224` | GR00T VLA 입력 이미지 크기 |
| `OMX_CONTROL_HZ` | `100` | ROS 2 ros2_control 제어 주기 |
| `OMX_JOINT_NAMES` | `["joint1", ..., "gripper"]` | VLA 관절 이름 (기능 무관) |
| `OMX_JOINT_AXES` | `["Z", "Y", "Y", "Y", "X"]` | 관절 회전축 (OMX-F/L 공통) |

### URDF 기반 Kinematic 파라미터

URDF에서 추출한 각 관절의 회전축과 부모→자식 링크 간 오프셋입니다. `omx_fk.py`에서 FK 계산에 사용됩니다.

**OMX-F (Follower, 매니퓰레이션, 작업 반경 400mm)**:

| 관절 | 축 | 오프셋 (x, y, z) m | 설명 |
|------|-----|-------------------|------|
| joint1 | Z | (-0.01125, 0, 0.034) | 베이스 수평 회전 |
| joint2 | Y | (0, 0, 0.0635) | 어깨 수직 회전 |
| joint3 | Y | (0.0415, 0, 0.11315) | 팔꿈치 굴곡 |
| joint4 | Y | (0.162, 0, 0) | 손목 굴곡 |
| joint5 | X | (0.0287, 0, 0) | 손목 회전 |
| EE offset | - | (0.09193, -0.0016, 0) | End-Effector (fixed) |

**OMX-L (Leader, 텔레오퍼레이션, 작업 반경 335mm)**:

| 관절 | 축 | 오프셋 (x, y, z) m | 설명 |
|------|-----|-------------------|------|
| joint1 | Z | (-0.0095, 0, 0.0545) | 베이스 수평 회전 |
| joint2 | Y | (0, 0, 0.042) | 어깨 수직 회전 |
| joint3 | Y | (0.0375, 0, 0.09) | 팔꿈치 굴곡 |
| joint4 | Y | (0.1275, 0, 0) | 손목 굴곡 |
| joint5 | X | (0.0287, 0, 0) | 손목 회전 |

**그리퍼 차이**:
- OMX-F: 2핑거 mimic (Z축), `gripper_joint_2 = -1 * gripper_joint_1`
- OMX-L: 1핑거 (Y축), `end_effector_link` 미정의

### IDM ↔ VLA 관절 이름 매핑

GR00T IDM은 해부학적 이름을, VLA는 기능 무관 이름을 사용합니다. 두 모델 간 데이터 교환 시 매핑이 필요합니다.

| VLA (GR00T N1.6) | IDM (GR00T) | 축 | 설명 |
|-------------------|------------|-----|------|
| `joint1` | `shoulder_pan` | Z | 베이스 수평 회전 |
| `joint2` | `shoulder_lift` | Y | 어깨 수직 회전 |
| `joint3` | `elbow_flex` | Y | 팔꿈치 굴곡 |
| `joint4` | `wrist_flex` | Y | 손목 굴곡 |
| `joint5` | `wrist_roll` | X | 손목 회전 |
| `gripper` | `gripper` | - | 그리퍼 개폐 |

```python
from utils.omx_constants import OMX_JOINT_MAPPING, OMX_JOINT_MAPPING_INV

# VLA → IDM
OMX_JOINT_MAPPING["joint1"]  # → "shoulder_pan"

# IDM → VLA
OMX_JOINT_MAPPING_INV["shoulder_pan"]  # → "joint1"
```

### 비디오 dtype 가이드

| 모델 | dtype | 범위 | 용도 |
|------|-------|------|------|
| GR00T N1.6 VLA | `float32` | `[0.0, 1.0]` | 추론/파인튜닝 |
| GR00T IDM | `uint8` | `[0, 255]` | pseudo labeling |

```python
from utils.omx_constants import convert_video_vla_to_idm, convert_video_idm_to_vla

# VLA → IDM (Cosmos-IDM 시너지 파이프라인에서 사용)
idm_video = convert_video_vla_to_idm(vla_video)  # float32 → uint8

# IDM → VLA (pseudo label 학습 데이터 생성 시)
vla_video = convert_video_idm_to_vla(idm_video)  # uint8 → float32
```

### 더미 관측 함수

| 함수 | 포맷 | 용도 |
|------|------|------|
| `create_omx_observation(task)` | VLA (float32) | GR00T N1.6 추론 테스트 |
| `create_omx_observation_idm()` | IDM (uint8) | GR00T IDM 추론 테스트 |

---

## 15. utils/omx_fk.py

| 항목 | 내용 |
|------|------|
| **목적** | URDF 기반 Forward Kinematics: OMX joint → EE pose → Cosmos Predict2.5 입력 변환 |
| **참조 스크립트** | week10 (Cosmos 후훈련 데이터 변환) |
| **라인 수** | 275 |

### 역할

Cosmos Predict2.5 Action-Conditioned 모델은 EE-space 입력을 요구합니다. OMX는 joint-space로 데이터를 수집하므로, FK를 통해 joint → EE 변환이 필요합니다.

```
OMX joint positions (5-dim)
  → FK (동차 변환 행렬 체인)
  → EE pose: [x, y, z, roll, pitch, yaw] (6-dim)
  → Cosmos state/action 형식
```

### 클래스: OMXForwardKinematics

| 메서드 | 입력 | 출력 | 설명 |
|--------|------|------|------|
| `compute()` | joint positions (5) | position + rpy + transform | EE 포즈 계산 |
| `to_cosmos_state()` | joint positions + gripper | state (6) + gripper (1) | Cosmos 입력 변환 |
| `compute_cosmos_action()` | 연속 두 프레임 joints | action (7-dim) | 상대 변위 계산 |
| `batch_to_cosmos_states()` | 궤적 (T, 5) + (T,) | states + actions + grippers | 일괄 변환 |

### Cosmos Predict2.5 데이터 형식

| 키 | 차원 | 설명 |
|----|------|------|
| `state` | (6,) | `[x, y, z, roll, pitch, yaw]` EE 포즈 |
| `continuous_gripper_state` | (1,) | `[0.0 ~ 1.0]` 그리퍼 개폐 |
| `action` | (7,) | 연속 프레임 간 상대 변위 `[dx, dy, dz, dr, dp, dy, dg]` |

### 사용법

```python
from utils.omx_fk import OMXForwardKinematics

fk = OMXForwardKinematics(robot="omx_f")

# 단일 관절 → EE 포즈
ee = fk.compute([0.0, -0.5, 0.3, 0.2, 0.0])
# → {"position": [x, y, z], "rpy": [roll, pitch, yaw], "transform": 4x4}

# Cosmos state 변환
cosmos_state = fk.to_cosmos_state([0.0, -0.5, 0.3, 0.2, 0.0], gripper=0.5)
# → {"state": [x, y, z, r, p, y], "continuous_gripper_state": 0.5}

# 궤적 일괄 변환
batch = fk.batch_to_cosmos_states(joint_trajectory, gripper_trajectory)
# → {"states": (T, 6), "actions": (T-1, 7), "continuous_gripper_states": (T,)}
```

### FK 검증 결과 (홈 포지션)

| 로봇 | EE 위치 (x, y, z) m | 설명 |
|------|---------------------|------|
| OMX-F | (0.313, -0.002, 0.211) | 팔 완전 전방 신전 |
| OMX-L | (0.184, 0.0, 0.187) | 작업 반경 335mm 반영 |

### 의존성

- `numpy`
- `utils.omx_constants` → `OMX_F_JOINT_PARAMS`, `OMX_F_EE_OFFSET`, `OMX_L_JOINT_PARAMS`

---

## 16. scripts/week8_cosmos_groot_comparison.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 8 |
| **목적** | Cosmos Policy vs GR00T N1.6 정량적 비교 분석 |
| **실행 위치** | 프로젝트 루트 |
| **라인 수** | 290 |

### 분석 내용

1. **데이터 포맷 차이**: LeRobot v2 vs Cosmos pickle (비디오 dtype, 상태 포맷, 언어 입력, 액션 출력)
2. **정량적 비교 테이블**: 성공률, 추론 속도, 데이터 요구량, GPU 요구사항
3. **강점/약점 실증**: 각 모델 5개 이상의 강점/약점 분석
4. **적합 시나리오 판정**: 6개 시나리오별 권장 모델

### 핵심 함수

| 함수 | 역할 |
|------|------|
| `compare_data_formats()` | 두 모델의 데이터 포맷 구조적 차이 분석 |
| `build_comparison_table()` | Week 6/7 결과 기반 정량 비교 테이블 생성 |
| `analyze_strengths_weaknesses()` | 강점/약점 실증 분석 |
| `determine_scenarios()` | 적합 시나리오별 권장 모델 판정 |
| `generate_report()` | 마크다운 비교 보고서 생성 |

### 사용법

```bash
python scripts/week8_cosmos_groot_comparison.py \
    --groot-eval-dir outputs/eval \
    --cosmos-eval-dir outputs/cosmos_eval \
    --output-dir outputs/week8_comparison
```

### 출력

- `outputs/week8_comparison/cosmos_groot_comparison.md` (마크다운 보고서)
- `outputs/week8_comparison/cosmos_groot_comparison.json` (구조화된 데이터)

---

## 17. scripts/week10_cross_model_analysis.py

| 항목 | 내용 |
|------|------|
| **Week** | Week 10 |
| **목적** | 통합 파이프라인 (Cosmos Predict2.5 + IDM 시너지) + 증강 재학습 |
| **실행 위치** | 프로젝트 루트 |
| **라인 수** | 380 |

### Cosmos Predict2.5 + IDM 시너지 파이프라인

```
Cosmos Predict2.5 합성 비디오 (action-conditioned)
  → resize + 2-frame pairing
  → GR00T IDM pseudo labeling
  → 관절 이름 매핑 (IDM→VLA: shoulder_pan→joint1 등)
  → 품질 평가 (grade ≥ B만 통과)
  → dtype 변환 (uint8→float32)
  → GR00T N1.6 VLA 재학습 데이터 증강
```

### 핵심 함수

| 함수 | 역할 |
|------|------|
| `load_cosmos_rollout()` | Cosmos Predict2.5 합성 비디오 로드 (또는 더미 생성) |
| `prepare_idm_input()` | 합성 비디오 프레임 → IDM 2-frame pair 변환 |
| `simulate_idm_inference()` | IDM pseudo labeling 시뮬레이션 |
| `evaluate_pseudo_labels()` | jerk, temporal consistency 기반 품질 평가 |
| `build_model_comparison()` | 모델 종합 비교 테이블 |
| `design_integration_pipeline()` | 통합 파이프라인 설계 (증강 전/후 비교) |

### 품질 평가 메트릭

| 메트릭 | 계산 방법 | 의미 |
|--------|----------|------|
| `jerk` | 3차 미분 크기 | 낮을수록 부드러운 동작 |
| `temporal_consistency` | 연속 예측 간 차이 | 낮을수록 일관성 높음 |
| `quality_grade` | jerk + consistency 기반 | A~D 등급 (B 이상 권장) |

### 데이터 변환 흐름

| 단계 | 변환 | 유틸리티 |
|------|------|---------|
| OMX joint → EE pose | FK (Z,Y,Y,Y,X) | `utils/omx_fk.py` |
| IDM 관절 → VLA 관절 | shoulder_pan → joint1 등 | `OMX_JOINT_MAPPING_INV` |
| IDM 비디오 → VLA 비디오 | uint8 [0,255] → float32 [0,1] | `convert_video_idm_to_vla()` |

### 사용법

```bash
python scripts/week10_cross_model_analysis.py \
    --cosmos-predict-dir outputs/cosmos_predict \
    --groot-eval-dir outputs/eval \
    --cosmos-eval-dir outputs/cosmos_eval \
    --output-dir outputs/week10_analysis
```

### 출력

- `outputs/week10_analysis/cross_model_analysis.md` (마크다운 보고서)
- `outputs/week10_analysis/cross_model_analysis.json` (구조화된 데이터)

### 참조 프로젝트

- `GR00T-IDM-Documentation/src/vla_action_quality.py` (품질 메트릭 원본)
- `GR00T-IDM-Documentation/src/idm_inference_example.py` (IDM 추론 패턴)

---

## 실행 순서 (전체 파이프라인)

```
Week 1    → week1_setup_omx_env.sh              (환경 구축)
Week 2    → week2_setup_lerobot.sh               (LeRobot 설치)
Week 3    → week3_test_groot_inference.py        (GR00T 추론 검증)
Week 4    → week4_collect_omx_data.sh            (데이터 수집)
          → week4_convert_omx_to_groot.py        (포맷 변환)
Week 5    → week5_train_act_baseline.sh          (ACT 학습)
          → week5_finetune_groot_omx.sh          (GR00T 파인튜닝)
Week 6    → week6_deploy_groot_omx.py            (GR00T 배포)
          → week6_eval_omx.py                   (ACT vs GR00T 평가)
Week 7-8  → [트랙A] week7_eval_cosmos_libero.py  (Cosmos Policy LIBERO 평가)
          → [트랙A] week8_cosmos_groot_comparison.py (Cosmos vs GR00T 비교)
          → [트랙B] utils/omx_fk.py 검증 + FK 변환 (Cosmos 데이터 준비)
Week 9-10 → [트랙A] Cosmos Predict2.5 OMX 후훈련 + 합성 비디오 생성
          → [트랙B] GR00T IDM pseudo labeling + 품질 필터링
          → week10_cross_model_analysis.py       (통합 파이프라인 + 증강 재학습)
Week 11   → week11_benchmark_report.py           (종합 보고서)
```

---

## API 참조 요약

### GR00T N1.6

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device="cuda",
)
action, info = policy.get_action(obs)  # → tuple[dict, dict]
```

### Cosmos Policy

```python
from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

cfg = PolicyEvalConfig(config=..., ckpt_path=..., ...)
model, cosmos_config = get_model(cfg)
```

### Cosmos Predict2.5 Action-Conditioned

```python
# OMX joint → Cosmos EE-space 변환
from utils.omx_fk import OMXForwardKinematics

fk = OMXForwardKinematics(robot="omx_f")
cosmos_state = fk.to_cosmos_state(joint_positions, gripper=0.5)
# → {"state": [x, y, z, r, p, y], "continuous_gripper_state": 0.5}

batch = fk.batch_to_cosmos_states(joint_trajectory, gripper_trajectory)
# → {"states": (T, 6), "actions": (T-1, 7), "continuous_gripper_states": (T,)}
```

### DreamDojo (참고)

```bash
# Python 패키지명: cosmos-predict2
cd DreamDojo && bash launch.sh configs/2b_480_640_gr1.yaml
```

### ACT (LeRobot)

```bash
# 학습
lerobot-train --dataset.repo_id=... --policy.type=act

# 배포
python -m lerobot.scripts.control_robot \
    --robot.type=omx_follower --control.policy.path=<checkpoint>
```
