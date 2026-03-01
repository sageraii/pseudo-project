# NVIDIA 로봇 AI 모델 벤치마킹 on ROBOTIS OMX

GR00T N1.6, Cosmos Policy, DreamDojo 세 가지 NVIDIA 로봇 AI 모델을 ROBOTIS OMX 6-DOF 매니퓰레이터에서 비교 평가하는 12주 연구 프로젝트입니다.

## 프로젝트 개요

Option C 하이브리드 접근법을 채택하여, ACT 베이스라인 학습과 GR00T 파인튜닝을 수행하고 Cosmos Policy 및 DreamDojo는 추론 기반으로 분석합니다.

| 모델 | 유형 | 역할 |
|------|------|------|
| GR00T N1.6 | Vision-Language-Action (VLA) | 카메라+언어 입력으로 직접 로봇 행동 생성 |
| Cosmos Policy | Video-to-Policy | 비디오 모델을 로봇 제어 정책으로 변환 |
| DreamDojo | World Foundation Model | 물리 시뮬레이션 및 미래 상태 예측 |

## 모델 비교

| 항목 | GR00T N1.6 | Cosmos Policy | DreamDojo |
|------|-----------|--------------|-----------|
| 입력 | 카메라 이미지 + 언어 + 로봇 상태 | 시각적 관찰 + 로봇 시연 | 로봇 모터 제어 명령 |
| 출력 | 로봇 행동 (액션 청크) | 행동 + 미래 상태 + 가치 | 미래 시각적 상태 시퀀스 |
| 데이터 규모 | 수천 시간 로봇 시연 | 50-300개 시연 (작업별) | 44,000시간 인간 영상 |
| GPU 요구사항 | 1x RTX 4090+ (추론/파인튜닝) | 8x H100 80GB (파인튜닝) | 고해상도 메모리 필요 |
| 실시간 성능 | 로봇 제어 주기 추론 | 플래닝 포함 실시간 | 10 FPS 생성 |
| 강점 | 범용성, 언어 이해, 오픈소스 | 데이터 효율성, SOTA 성능 | 물리 지식, 환경 일반화 |

## 디렉토리 구조

```
pseudo-project/
├── configs/
│   └── omx_modality_config.py       # GR00T N1.6 파인튜닝용 모달리티 설정
├── utils/
│   ├── __init__.py
│   └── omx_constants.py             # OMX 상수, 관절 매핑, dtype 변환
├── scripts/
│   ├── week1_setup_omx_env.sh       # OMX 하드웨어 환경 구축
│   ├── week2_setup_lerobot.sh       # LeRobot 설치 + 텔레오퍼레이션
│   ├── week3_test_groot_inference.py # GR00T 추론 파이프라인 검증
│   ├── week4_collect_omx_data.sh    # OMX 데이터 수집 (50 ep x 3 tasks)
│   ├── week4_convert_omx_to_groot.py# HF datasets -> GR00T LeRobot v2 변환
│   ├── week5_train_act_baseline.sh  # ACT 베이스라인 학습
│   ├── week5_finetune_groot_omx.sh  # GR00T N1.6 파인튜닝
│   ├── week6_deploy_groot_omx.py    # GR00T OMX 실제 로봇 배포
│   ├── week6_eval_omx.py           # ACT vs GR00T 비교 평가
│   ├── week7_eval_cosmos_libero.py  # Cosmos Policy LIBERO 벤치마크
│   ├── week8_cosmos_groot_comparison.py # Cosmos vs GR00T 비교 분석
│   ├── week9_run_dreamdojo_rollout.py  # DreamDojo 롤아웃 생성 + 분석
│   ├── week10_cross_model_analysis.py  # 세 모델 크로스 분석 + IDM 시너지
│   └── week11_benchmark_report.py      # 종합 벤치마킹 보고서 생성
├── Project.md                       # 상세 프로젝트 명세
├── SCRIPTS.md                       # 스크립트별 기술 문서
└── README.md
```

## 주차별 실행 파이프라인

| 주차 | 단계 | 스크립트 |
|------|------|---------|
| 1-2 | 환경 구축 (Docker, LeRobot, OMX) | `week1_setup_omx_env.sh`, `week2_setup_lerobot.sh` |
| 3 | GR00T 추론 검증 | `week3_test_groot_inference.py` |
| 4 | 데이터 수집 및 변환 | `week4_collect_omx_data.sh`, `week4_convert_omx_to_groot.py` |
| 5 | ACT 베이스라인 + GR00T 파인튜닝 | `week5_train_act_baseline.sh`, `week5_finetune_groot_omx.sh` |
| 6 | 배포 및 평가 (ACT vs GR00T) | `week6_deploy_groot_omx.py`, `week6_eval_omx.py` |
| 7 | Cosmos Policy LIBERO 벤치마크 | `week7_eval_cosmos_libero.py` |
| 8 | Cosmos vs GR00T 비교 | `week8_cosmos_groot_comparison.py` |
| 9 | DreamDojo 롤아웃 | `week9_run_dreamdojo_rollout.py` |
| 10 | 크로스 모델 분석 + IDM 시너지 | `week10_cross_model_analysis.py` |
| 11 | 최종 벤치마크 보고서 | `week11_benchmark_report.py` |

## 사전 요구사항

### 하드웨어

- ROBOTIS OMX 매니퓰레이터 (5-DOF + 1 Gripper)
- NVIDIA GPU (RTX 4090 이상 권장)
- USB 카메라 (224x224 해상도)

### 소프트웨어

- Docker + NVIDIA Container Toolkit
- ROS 2 Jazzy
- Python 3.10+
- Isaac-GR00T, LeRobot, cosmos-policy Docker 이미지

## 빠른 시작

### 1. 환경 설정

```bash
# OMX 환경 구축
bash scripts/week1_setup_omx_env.sh

# LeRobot 설치
bash scripts/week2_setup_lerobot.sh
```

### 2. GR00T 추론 테스트

```bash
python scripts/week3_test_groot_inference.py
```

### 3. 데이터 수집 및 변환

```bash
# OMX 데이터 수집 (50 에피소드 x 3 태스크)
bash scripts/week4_collect_omx_data.sh

# GR00T LeRobot v2 포맷으로 변환
python scripts/week4_convert_omx_to_groot.py
```

### 4. 학습 및 평가

```bash
# ACT 베이스라인 학습
bash scripts/week5_train_act_baseline.sh

# GR00T 파인튜닝
bash scripts/week5_finetune_groot_omx.sh

# 비교 평가
python scripts/week6_eval_omx.py
```

## 핵심 유틸리티

### `utils/omx_constants.py`

OMX 로봇의 공통 상수와 유틸리티를 제공합니다.

- **기본 상수**: `OMX_DOF=6`, `OMX_IMG_SIZE=224`, `OMX_CONTROL_HZ=100`
- **VLA 관절 이름**: `joint1`~`joint5`, `gripper` (GR00T N1.6에서 사용)
- **IDM 관절 이름**: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- **비디오 dtype 변환**: VLA(float32 [0,1]) <-> IDM(uint8 [0,255])

### `configs/omx_modality_config.py`

GR00T N1.6 파인튜닝을 위한 OMX 모달리티 설정입니다. `launch_finetune.py`의 `--modality-config-path` 플래그로 로드됩니다.

| 모달리티 | 키 | 설명 |
|----------|-----|------|
| video | `cam1` | 단일 카메라 (224x224) |
| state | `joint1`~`gripper` | 6차원 현재 상태 |
| action | `joint1`~`gripper` | 6차원, action_horizon=16 |
| language | `annotation.human.action.task_description` | 자연어 작업 지시 |

## IDM 시너지 파이프라인

DreamDojo가 생성한 합성 비디오를 GR00T IDM(Inverse Dynamics Model)으로 pseudo labeling하여 VLA 학습 데이터를 증강하는 파이프라인입니다.

```
DreamDojo 합성 비디오 -> IDM pseudo labeling -> VLA 학습 데이터 증강
```

**관절 이름 매핑** (VLA <-> IDM):

```
joint1 <-> shoulder_pan
joint2 <-> shoulder_lift
joint3 <-> elbow_flex
joint4 <-> wrist_flex
joint5 <-> wrist_roll
gripper <-> gripper
```

**비디오 dtype 차이**: VLA는 `float32 [0.0, 1.0]`, IDM은 `uint8 [0, 255]`를 사용합니다. 파이프라인 간 데이터 전달 시 `convert_video_vla_to_idm()` / `convert_video_idm_to_vla()` 함수로 변환해야 합니다.

## 참고 문서

- [Project.md](Project.md) - 상세 프로젝트 명세 (모델 분석, 아키텍처, 시너지 전략)
- [SCRIPTS.md](SCRIPTS.md) - 스크립트별 기술 문서 (의존성, 설계 결정, 사용법)
