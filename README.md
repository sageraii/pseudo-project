# NVIDIA 로봇 AI 모델 벤치마킹 on ROBOTIS OMX

GR00T N1.6, Cosmos Policy, Cosmos Predict2.5를 ROBOTIS OMX 6-DOF 매니퓰레이터에서 비교 평가하고, Cosmos Predict2.5 + IDM 시너지 파이프라인으로 합성 데이터 증강까지 수행하는 12주 연구 프로젝트입니다.

## 프로젝트 개요

Option C+ 하이브리드 접근법을 채택하여, ACT 베이스라인 및 GR00T 파인튜닝을 수행하고, Cosmos Predict2.5 세계 모델과 IDM 시너지 파이프라인을 통해 합성 데이터 증강까지 달성합니다.

| 모델 | 유형 | 역할 |
|------|------|------|
| ACT | Imitation Learning Policy | 베이스라인 (OMX 학습 + 배포) |
| GR00T N1.6 | Vision-Language-Action (VLA) | 카메라+언어 입력으로 직접 로봇 행동 생성 (두뇌) |
| Cosmos Predict2.5 | World Model (Action-Conditioned) | 행동에 따른 미래 비디오 예측 (상상력) |
| GR00T IDM | Inverse Dynamics Model | 비디오에서 행동 역추정 (관찰자) |
| Cosmos Policy | Video-to-Policy | 비디오 모델 기반 로봇 제어 정책 (비교 분석) |

## 모델 비교

| 항목 | ACT | GR00T N1.6 | Cosmos Predict2.5 | GR00T IDM | Cosmos Policy |
|------|-----|-----------|-------------------|-----------|--------------|
| 역할 | 베이스라인 | 행동 결정 | 미래 비디오 생성 | pseudo labeling | 비교 분석 |
| 입력 | 카메라 + 상태 | 카메라 + 언어 + 상태 | 초기 프레임 + 행동(EE) | 비디오 2프레임 쌍 | 관찰 + 시연 |
| 출력 | 로봇 행동 (6-dim) | 로봇 행동 (6-dim) | 합성 비디오 | 행동 pseudo label | 행동 + 미래 + 가치 |
| GPU 요구 | CPU 가능 | 1x RTX 4090 | 1x RTX 4090 | 1x RTX 4090 | 1x (추론만) |
| 추론 속도 | <10ms | ~44ms | 비실시간 | ~20ms | 측정필요 |
| 언어 이해 | 불가 | 지원 | 불가 | 불가 | 제한적 |
| 본 프로젝트 | 학습 + 배포 | 파인튜닝 + 배포 | 후훈련 + 추론 | 추론 | 추론 전용 |

## 디렉토리 구조

```
pseudo-project/
├── configs/
│   └── omx_modality_config.py       # GR00T N1.6 파인튜닝용 모달리티 설정
├── utils/
│   ├── __init__.py
│   ├── omx_constants.py             # OMX 상수, URDF kinematic 파라미터, 관절 매핑, dtype 변환
│   └── omx_fk.py                    # URDF 기반 FK (joint → EE pose → Cosmos state/action)
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
│   ├── week9_run_dreamdojo_rollout.py  # DreamDojo 롤아웃 생성 + 분석 (참고)
│   ├── week10_cross_model_analysis.py  # 통합 파이프라인 + Cosmos Predict2.5-IDM 시너지
│   └── week11_benchmark_report.py      # 종합 벤치마킹 보고서 생성
├── Project.md                       # 상세 프로젝트 명세
├── SCRIPTS.md                       # 스크립트별 기술 문서
└── README.md
```

## 주차별 실행 파이프라인

| 주차 | 단계 | 핵심 산출물 |
|------|------|------------|
| 1-2 | 환경 구축 (Docker, LeRobot, OMX) | OMX 하드웨어 동작 검증 |
| 3 | GR00T 추론 검증 | Isaac-GR00T 추론 파이프라인 |
| 4 | 데이터 수집 및 변환 | 50 에피소드 x 3 태스크 (LeRobot v2) |
| 5 | ACT 베이스라인 + GR00T 파인튜닝 | 두 모델 체크포인트 |
| 6 | 배포 및 평가 (ACT vs GR00T) | 성공률/속도 비교 보고서 |
| 7-8 | **트랙A**: Cosmos Policy LIBERO 평가 | LIBERO 벤치마크 결과 |
| | **트랙B**: Cosmos Predict2.5 환경 + FK 변환 | `omx_fk.py`, Cosmos 데이터 변환 |
| 9 | **트랙A**: Cosmos Predict2.5 OMX 후훈련 | OMX 전용 세계 모델 + 합성 비디오 |
| | **트랙B**: GR00T IDM pseudo labeling | 품질 필터링된 pseudo label |
| 10 | 통합 파이프라인 (Cosmos Predict2.5-IDM 증강 재학습) | 증강 전/후 성능 비교 |
| 11-12 | 최종 벤치마크 + 프로젝트 마무리 | 종합 분석 보고서 |

## 사전 요구사항

### 하드웨어

- ROBOTIS OMX 매니퓰레이터 (5-DOF + 1 Gripper)
- NVIDIA GPU (RTX 4090 이상 권장)
- USB 카메라 (224x224 해상도)

### 소프트웨어

- Docker + NVIDIA Container Toolkit
- ROS 2 Jazzy
- Python 3.10+
- Isaac-GR00T, LeRobot, Cosmos Predict2.5, cosmos-policy Docker 이미지

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

# ACT vs GR00T 비교 평가
python scripts/week6_eval_omx.py
```

### 5. Cosmos 평가 및 시너지 파이프라인

```bash
# Cosmos Policy LIBERO 벤치마크
python scripts/week7_eval_cosmos_libero.py

# Cosmos vs GR00T 비교 분석
python scripts/week8_cosmos_groot_comparison.py

# 통합 파이프라인 + Cosmos Predict2.5-IDM 시너지
python scripts/week10_cross_model_analysis.py \
    --eval-dir outputs/eval \
    --cosmos-predict-dir outputs/cosmos_predict

# 종합 벤치마킹 보고서 생성
python scripts/week11_benchmark_report.py \
    --eval-dir outputs/eval \
    --synergy-dir outputs/week10_analysis \
    --output outputs/final_report.md
```

## 핵심 유틸리티

### `utils/omx_constants.py`

OMX 로봇의 공통 상수, URDF kinematic 파라미터, 유틸리티를 제공합니다.

- **기본 상수**: `OMX_DOF=6`, `OMX_IMG_SIZE=224`, `OMX_CONTROL_HZ=100`
- **URDF kinematic 파라미터**: OMX-F/OMX-L 관절 오프셋, 축(Z,Y,Y,Y,X), EE 오프셋, 그리퍼 설정
- **VLA/IDM 관절 매핑**: `joint1`~`gripper` <-> `shoulder_pan`~`gripper`
- **비디오 dtype 변환**: VLA(float32 [0,1]) <-> IDM(uint8 [0,255])

### `utils/omx_fk.py`

URDF 기반 Forward Kinematics 유틸리티입니다. OMX joint positions를 Cosmos Predict2.5가 요구하는 EE pose로 변환합니다.

```python
from utils.omx_fk import OMXForwardKinematics

fk = OMXForwardKinematics(robot="omx_f")

# joint → EE pose → Cosmos state
cosmos_state = fk.to_cosmos_state([0.0, -0.5, 0.3, 0.2, 0.0], gripper=0.5)
# → {"state": [x, y, z, roll, pitch, yaw], "continuous_gripper_state": 0.5}

# 궤적 일괄 변환
batch = fk.batch_to_cosmos_states(joint_trajectory, gripper_trajectory)
# → states: (T, 6), actions: (T-1, 7), continuous_gripper_states: (T,)
```

### `configs/omx_modality_config.py`

GR00T N1.6 파인튜닝을 위한 OMX 모달리티 설정입니다. `launch_finetune.py`의 `--modality-config-path` 플래그로 로드됩니다.

| 모달리티 | 키 | 설명 |
|----------|-----|------|
| video | `cam1` | 단일 카메라 (224x224) |
| state | `joint1`~`gripper` | 6차원 현재 상태 |
| action | `joint1`~`gripper` | 6차원, action_horizon=16 |
| language | `annotation.human.action.task_description` | 자연어 작업 지시 |

## Cosmos 데이터 증강 파이프라인

### 방안 A: Cosmos Predict2.5 + IDM (OMX 로봇 데이터 필요)

> **실증 결과**: Post-training 없이는 새 로봇에 사용 불가 확인됨.
> 실제 OMX 로봇 데이터 50~100 에피소드 선수집 필요. (실험 로그: `.omc/research/`)

```
Phase 1: OMX 실제 시연 (50+회) → VLA 파인튜닝 + Cosmos Base 후훈련
Phase 2: 텍스트 프롬프트 → Cosmos 합성 비디오 → IDM pseudo label → 품질 필터
Phase 3: 실제 + 합성 → VLA 재학습
```

### 방안 B: Cosmos Transfer 2.5 시각 증강 (기존 데이터 활용 가능)

기존 시연 영상의 **시각적 외관만 변환** (조명, 재질, 배경), action/state 원본 유지.

```
원본 비디오 + seg/edge/depth → Cosmos Transfer 2B → 외관 변환 → 자동 검수 → 혼합 학습
```

**데이터 변환 흐름**:

| 단계 | 변환 | 유틸리티 |
|------|------|---------|
| OMX joint → EE pose | FK (Z,Y,Y,Y,X) | `utils/omx_fk.py` |
| IDM 관절 → VLA 관절 | shoulder_pan → joint1 등 | `OMX_JOINT_MAPPING_INV` |
| IDM 비디오 → VLA 비디오 | uint8 [0,255] → float32 [0,1] | `convert_video_idm_to_vla()` |

**관절 이름 매핑** (VLA <-> IDM):

```
joint1 <-> shoulder_pan    (베이스 수평 회전, Z축)
joint2 <-> shoulder_lift   (어깨 수직 회전, Y축)
joint3 <-> elbow_flex      (팔꿈치 굴곡, Y축)
joint4 <-> wrist_flex      (손목 굴곡, Y축)
joint5 <-> wrist_roll      (손목 회전, X축)
gripper <-> gripper
```

## 참고 문서

- [Project.md](Project.md) - 모델 분석 및 전략 (§1-7: 모델 정의, 비교, 시너지, 결론)
- [CURRICULUM.md](CURRICULUM.md) - 12주 실행 로드맵 (§8-10: 커리큘럼, 팁, 예상 결과)
- [PIPELINE.md](PIPELINE.md) - IDM/Cosmos 데이터 파이프라인 (§11: GR00T-IDM 연계, Cosmos v4, Transfer 2.5)
- [SCRIPTS.md](SCRIPTS.md) - 스크립트 기술 문서 (의존성, 설계 결정, 사용법)
