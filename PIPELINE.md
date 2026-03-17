# GR00T-IDM 연계 및 Cosmos 데이터 파이프라인

> 관련 문서: [Project.md](Project.md) — 모델 분석 및 전략 | [CURRICULUM.md](CURRICULUM.md) — 12주 실행 로드맵

## 11. GR00T-IDM 연계 활용

> **참조 프로젝트**: [GR00T-Dreams-IDM](../GR00T-Dreams-IDM/) (IDM 전용 통합 레포)
>
> GR00T-IDM (Inverse Dynamics Model)은 비디오에서 로봇 action을 추출하는 모델입니다.
> GR00T-IDM-Documentation이 GR00T-Dreams-IDM에 통합되었으며, VLA 코드는 제거되고 IDM 전용으로 정리되었습니다.
> Cosmos Predict2.5 합성 비디오와 결합하면 VLA 학습 데이터를 크게 증강할 수 있습니다.

### VLA ↔ IDM 데이터 포맷 차이

| 항목 | GR00T N1.6 VLA | GR00T IDM |
|------|---------------|-----------|
| 비디오 dtype | `float32` [0.0, 1.0] | `uint8` [0, 255] |
| 비디오 해상도 | 224×224 | 256×256 |
| 관절 이름 | `joint1~5, gripper` (기능 무관) | `shoulder_pan, shoulder_lift, ...` (해부학적) |
| Action 표현 | joints=RELATIVE, gripper=ABSOLUTE | 모두 absolute |
| 설정 포맷 | Python (`register_modality_config`) | JSON (`modality.json`) |
| 입력 프레임 | 1 프레임 | 2 프레임 (t, t+1) |

### 관절 이름 매핑 테이블

| VLA (GR00T N1.6) | IDM (GR00T-Dreams) | 기능 |
|-------------------|-------------------|------|
| `joint1` | `shoulder_pan` | 어깨 수평 회전 |
| `joint2` | `shoulder_lift` | 어깨 수직 회전 |
| `joint3` | `elbow_flex` | 팔꿈치 굴곡 |
| `joint4` | `wrist_flex` | 손목 굴곡 |
| `joint5` | `wrist_roll` | 손목 회전 |
| `gripper` | `gripper` | 그리퍼 개폐 |

매핑 유틸리티: `utils/omx_constants.py`의 `OMX_JOINT_MAPPING` / `OMX_JOINT_MAPPING_INV`

### DreamDojo + IDM 시너지 파이프라인 (기존)

```
DreamDojo 합성 비디오 (uint8, 480×640)
    ↓ resize + 2-frame pairing
GR00T-IDM pseudo action labeling (uint8, 256×256)
    ↓ 관절 이름 매핑 (IDM→VLA)
품질 평가 (jerk, temporal consistency → grade ≥ B)
    ↓ dtype 변환 (uint8→float32)
GR00T N1.6 VLA 파인튜닝 데이터 증강
```

> **제약**: DreamDojo 후훈련에 8x H100 필요, 증류 파이프라인 미공개, OMX 미지원.
> 아래 Cosmos Predict2.5 파이프라인으로 대체 권장.

### Cosmos 합성 데이터 파이프라인 (v4 — 실증 반영)

> **실증 결과 (2025-03-14)**: Cosmos Predict2.5를 실제 실행하여 검증한 결과,
> **post-training 없이는 새 로봇 도메인에서 사용 불가**함을 확인했습니다.
> 상세 실험 로그: `.omc/research/cosmos-predict-experiment-log.md`
>
> - Base Video2World (샘플): 품질 낮음 (post-training 전 범용 모델 한계)
> - Action-Conditioned (Bridge 데이터): 정상 작동 (학습 도메인 일치)
> - Action-Conditioned (OMX 변환 데이터): **실패** — 노이즈, 전혀 다른 영상 (OOD)
> - 파라미터 튜닝 (guidance, scaler, steps): 근본적 개선 불가
>
> **핵심 교훈**: Post-training에는 실제 타겟 로봇 영상이 필요하나,
> 현재 보유 데이터는 SO-100/SO-101 영상(LeRobot)이지 실제 OMX가 아님.
> Cosmos Predict로 데이터를 "늘리려면" 먼저 데이터가 "있어야" 하는 순환 의존 발생.

이 파이프라인은 **실제 OMX 로봇 데이터 수집이 선행**되어야 실행 가능합니다.
OMX 로봇 없이 진행 가능한 대안으로 **Cosmos Transfer 2.5 시각 증강**을 아래에 추가합니다.

> **참고**: GR00T-Dreams 공식 파이프라인은 합성 비디오 생성 + 품질 필터링까지만
> 문서화되어 있으며, IDM을 통한 action label 추출은 본 프로젝트의 **독자적 확장**입니다.

> **중요**: Action-Conditioned 모델은 데이터 증강이 아닌 **정책 평가 도구**로 사용합니다.

```
Phase 0: 선행 검증 (필수)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① omx_fk.py rotation 수정 완료 (local frame 상대 변위)
② Cosmos Base 2B 후훈련 VRAM 테스트 (256×320, 45 frames)
③ Cosmos 합성 비디오 → IDM 정확도 벤치마크
   (합성 비디오의 IDM MAE가 허용 범위 내인지 확인)
④ action_scaler OMX 재보정 (Bridge 기본값 20.0 부적합)

Phase 1: 기본 학습
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사람이 OMX로 시연 50+ 회] (다양한 task 포함 권장)
    ↓ 카메라 영상 + joint1~5 + gripper 기록
    ↓
두 갈래로 활용 (병렬)
┌──────────────────────┬──────────────────────────┐
│ GR00T VLA 파인튜닝    │ 영상 + 텍스트 프롬프트     │
│ (로봇 두뇌 학습)      │ → Cosmos Base 2B 후훈련   │
│ → OMX 배포/검증       │ (OMX 세계 모델 학습)      │
└──────────────────────┴──────────────────────────┘

Phase 2A: 합성 데이터 증강 (Cosmos Base Video2World)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
텍스트 프롬프트 다양화 + 초기 프레임 + seed/guidance 변형
    ↓
Cosmos Base Video2World 추론 (image2world 모드)
    ↓ 합성 비디오 (수백 회, action label 없음)
Cosmos Reason (1-7B 또는 2-2B) → 물리적 타당성 필터 (≥ 4.0)
    ↓
[IDM 정확도 OK] GR00T IDM → pseudo action label → VLA 학습 데이터
[IDM 정확도 부족] 합성 비디오를 시각 사전학습에만 활용 (fallback)

Phase 2B: 정책 평가 도구 (Cosmos Action-Conditioned)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(선행: omx_fk.py 수정 + action_scaler 재보정 완료)
FK 변환 → EE 포즈 → Action-Conditioned 2B 후훈련
    ↓
VLA가 예측한 action → Cosmos 시각화 → 실 로봇 배포 전 검증

Phase 3: 혼합 재학습
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실제 데이터 (50+ 회) + 합성 데이터 (IDM labeled 또는 시각 only)
    + torchvision 시각 증강 (ColorJitter, RandomAffine, GaussianBlur)
    ↓
GR00T VLA 재파인튜닝 → 성능 향상된 로봇 두뇌
    ↓
OMX 최종 배포/검증
```

**각 모델의 역할 (수정됨):**

| 모델 | 역할 | Cosmos 변형 | GPU 요구 |
|------|------|------------|---------|
| GR00T N1.6 VLA | 카메라를 보고 행동 결정 (두뇌) | - | 1x RTX 4090+ |
| Cosmos Base Video2World | 텍스트→미래 비디오 생성 (**데이터 증강**) | `Cosmos-Predict2.5-2B` | 1x GPU (2B) |
| Cosmos Action-Cond | action→비디오 시각화 (**정책 평가**) | `.../robot/action-cond` | 1x GPU (2B) |
| GR00T IDM | action-free 비디오 → pseudo action label | - | 1x GPU |
| Cosmos Reason | 합성 비디오 물리적 타당성 점수 | `Cosmos-Reason1-7B` 또는 `Reason2-2B` | 1x GPU |

**데이터 포맷:**

| 용도 | 포맷 | 필요 데이터 |
|------|------|-----------|
| Base 후훈련 (Phase 2A) | `videos/*.mp4` + `metas/*.txt` | 영상 + 텍스트 프롬프트 |
| Action-Cond 후훈련 (Phase 2B) | `videos/*.mp4` + `annotations/*.json` | 영상 + EE state/action |
| IDM pseudo label | 2-frame pairing | 합성 비디오 프레임 쌍 |

**OMX용 Cosmos 입력 데이터 변환 (`utils/omx_fk.py`):**

```python
from utils.omx_fk import OMXForwardKinematics

fk = OMXForwardKinematics(robot="omx_f")

# OMX-F URDF kinematic chain (Z, Y, Y, Y, X):
#   joint1~5 → FK → EE 포즈 (x, y, z, roll, pitch, yaw)
#   gripper → 그대로 continuous_gripper_state로 사용
# 홈 포지션 EE: (0.313m, -0.002m, 0.211m)

# 단일 프레임 변환 (state 생성)
cosmos_state = fk.to_cosmos_state([0.0, -0.5, 0.3, 0.2, 0.0], gripper=0.5)
# → {"state": [x, y, z, roll, pitch, yaw], "continuous_gripper_state": 0.5}

# Cosmos action 계산 (local frame 상대 변위, Cosmos convention)
# 참조: cosmos_predict2/action_conditioned.py:82-103 (_get_actions)
action = fk.compute_cosmos_action(joints_t, joints_t1, grip_t, grip_t1)
# → [rel_xyz(3), rel_rpy(3), gripper(1)] in previous EE local frame

# 궤적 전체 변환 (annotations/*.json 생성용, Phase 2B)
batch = fk.batch_to_cosmos_states(joint_trajectory, gripper_trajectory)
# → states: (T, 6), actions: (T-1, 7), continuous_gripper_states: (T,)
```

> **주의**: `action_scaler`(기본값 20.0)는 Bridge 로봇 EE 변위에 맞춰져 있음.
> OMX EE 변위 범위를 측정하여 적절한 값으로 재보정 필요. (실측: OMX ds=6 기준 30.6 적정)

> **참고**: Cosmos는 annotation JSON의 `state`와 `continuous_gripper_state`에서 action을
> 내부적으로 재계산합니다 (`get_action_sequence_from_states()`→`_get_actions()`).
> 사전에 `action` 필드를 계산해서 넣을 필요 없습니다.

**위험 요소 및 완화:**

| 위험 | 영향 | 완화 | 실증 상태 |
|------|------|------|----------|
| Post-training 없이 품질 부족 | 증강 데이터 무용 | 실제 OMX 데이터 수집 선행 | **확인됨** (노이즈, OOD) |
| IDM 합성 비디오 정확도 미검증 | 파이프라인 전체 | Phase 0에서 사전 벤치마크, fallback 경로 포함 | 미검증 |
| 순환 의존 (데이터↔post-training) | 파이프라인 착수 불가 | 50+ OMX 시연 선수집, Transfer 대안 검토 | **확인됨** |
| Base 후훈련 VRAM (432×768) | OOM | 256×320으로 시작 (inference 실측 ~20GB) | 부분 확인 |
| action_scaler Bridge 전용 | 부정확한 action | OMX 재보정 필요 (실측: 30.6 적정) | **실측 완료** |

**실측 기준 수치 (2025-03-14):**

| 항목 | 값 |
|------|-----|
| Predict 2B inference VRAM | 19.80~19.95 GB |
| Bridge action_scaler | 20.0 (action xyz mean 0.0094) |
| OMX action_scaler (ds=6) | 30.6 (action xyz mean 0.0063) |
| Base V2W 생성 시간 (93fr) | ~121초 (36 denoising steps) |
| Action-Cond 생성 시간 (169fr) | ~33초 (13 chunks) |
| Post-training 최소 데이터 | 50~100 에피소드 (실제 로봇 영상) |

상세 구현: `scripts/week10_cross_model_analysis.py`
Cosmos Predict2.5 소스: `~/cosmos-predict2.5/`
Cosmos Cookbook: https://nvidia-cosmos.github.io/cosmos-cookbook/
실험 로그: `.omc/research/cosmos-predict-experiment-log.md`

### Cosmos Transfer 2.5 시각 증강 (대안 파이프라인)

Cosmos Predict의 순환 의존 문제를 우회하는 대안입니다.
**기존 시연 영상의 시각적 외관(조명, 재질, 배경)만 변환**하고,
action/state는 원본을 그대로 유지합니다.

> **핵심 차이**: Predict는 "새 비디오 생성" → post-training 필수.
> Transfer는 "기존 비디오 외관 변환" → 범용 모델로 바로 적용 가능.
> action/state를 건드리지 않으므로 정합성 문제 없음.

```
입력: 원본 시연 비디오 + seg/edge/depth 제어 맵
          ↓
Cosmos Transfer 2.5 2B (edge/seg/depth/blur)
          ↓ 외관만 변환된 비디오 (조명, 재질, 배경 톤)
          ↓ action/state/timestamp 원본 유지
자동 검수 (seg IoU, centroid drift, flicker)
          ↓ 통과본만
원본 + 증강 혼합 학습 (비율 1:0.25 → 최대 1:0.5)
```

**적용 원칙:**
- RGB 영상만 증강, action/state 절대 수정 금지
- seg/edge/depth 제어를 반드시 사용 (RGB 단독 증강은 구조 붕괴 위험)
- clip 단위 2~6초로 분할 후 증강 (drift 누적 방지)
- 증강 강도는 보수적으로 (조명/재질/배경 톤 수준)
- 물체 shape, 로봇 geometry, object identity 변경 금지
- 자동 검수 통과본만 사용, 통과율 낮으면 대량 생성 금지

**미확인 사항 (실측 필요):**

| 항목 | 우려 | 확인 방법 |
|------|------|----------|
| Transfer 2B VRAM | ControlNet 추가로 28~32GB 예상, OOM 위험 | SO-100 clip 1개 실측 |
| SO-100 로봇 구조 보존 | 학습 분포 밖 로봇에 대한 품질 미지수 | edge + seg 제어 테스트 |
| 증강 유효성 | 시각 증강이 실제 정책 성능을 개선하는지 | baseline 대비 비교 실험 |

**커뮤니티 보고 시행착오:**
- 조명만 바꿔도 geometry drift 발생 가능
- 효과 강도 ↔ 구조 보존은 trade-off (약한 변화 여러 개가 안전)
- seg mask 없이 증강하면 annotation 일관성 저하
- 새 로봇 embodiment에 대한 기대치를 낮춰야 함

참고: Cosmos Transfer 증강 운영 문서는 별도 작성 예정

### 비디오 dtype 변환

```python
from utils.omx_constants import convert_video_vla_to_idm, convert_video_idm_to_vla

# VLA float32 [0,1] → IDM uint8 [0,255]
idm_video = convert_video_vla_to_idm(vla_video)

# IDM uint8 [0,255] → VLA float32 [0,1]
vla_video = convert_video_idm_to_vla(idm_video)
```

