# IDM 학습용 데이터셋 선정 및 변환

> SO-100 공개 데이터셋을 OMX 포맷으로 변환하여 Inverse Dynamics Model 학습 데이터를 준비하는 과정을 기술합니다.

---

## 1. 개요

GR00T-Dreams 파이프라인에서 Inverse Dynamics Model(IDM)은 비디오 프레임 쌍으로부터 로봇 action을 예측하는 핵심 모듈입니다. 이를 통해 라벨이 없는 인터넷 비디오에서도 pseudo action label을 생성할 수 있습니다.

NVIDIA는 GR00T-Dreams-IDM 학습 코드를 공개했으나 사전 학습된 가중치는 비공개입니다. 따라서 자체 데이터로 IDM을 학습해야 하며, 이를 위해 **관절 공간 action 라벨이 포함된 로봇 조작 데이터셋**이 필요합니다.

```
데이터 확보 → [OMX 변환] → IDM 학습 → pseudo labeling → 정책 학습
              ^^^^^^^^^^
              본 문서 범위
```

## 2. 데이터셋 선정 기준 및 사유

### IDM 입력 요구사항

IDM 학습에는 연속된 비디오 프레임 쌍 `(frame_t, frame_t+1)`과 해당 timestep의 관절 공간(joint-space) action 라벨이 필요합니다. `GR00T-Dreams-IDM`의 `LeRobotSingleDataset`은 LeRobot 포맷의 parquet + MP4 구조를 직접 로드합니다.

### 선정 기준

| # | 기준 | 사유 |
|---|------|------|
| 1 | **6-DOF joint-space action** | IDM은 관절 각도 변화를 예측; Cartesian delta는 부적합 |
| 2 | **30 FPS 이상** | 프레임 간 action 변화가 충분히 작아야 학습 안정 |
| 3 | **카메라 관측** (top + wrist) | IDM 입력은 이미지 프레임; 최소 1개 카메라 필요 |
| 4 | **LeRobot v2/v3 포맷** | 변환 파이프라인 호환; parquet + MP4 구조 |
| 5 | **Apache 2.0 라이선스** | 학습 및 재배포 제약 없음 |

### 후보군 비교

| 데이터셋 | 로봇 | 에피소드 | 프레임 | Action 형식 | FPS | 라이선스 |
|----------|------|---------|--------|------------|-----|---------|
| **svla_so100_stacking** | SO-100 | 56 | 22,956 | 6-DOF joint (deg) | 30 | Apache 2.0 |
| **svla_so100_pickplace** | SO-100 | 50 | 19,631 | 6-DOF joint (deg) | 30 | Apache 2.0 |
| **svla_so100_sorting** | SO-100 | 52 | 35,713 | 6-DOF joint (deg) | 30 | Apache 2.0 |
| community_dataset_v1 | SO-100 | ~700 | ~250K | 6-DOF joint (deg) | 30 | Apache 2.0 |
| 개별 커뮤니티 (misc) | 다양 | 5~30 | 다양 | 혼재 | 다양 | 다양 |

### 최종 선정

**1차 선정: SVLA 공식 3개 데이터셋** — 품질 우선 전략

- NVIDIA SVLA 팀이 직접 수집하여 데이터 품질이 검증됨
- 동일한 SO-100 로봇, 동일한 수집 환경, 일관된 포맷
- 3가지 태스크(stacking, pickplace, sorting)로 행동 다양성 확보

**향후 스케일업: community_dataset_v1** — 볼륨 확보

- ~700 에피소드로 10배 이상 데이터 증강 가능
- SVLA 기반 파이프라인 검증 후 추가 투입

### 비추천 데이터셋

- **Cartesian delta action**: end-effector 좌표 변화만 기록, 관절 각도 복원 불가
- **저해상도 (128x128)**: IDM 비전 백본(ViT)의 입력 해상도에 미달
- **소규모 (<5 에피소드)**: 학습에 통계적으로 불충분

## 3. 선정 데이터셋 상세

| 데이터셋 | 태스크 | 에피소드 | 프레임 | 크기 |
|----------|--------|---------|--------|------|
| svla_so100_stacking | 블록 쌓기 (정밀 조작) | 56 | 22,956 | ~1.2 GB |
| svla_so100_pickplace | 집어서 놓기 (기본 조작) | 50 | 19,631 | ~898 MB |
| svla_so100_sorting | 분류 (다중 물체) | 52 | 35,713 | ~1.6 GB |
| **합계** | | **158** | **78,300** | **~3.7 GB** |

**공통 사양**: 30 FPS, 듀얼 카메라 top + wrist (480x640), AV1 코덱 (yuv420p), 6-DOF joint (degrees)

### LeRobot v3 포맷 구조

각 데이터셋은 동일한 디렉토리 구조를 따릅니다:

```
svla_so100_{task}/
├── data/chunk-000/          # parquet 파일 (state, action, 메타)
├── videos/                  # MP4 비디오 (top, wrist 카메라별)
│   ├── observation.images.top/chunk-000/
│   └── observation.images.wrist/chunk-000/
├── meta/
│   ├── info.json            # 데이터셋 메타정보 (robot_type, fps, features)
│   ├── episodes.jsonl       # 에피소드별 정보
│   └── tasks.jsonl          # 태스크 설명
└── README.md
```

**Parquet 컬럼**: `observation.state[6]`, `action[6]`, `observation.images.top`, `observation.images.wrist`, `timestamp`, `frame_index`, `episode_index`

**관절 이름 (공통)**: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

## 4. SO-100 → OMX 변환 방법론

### 왜 변환이 필요한가

SO-100과 OMX는 kinematic chain이 다릅니다:

- **SO-100**: rpy 오프셋이 많은 5-DOF + gripper, 관절 값이 서보 degrees
- **OMX**: 깔끔한 구조의 5-DOF + gripper, 관절 값이 radians

IDM 학습 대상 로봇이 OMX이므로, SO-100 데이터를 OMX 관절 공간으로 재매핑해야 합니다.

### URDF 기반 FK/IK 변환 파이프라인

```
SO-100 parquet (degrees)
  │
  ▼ deg2rad (calibration offset 적용)
SO-100 joint angles (radians)
  │
  ▼ SO-100 Forward Kinematics (URDF chain)
4x4 End-Effector Transform (position + orientation)
  │
  ▼ OMX Inverse Kinematics (scipy L-BFGS-B, warm-start)
OMX joint angles (radians)
  │
  ▼ Write OMX LeRobot format
OMX parquet + 비디오 symlink
```

### 변환 상세

**1단계: 캘리브레이션 (deg → rad)**

SO-100 서보 값은 절대 degrees입니다. URDF 0도 기준으로 변환하기 위해 각 관절의 영점 오프셋을 빼고 라디안으로 변환합니다:

```python
# 영점 오프셋 (stats.json에서 추출한 중심값)
SO100_ZERO_OFFSET_DEG = {
    "shoulder_pan": 123.0, "shoulder_lift": 116.0,
    "elbow_flex": 98.0, "wrist_flex": 54.0, "wrist_roll": -52.0,
}
arm_rad[i] = deg2rad(servo_deg[i] - offset[i])
```

**2단계: SO-100 Forward Kinematics**

URDF에서 추출한 5개 관절의 DH 파라미터(origin_xyz, origin_rpy, axis)로 순기구학을 계산하여 end-effector의 4x4 동차 변환 행렬을 얻습니다.

**3단계: OMX Inverse Kinematics**

목표 EE 위치를 OMX 관절 각도로 역산합니다. `scipy.optimize.minimize`의 L-BFGS-B를 사용하며, 위치 오차에 높은 가중치(100)를, 방향 오차에 낮은 가중치(0.1)를 부여합니다.

**Warm-start 최적화**: 궤적의 연속성을 활용하여 이전 프레임의 IK 해를 다음 프레임의 초기값으로 사용합니다. 2mm 이내 수렴 시 다중 시작점 탐색을 건너뛰어 속도를 크게 향상시킵니다.

### URDF 소스

| 로봇 | URDF 경로 |
|------|----------|
| SO-100 | `~/claude/SO-ARM100/Simulation/SO100/so100.urdf` |
| OMX-F | `~/claude/open_manipulator/open_manipulator_description/urdf/omx_f/omx_f.urdf` |

### 실행 명령어

```bash
# FK/IK 자체 검증
python pseudo-project/scripts/convert_so100_to_omx.py --self-test

# 배치 변환 (3개 데이터셋)
for task in stacking pickplace sorting; do
    python pseudo-project/scripts/convert_so100_to_omx.py \
        --so100-dataset datasets/so100/svla_so100_${task} \
        --output-dir datasets/omx/omx_${task}
done
```

## 5. 변환 품질 검증

### Self-test 결과

FK → IK 라운드트립 테스트 (SO-100 목표 위치 → OMX IK → 복원 오차):

| 포즈 | 위치 오차 | 상태 |
|------|----------|------|
| 홈 (0,0,0,0,0) | 0.04 mm | PASS |
| 앞쪽 (0.3,0.5,-0.8,0.2,0.1) | 0.14 mm | PASS |
| 옆쪽 (-0.3,0.8,-1.0,-0.3,-0.5) | 0.51 mm | PASS |
| 위쪽 (1.0,0.3,-0.5,0.8,1.0) | 6.12 mm | PASS |

### 워크스페이스 도달 가능성

SO-100 실제 데이터 범위에서 랜덤 50 포즈를 샘플링한 결과, 도달 가능 비율(<10mm 오차)은 **78~86%** 입니다. 두 로봇의 워크스페이스 차이로 인해 극단적 포즈에서는 오차가 증가합니다.

### 실제 변환 결과

| 데이터셋 | 에피소드 | 평균 수렴률 | 평균 위치 오차 | 상태 |
|----------|---------|------------|--------------|------|
| omx_pickplace | 50 | 94% | 3.04 mm | 완료 |
| omx_stacking | 56 | — | — | 진행 중 |
| omx_sorting | 52 | — | — | 진행 중 |

> stacking, sorting은 변환 완료 후 본 테이블을 업데이트합니다.

## 6. 파일 경로 및 디렉토리

| 구분 | 경로 |
|------|------|
| 원본 (SO-100) | `datasets/so100/svla_so100_{stacking,pickplace,sorting}/` |
| 변환 결과 (OMX) | `datasets/omx/omx_{stacking,pickplace,sorting}/` |
| 변환 스크립트 | `pseudo-project/scripts/convert_so100_to_omx.py` |
| IDM 학습 | `GR00T-Dreams-IDM/scripts/idm_training.py` |
| 데이터 설정 | `GR00T-Dreams-IDM/gr00t/experiment/data_config_idm.py` |

> 모든 경로는 `/home/lambda/claude/` 기준 상대 경로입니다.

## 7. 다음 단계

1. **stacking, sorting 변환 완료** — 동일 파이프라인으로 나머지 2개 데이터셋 변환
2. **IDM 학습** — GR00T-Dreams-IDM Docker 환경에서 `idm_training.py` 실행
3. **multiprocessing 최적화** — IK 연산 병렬화로 변환 속도 개선 검토
4. **community_dataset_v1 스케일업** — SVLA 파이프라인 검증 후 ~700 에피소드 추가 변환
