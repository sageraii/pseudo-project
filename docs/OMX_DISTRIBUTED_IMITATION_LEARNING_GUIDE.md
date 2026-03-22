# OMX 분산 모방학습 가이드 (3인 협업)

> 3명이 각자의 공간에서 동일한 태스크의 모방학습 데이터를 수집하고, 통합하여 학습하는 방법

## 1. 개요

모방학습(Imitation Learning)은 사람이 리더 장치로 시범 동작을 보여주면, 로봇이 이를 학습하여 재현하는 방식입니다. 데이터의 양과 다양성이 학습 성능에 직결되므로, 여러 명이 분담하여 데이터를 수집하면 효율적입니다.

본 가이드는 ROBOTIS OMX 매니퓰레이터 + Physical AI Tools + LeRobot 환경을 기준으로 합니다.

## 2. 각 사람이 필요한 장비

| 장비 | 용도 | 비고 |
|------|------|------|
| OMX-F (팔로워) | 실제 작업 수행 로봇 | Dynamixel ID 11~16 |
| OMX-L (리더) | 사람이 조작하는 입력 장치 | Dynamixel ID 1~6 |
| USB 카메라 | 영상 데이터 수집 | UVC 호환 필수 |
| Linux PC | Docker 실행, 데이터 저장 | GPU 불필요 (녹화만 할 경우) |

### 카메라 요구사양

| 항목 | 권장값 |
|------|--------|
| 해상도 | 640x480 (녹화), 1920x1080 (추론 시) |
| FPS | 30fps |
| 인터페이스 | USB, UVC 호환 |
| 포커스 | 고정 초점 권장 (오토포커스는 녹화 중 불안정) |

## 3. 사전 환경 통일

3명의 환경이 다르면 학습 데이터의 일관성이 떨어집니다. 다음 항목을 반드시 통일해야 합니다.

### 3.1 소프트웨어 버전

```bash
# 모든 참여자가 동일한 버전을 사용
open_manipulator: main 브랜치 (v4.1.2 이상)  # jazzy는 그리퍼 반전 버그 있음
physical_ai_tools: jazzy 브랜치
```

### 3.2 카메라 설정

```yaml
# physical_ai_server/config/omx_f_config.yaml
# 3명 모두 동일한 카메라 토픽명, 해상도, FPS 사용
camera_topic_list:
  - camera1:/camera1/image_raw/compressed

# LeRobot 직접 사용 시
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

### 3.3 태스크 정의

```
태스크 이름: 정확히 동일한 문자열 사용 (예: "Pick up the Dynamixel Motor")
시작 조건: 로봇 팔이 초기 위치에 있을 때
종료 조건: 물체를 목표 위치에 놓았을 때
```

### 3.4 작업 환경 가이드라인

| 항목 | 권장 |
|------|------|
| 카메라 위치 | 정면 약 45도 각도, 작업대 전체가 보이는 위치 |
| 조명 | 균일한 실내 조명, 역광 방지 |
| 배경 | 단순한 배경 (복잡한 패턴 피하기) |
| 물체 | 동일한 대상 물체 사용 |

## 4. 데이터 수집

### 4.1 방법 A: Physical AI Tools 웹 UI 사용

Docker 환경에서 Physical AI Manager 웹 UI를 통해 녹화합니다.

```bash
# 1. open_manipulator Docker 시작
cd ~/workspace/open_manipulator/docker
./container.sh start
./container.sh enter
ros2 launch open_manipulator_bringup omx_ai.launch.py

# 2. physical_ai_tools Docker 시작 (별도 터미널)
cd ~/workspace/physical_ai_tools/docker
./container.sh start
./container.sh enter
ros2 launch physical_ai_server physical_ai_server_bringup.launch.py

# 3. 웹 브라우저에서 Physical AI Manager 접속
# http://localhost (기본 포트)
```

웹 UI에서:
1. 로봇 타입 "omx_f" 선택
2. Record 페이지에서 태스크 이름 입력
3. 녹화 시작 -> 리더로 시범 동작 -> 녹화 종료
4. 여러 에피소드 반복

### 4.2 방법 B: LeRobot CLI 직접 사용

```bash
lerobot-record \
    --robot.type=omx_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=omx_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=omx_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=omx_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/pick_up_motor \
    --dataset.num_episodes=30 \
    --dataset.single_task="Pick up the Dynamixel Motor"
```

**녹화 중 키보드 조작:**
- 오른쪽 화살표: 현재 에피소드 종료, 다음 에피소드로 이동
- 왼쪽 화살표: 현재 에피소드 폐기, 재녹화
- Esc: 세션 종료 및 데이터셋 저장

**중단 후 재개:**
```bash
# --resume=true 플래그로 이전 녹화 세션 이어서 진행
lerobot-record \
    --dataset.repo_id=${HF_USER}/pick_up_motor \
    --resume=true \
    ...
```

## 5. 데이터 통합

3명이 각자 수집한 데이터를 하나의 데이터셋으로 합쳐야 학습에 사용할 수 있습니다.

### 5.1 방법 A: HuggingFace Hub 활용 (권장)

HuggingFace Hub를 중앙 저장소로 사용하는 방법입니다.

**주의:** Physical AI Tools의 HuggingFace 업로드는 사용자별 repo (`user_id/repo_name`)를 생성합니다. 여러 사용자가 동일한 repo에 직접 push하는 기능은 기본적으로 지원되지 않습니다.

따라서 다음과 같은 워크플로우를 권장합니다:

```
[수집 단계]
사람 A -> HF 업로드 -> userA/pick_up_motor  (에피소드 30개)
사람 B -> HF 업로드 -> userB/pick_up_motor  (에피소드 30개)
사람 C -> HF 업로드 -> userC/pick_up_motor  (에피소드 30개)

[통합 단계 - 학습 담당자가 수행]
학습 PC에서 3개 데이터셋 다운로드
  -> 에피소드 인덱스 재정렬하여 하나의 데이터셋으로 병합
  -> 병합된 데이터셋을 team/pick_up_motor_merged 로 업로드
```

**HuggingFace 인증 설정:**
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER  # 본인의 HF 사용자명 확인
```

**업로드 (LeRobot CLI):**
```bash
# 녹화 시 자동 업로드 (기본 동작)
# 자동 업로드 비활성화 시:
--dataset.push_to_hub=false

# 수동 업로드:
huggingface-cli upload ${HF_USER}/pick_up_motor \
  ~/.cache/huggingface/lerobot/${HF_USER}/pick_up_motor \
  --repo-type dataset
```

**다운로드:**
```bash
huggingface-cli download ${HF_USER}/pick_up_motor \
  --repo-type dataset \
  --local-dir ~/.cache/huggingface/lerobot/${HF_USER}/pick_up_motor
```

### 5.2 방법 B: HuggingFace Organization 활용

HuggingFace Organization을 생성하면 여러 사용자가 동일한 namespace 아래 데이터셋을 관리할 수 있습니다.

```bash
# Organization: my-robotics-team
# 모든 참여자가 동일한 organization repo에 접근
--dataset.repo_id=my-robotics-team/pick_up_motor
```

1. https://huggingface.co/organizations/new 에서 Organization 생성
2. 3명의 HuggingFace 계정을 Organization에 초대
3. 각 참여자가 동일한 repo_id로 녹화하면 에피소드가 누적됨

### 5.3 방법 C: 로컬 파일 수동 병합

네트워크 환경이 제한적인 경우, USB 등으로 데이터를 직접 복사하여 병합합니다.

```
[각 사람의 로컬 데이터 구조 - LeRobot v2.1 포맷]
~/.cache/huggingface/lerobot/{user}/pick_up_motor/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
├── videos/chunk-000/observation.images.camera1/
│   ├── episode_000000.mp4
│   └── ...
└── meta/
    ├── info.json          # 데이터셋 전체 정보
    ├── episodes.jsonl     # 에피소드별 메타데이터
    ├── tasks.jsonl        # 태스크 정의
    └── episodes_stats.jsonl  # 에피소드별 통계
```

**병합 시 주의사항:**
- 에피소드 인덱스를 연속으로 재정렬해야 함 (A: 0~29, B: 30~59, C: 60~89)
- parquet 파일명과 mp4 파일명도 에피소드 인덱스에 맞게 변경
- `meta/info.json`의 `total_episodes`, `total_frames` 갱신
- `meta/episodes.jsonl`에 모든 에피소드 메타데이터 병합
- `meta/episodes_stats.jsonl`에 통계 정보 병합

> **참고:** LeRobot 라이브러리 내부적으로 `concatenate_datasets()` (HuggingFace datasets 라이브러리)를 사용하여 parquet 데이터를 결합합니다. 별도의 "데이터셋 병합" API는 제공되지 않으므로, 수동 병합보다는 방법 A 또는 B를 권장합니다.

## 6. 학습

통합된 데이터셋으로 정책 모델을 학습합니다. GPU가 있는 서버 1대에서 수행합니다.

### 6.1 지원되는 정책

Physical AI Tools의 TrainingManager에서 지원하는 정책 목록 (총 7종):

| 정책 | 설명 | 특징 |
|------|------|------|
| `act` | Action Chunking with Transformers | 범용, 안정적 |
| `diffusion` | Diffusion Policy | 복잡한 동작에 강함 |
| `pi0` | Pi-Zero (Vision-Language) | 멀티모달 |
| `pi0fast` | Pi-Zero Fast | pi0의 경량 버전 |
| `smolvla` | Small Vision-Language-Action | 경량 VLA 모델 |
| `tdmpc` | Temporal Difference MPC | 모델 기반 |
| `vqbet` | VQ-BeT | 이산 행동 표현 |

> **참고:** LeRobot 코드베이스에는 `sac` (Soft Actor-Critic) 정책도 존재하나, Physical AI Tools의 TrainingManager에서는 지원하지 않습니다.

### 6.2 학습 명령

```bash
lerobot-train \
  --dataset.repo_id=my-robotics-team/pick_up_motor \
  --policy.type=act \
  --output_dir=outputs/train/omx_act_policy \
  --job_name=act_pick_up_motor \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=1000 \
  --wandb.enable=true
```

### 6.3 주요 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--policy.type` | (필수) | 정책 종류 (위 표 참고) |
| `--dataset.repo_id` | (필수) | HF 데이터셋 repo ID |
| `--policy.device` | cuda | 학습 디바이스 (cuda, cpu, mps) |
| `--batch_size` | 8 | 배치 크기 |
| `--steps` | 100000 | 총 학습 스텝 수 |
| `--seed` | 1000 | 랜덤 시드 |
| `--num_workers` | 4 | 데이터 로딩 워커 수 |
| `--save_freq` | 1000 | 체크포인트 저장 간격 (스텝) |
| `--eval_freq` | 20000 | 평가 간격 (스텝) |
| `--log_freq` | 200 | 로그 출력 간격 (스텝) |
| `--output_dir` | (필수) | 학습 결과 저장 경로 |

### 6.4 학습 재개

```bash
lerobot-train \
  --config_path=outputs/train/omx_act_policy/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 7. 추론 (Inference)

학습된 모델로 로봇을 자율 동작시킵니다.

### 7.1 LeRobot CLI 사용

```bash
python -m lerobot.record \
  --robot.type=omx_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --robot.id=omx_follower_arm \
  --dataset.repo_id=${HF_USER}/eval_act_omx \
  --dataset.single_task="Pick up the Dynamixel Motor" \
  --policy.path=${HF_USER}/omx_act_policy
```

### 7.2 Physical AI Tools 웹 UI 사용

Physical AI Manager의 Inference 페이지에서:
1. 학습된 정책 모델 선택
2. FPS 설정
3. 추론 시작

## 8. 데이터 품질 체크리스트

### 녹화 전

- [ ] 3명의 SW 버전이 동일한가?
- [ ] 카메라 해상도/FPS가 동일한가?
- [ ] 태스크 이름이 정확히 동일한가?
- [ ] 카메라 앵글 가이드를 공유했는가?
- [ ] 대상 물체가 동일한가?

### 녹화 중

- [ ] 시범 동작이 너무 빠르거나 느리지 않은가?
- [ ] 카메라 화면에 작업 전체 과정이 보이는가?
- [ ] 실패한 에피소드를 바로 폐기했는가? (왼쪽 화살표 키)
- [ ] 각 에피소드의 시작/종료 조건이 일관적인가?

### 녹화 후

- [ ] 에피소드 수가 목표치에 도달했는가?
- [ ] 데이터셋의 info.json에서 total_episodes, fps 확인
- [ ] 영상 파일이 정상 재생되는가?

## 9. 이전 분석에서의 수정 사항

본 리포트 작성 시 코드 기반 검증을 통해 다음 사항을 수정했습니다:

| 항목 | 이전 답변 | 검증 후 수정 |
|------|----------|-------------|
| 다수 사용자 HF 업로드 | "같은 repo에 에피소드 추가 가능" | 사용자별 별도 repo 생성됨. 동일 repo 공유는 Organization 필요 |
| 데이터셋 병합 API | 별도 언급 없음 | LeRobot에 명시적 병합 API 없음. concatenate_datasets()로 내부 결합만 지원 |
| 권장 에피소드 수 | "ACT 최소 50, 권장 100~200" | ROBOTIS 공식 문서에서는 기본 50 에피소드, 테스트용 5 에피소드만 언급. 정책별 최소 에피소드 수는 공식 문서에 명시되어 있지 않음 |
| sac 정책 | 언급 없음 | LeRobot에 존재하나 TrainingManager에서 미지원 |

## 10. 참고 자료

| 자료 | 링크 |
|------|------|
| ROBOTIS OMX 공식 문서 | https://ai.robotis.com/omx/introduction_omx.html |
| OMX 모방학습 가이드 (LeRobot) | https://ai.robotis.com/omx/lerobot_imitation_learning_omx |
| OMX 모방학습 가이드 (Physical AI Tools) | https://ai.robotis.com/omx/imitation_learning_omx.html |
| Physical AI Tools 설정 | https://ai.robotis.com/omx/setup_guide_physical_ai_tools.html |
| LeRobot 카메라 문서 | https://huggingface.co/docs/lerobot/en/cameras |
| LeRobot GitHub | https://github.com/huggingface/lerobot |
| HuggingFace ROBOTIS 모델 | https://huggingface.co/ROBOTIS |

---

**작성일:** 2026-03-22
**환경:** Physical AI Tools (jazzy), open_manipulator (main v4.1.2), LeRobot v2.1
