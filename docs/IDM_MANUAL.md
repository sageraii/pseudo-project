# GR00T-Dreams-IDM 매뉴얼

> **IDM (Inverse Dynamics Model)**: 비디오 프레임 2장으로부터 로봇 Action을 예측하는 모델 (612M 파라미터)
>
> 환경: RTX 5090 32GB, Python 3.12, PyTorch nightly+cu128
> 작성일: 2026-03-03 | 최종수정: 2026-03-03

---

## 목차

1. [개요 - IDM이란?](#1-개요---idm이란)
2. [환경 설정](#2-환경-설정)
3. [추론에 필요한 데이터](#3-추론에-필요한-데이터)
4. [데이터셋 구성 방법](#4-데이터셋-구성-방법-lerobot-v2)
5. [추론 방법 (Action 추출)](#5-추론-방법-action-추출)
6. [Action Dump (대량 추출)](#6-action-dump-대량-추출)
7. [파인튜닝](#7-파인튜닝)
8. [핵심 파라미터 레퍼런스](#8-핵심-파라미터-레퍼런스)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. 개요 - IDM이란?

### 1.1 IDM의 역할

IDM은 **"이 장면에서 저 장면으로 가려면 로봇이 어떤 동작을 해야 하는가?"** 를 예측합니다.

```
[프레임 t] ──┐
             ├──→ IDM ──→ Action Prediction (16 timesteps × 32 dims)
[프레임 t+16] ─┘
```

**주요 활용 사례:**
- 비디오에서 Action Label 자동 생성 (teleoperation 없이 데이터 수집)
- Cosmos-Predict2가 생성한 합성 비디오에 Action 부여 → VLA 학습 데이터 생성
- 기존 데이터셋의 Action 보정/재생성

### 1.2 아키텍처

| 구성 요소 | 파라미터 | 역할 |
|-----------|---------|------|
| SiGLIP2 Vision Tower | ~400M | 이미지 인코딩 (google/siglip2-large-patch16-256) |
| DiT (Diffusion Transformer) | ~122M | Flow Matching으로 action 생성 |
| Self-Attention Transformer | ~50M | Vision-Language 토큰 처리 |
| CategorySpecificLinear | ~40M | **로봇별** action 인코딩/디코딩 (multi-embodiment 지원) |
| **총계** | **~612M** | |

### 1.3 입출력

- **입력**: 비디오 프레임 2장 (256×256, SiGLIP 전처리)
- **출력**: `action_pred` — shape `(batch, 16, 32)`
  - `16` = action horizon (16 timestep 예측)
  - `32` = max action dimension (실제 사용 시 로봇 관절 수만큼만 유효)
  - 값 범위: **정규화된 [-1, 1]** → 실제 값으로 변환 필요

---

## 2. 환경 설정

### 2.1 가상환경 생성

```bash
cd /path/to/GR00T-Dreams-IDM
uv venv .venv --python 3.12
source .venv/bin/activate
```

### 2.2 의존성 설치

```bash
# Blackwell GPU (RTX 5090)는 PyTorch nightly 필수
uv pip install --pre torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# 프로젝트 의존성 (torch 충돌 방지를 위해 --no-deps)
uv pip install -e . --no-deps

# 수동으로 핵심 패키지 설치
uv pip install albumentations av decord diffusers einops \
  hydra-core omegaconf transformers accelerate wandb \
  tensorboard pyarrow tianshou tyro peft lightning \
  pandas numpy pydantic kornia timm tqdm
```

### 2.3 검증

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
from gr00t.model.idm import IDM
print('IDM import OK')
"
```

---

## 3. 추론에 필요한 데이터

추론 방법에 따라 필요한 데이터가 다릅니다:

### 3.1 방법별 필요 데이터

| 추론 방법 | 필요한 데이터 | 결과물 |
|-----------|-------------|--------|
| **비디오 1개 추론** | 비디오 파일 (.mp4) 1개 | 정규화된 action 예측값 |
| **데이터셋 추론** | LeRobot v2 데이터셋 | action 예측값 + GT 비교 |
| **Action Dump** | LeRobot v2 데이터셋 | action이 채워진 새 데이터셋 |
| **더미 테스트** | 없음 (랜덤 생성) | 모델 동작 검증 |

### 3.2 최소 요구사항: 비디오 1개 추론

비디오 파일 1개만 있으면 추론 가능합니다. 추가로 필요한 것:

- **IDM 체크포인트**: `config.json` + `model.safetensors`가 포함된 디렉토리
  - 우리가 학습한 체크포인트: `/home/lambda/claude/idm_output_so100/checkpoint-5000/`
  - 또는 HuggingFace: `nvidia/GR00T-IDM`, `seonghyeonye/IDM_so100`
- **Embodiment ID**: 어떤 로봇인지 지정 (so100=26, omx=30, gr1=24 등)

### 3.3 데이터셋 기반 추론 / Action Dump

LeRobot v2 형식의 완전한 데이터셋이 필요합니다.
→ [4. 데이터셋 구성 방법](#4-데이터셋-구성-방법-lerobot-v2) 참조

---

## 4. 데이터셋 구성 방법 (LeRobot v2)

### 4.1 디렉토리 구조

```
my_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet    ← 에피소드별 parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   ├── observation.images.top/       ← 카메라 이름 = info.json의 feature key
│   │   └── chunk-000/
│   │       ├── episode_000000.mp4    ← h264 인코딩, 에피소드별 1개
│   │       ├── episode_000001.mp4
│   │       └── ...
│   └── observation.images.wrist/     ← (선택) 두 번째 카메라
│       └── chunk-000/
│           └── ...
└── meta/
    ├── info.json                     ← 데이터셋 메타정보
    ├── modality.json                 ← ★ 가장 중요! 키 매핑
    ├── stats.json                    ← 정규화용 통계
    ├── episodes.jsonl                ← 에피소드 목록
    └── tasks.jsonl                   ← 태스크 목록
```

### 4.2 Parquet 파일 형식

각 에피소드 parquet 파일에 필요한 컬럼:

| 컬럼 | dtype | 예시 | 설명 |
|------|-------|------|------|
| `action` | list[float32] | `[1.5, 130.2, ...]` | 관절 action 값 (관절 수 만큼) |
| `observation.state` | list[float32] | `[1.5, 130.2, ...]` | 관절 상태 값 |
| `timestamp` | float64 | `0.0, 0.0333, ...` | 프레임 타임스탬프 (초, 0부터 시작) |
| `frame_index` | int64 | `0, 1, 2, ...` | 에피소드 내 프레임 번호 |
| `episode_index` | int64 | `0` (에피소드 전체 동일) | 에피소드 번호 |
| `index` | int64 | `0, 1, 2, ...` | 전체 데이터셋 글로벌 인덱스 |
| `task_index` | int64 | `0` | 태스크 번호 |

**주의**: `timestamp`는 0부터 시작해야 합니다. FPS=30이면 `0.0, 0.0333, 0.0667, ...`

```python
# Parquet 생성 예시
import pandas as pd
import numpy as np

num_frames = 450
fps = 30

df = pd.DataFrame({
    "action": [np.array([1.5, 130.2, 129.5, 65.3, 87.9, 10.1], dtype=np.float32) for _ in range(num_frames)],
    "observation.state": [np.array([1.5, 130.2, 129.5, 65.3, 87.9, 10.1], dtype=np.float32) for _ in range(num_frames)],
    "timestamp": np.arange(num_frames) / fps,
    "frame_index": np.arange(num_frames),
    "episode_index": np.full(num_frames, 0),
    "index": np.arange(num_frames),  # 글로벌 인덱스 (에피소드가 여러 개면 누적)
    "task_index": np.zeros(num_frames, dtype=int),
})

df.to_parquet("data/chunk-000/episode_000000.parquet", index=False)
```

### 4.3 비디오 파일 요구사항

| 항목 | 값 |
|------|-----|
| 코덱 | **h264** (필수) |
| 컨테이너 | .mp4 |
| 해상도 | 원본 유지 (학습 시 자동 리사이즈됨) |
| FPS | parquet의 timestamp와 일치해야 함 |
| pixel format | yuv420p 권장 |

```bash
# 비디오 인코딩 확인
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of csv=p=0 episode_000000.mp4
# 출력: h264,640,480,30/1

# 비디오 인코딩 변환 (필요시)
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### 4.4 메타데이터 파일

#### info.json

```json
{
  "codebase_version": "v2.0",
  "robot_type": "so100",
  "total_episodes": 158,
  "total_frames": 78300,
  "total_tasks": 3,
  "chunks_size": 1000,
  "fps": 30,
  "splits": { "train": "0:158" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": ["main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
                "main_wrist_flex", "main_wrist_roll", "main_gripper"]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": ["main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
                "main_wrist_flex", "main_wrist_roll", "main_gripper"]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channels"],
      "info": {
        "video.fps": 30.0, "video.height": 480, "video.width": 640,
        "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false, "has_audio": false
      }
    }
  }
}
```

**핵심 포인트:**
- `data_path`는 반드시 `episode_chunk`, `episode_index` 변수 사용 (`chunk_index`, `file_index` 아님!)
- `video_path`에서 `{video_key}`는 `observation.images.top` 같은 feature key로 치환됨
- `features` 의 `names` 배열은 modality.json의 키와 매칭되어야 함

#### modality.json (★ 가장 중요한 파일)

이 파일은 DataConfig의 키 이름을 실제 parquet 데이터 키로 매핑합니다.

```json
{
  "state": {
    "main_shoulder_pan": { "start": 0, "end": 1, "original_key": "observation.state" },
    "main_shoulder_lift": { "start": 1, "end": 2, "original_key": "observation.state" },
    "main_elbow_flex": { "start": 2, "end": 3, "original_key": "observation.state" },
    "main_wrist_flex": { "start": 3, "end": 4, "original_key": "observation.state" },
    "main_wrist_roll": { "start": 4, "end": 5, "original_key": "observation.state" },
    "main_gripper": { "start": 5, "end": 6, "original_key": "observation.state" }
  },
  "action": {
    "main_shoulder_pan": { "start": 0, "end": 1, "absolute": false, "original_key": "action" },
    "main_shoulder_lift": { "start": 1, "end": 2, "absolute": false, "original_key": "action" },
    "main_elbow_flex": { "start": 2, "end": 3, "absolute": false, "original_key": "action" },
    "main_wrist_flex": { "start": 3, "end": 4, "absolute": false, "original_key": "action" },
    "main_wrist_roll": { "start": 4, "end": 5, "absolute": false, "original_key": "action" },
    "main_gripper": { "start": 5, "end": 6, "absolute": true, "original_key": "action" }
  },
  "video": {
    "webcam": { "original_key": "observation.images.top" }
  },
  "annotation": {
    "human.task_description": { "original_key": "task_index" }
  }
}
```

**필드 설명:**
- `start`, `end`: parquet의 action/state 배열에서의 인덱스 범위
- `absolute`: `true` = 그리퍼처럼 절대값, `false` = 관절 각도/속도
- `original_key`: parquet 컬럼 이름
- `video.webcam`: DataConfig의 `video_keys`에서 `video.` 뒤의 이름과 매칭
  - So100DataConfig의 `video_keys = ["video.webcam"]` → `"webcam": {"original_key": "observation.images.top"}`

#### stats.json

정규화를 위한 통계값. **추론 결과를 실제 값으로 변환할 때 필수.**

```json
{
  "observation.state": {
    "mean": [1.46, 129.6, 131.0, 65.7, 87.9, 10.1],
    "std": [31.4, 39.3, 30.0, 15.2, 14.3, 8.7],
    "min": [-79.4, 26.2, 2.9, 9.9, 45.9, -0.3],
    "max": [76.1, 179.8, 167.3, 98.6, 135.9, 34.7],
    "q01": [-75.1, 48.3, 47.5, 26.9, 56.5, 0.07],
    "q99": [69.9, 176.9, 165.0, 91.3, 124.0, 28.7]
  },
  "action": {
    "mean": [...], "std": [...],
    "min": [...], "max": [...],
    "q01": [...], "q99": [...]
  }
}
```

**생성 스크립트:**

```python
import pandas as pd, numpy as np, json
from pathlib import Path

dataset_dir = "/path/to/my_dataset"
data_dir = Path(dataset_dir) / "data" / "chunk-000"

all_actions, all_states = [], []
for ep_file in sorted(data_dir.glob("episode_*.parquet")):
    df = pd.read_parquet(ep_file)
    all_actions.append(np.stack(df["action"].values))
    all_states.append(np.stack(df["observation.state"].values))

actions = np.concatenate(all_actions)
states = np.concatenate(all_states)

def compute_stats(data):
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "q01": np.percentile(data, 1, axis=0).tolist(),
        "q99": np.percentile(data, 99, axis=0).tolist(),
    }

stats = {
    "observation.state": compute_stats(states),
    "action": compute_stats(actions),
}

with open(f"{dataset_dir}/meta/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
```

#### episodes.jsonl

```jsonl
{"episode_index": 0, "tasks": ["pick and place objects"], "length": 454}
{"episode_index": 1, "tasks": ["pick and place objects"], "length": 468}
{"episode_index": 2, "tasks": ["pick and place objects"], "length": 420}
```

- `length`는 해당 에피소드 parquet의 실제 행 수와 **정확히 일치**해야 함
- 불일치 시 `IndexError: index N out of bounds` 발생

#### tasks.jsonl

```jsonl
{"task_index": 0, "task": "pick and place objects"}
{"task_index": 1, "task": "stack blocks"}
{"task_index": 2, "task": "sort objects"}
```

### 4.5 데이터셋 검증 체크리스트

| # | 항목 | 검증 방법 |
|---|------|----------|
| 1 | 빈 에피소드 없는지 | `for f in data/chunk-000/*.parquet: if rows==0: FAIL` |
| 2 | episodes.jsonl length 일치 | 각 에피소드 parquet 행 수 == jsonl의 length |
| 3 | 비디오 수 == 에피소드 수 | `ls videos/*/chunk-000/*.mp4 | wc -l` |
| 4 | timestamp 0부터 시작 | 각 parquet에서 `timestamp.min() == 0.0` |
| 5 | video codec h264 | `ffprobe` 로 확인 |
| 6 | info.json data_path 형식 | `episode_chunk`, `episode_index` 사용 확인 |
| 7 | modality.json 키 매핑 | DataConfig의 키 이름과 일치 확인 |
| 8 | stats.json q01/q99 포함 | 역정규화에 필수 |

검증 스크립트:
```bash
python examples/idm_preprocessing_example.py --mode validate --input_dir /path/to/dataset
```

---

## 5. 추론 방법 (Action 추출)

### 5.1 방법 A: 추론 스크립트 사용 (가장 쉬움)

`examples/idm_inference_example.py` 스크립트로 코드 수정 없이 바로 사용 가능합니다.

#### 더미 데이터 테스트 (모델 동작 확인)

```bash
cd /home/lambda/claude/GR00T-Dreams-IDM
source .venv/bin/activate

python examples/idm_inference_example.py \
    --checkpoint /home/lambda/claude/idm_output_so100/checkpoint-5000 \
    --test-dummy \
    --device cuda
```

데이터 없이 모델 로드 및 추론 파이프라인을 검증합니다.

#### 단일 비디오에서 추론

```bash
python examples/idm_inference_example.py \
    --checkpoint /home/lambda/claude/idm_output_so100/checkpoint-5000 \
    --video /path/to/video.mp4 \
    --embodiment so100 \
    --device cuda
```

**필요한 것**: 비디오 파일 1개 + 체크포인트
**결과**: 정규화된 action 예측값 ([-1, 1] 범위)

#### 데이터셋에서 추론 (GT 비교 포함)

```bash
python examples/idm_inference_example.py \
    --checkpoint /home/lambda/claude/idm_output_so100/checkpoint-5000 \
    --dataset /home/lambda/claude/datasets/so100/so100_combined \
    --data-config so100 \
    --num-samples 10 \
    --batch-size 4 \
    --device cuda
```

**필요한 것**: LeRobot v2 데이터셋 + 체크포인트
**결과**: action 예측값 + Ground Truth와의 MSE/MAE 비교

#### 벤치마크 (추론 속도 측정)

```bash
python examples/idm_inference_example.py \
    --checkpoint /home/lambda/claude/idm_output_so100/checkpoint-5000 \
    --benchmark \
    --device cuda
```

### 5.2 방법 B: Python 코드로 직접 추론

#### 비디오 파일에서 추론

```python
import decord
import PIL.Image
import torch
import numpy as np
from gr00t.model.idm import IDM
from gr00t.model.action_head.siglip import SiglipProcessor

# 1. 모델 로드
model = IDM.from_pretrained("/home/lambda/claude/idm_output_so100/checkpoint-5000")
model.eval()
model.requires_grad_(False)
model.to("cuda")

# 2. 비디오에서 2개 프레임 추출 (frame 0과 frame 16)
vr = decord.VideoReader("video.mp4")
frames = vr.get_batch([0, 16]).asnumpy()

# 3. SiGLIP 전처리
processor = SiglipProcessor.from_pretrained("google/siglip2-large-patch16-256")
processed = []
for frame in frames:
    img = PIL.Image.fromarray(frame).resize((256, 256))
    p = processor.image_processor(images=[img])["pixel_values"]
    processed.append(p)
images = np.concatenate(processed, axis=0)

# 4. 배치 구성
#    Embodiment ID: gr1=24, franka=17, so100=26, omx=30, new_embodiment=31
EMBODIMENT_ID = 26  # so100
action_horizon = 16

batch = {
    "images": torch.from_numpy(images).to("cuda"),
    "view_ids": torch.tensor([0, 1], device="cuda"),
    "embodiment_id": torch.tensor([EMBODIMENT_ID], device="cuda"),
    "vl_token_ids": torch.zeros((1, 112), dtype=torch.long, device="cuda"),
    "sa_token_ids": torch.full((1, action_horizon), 4, dtype=torch.long, device="cuda"),
    "vl_attn_mask": torch.ones((1, 112), dtype=torch.bool, device="cuda"),
}

# 이미지 토큰 위치 설정 (마지막 32개 = 16 tokens/frame × 2 frames)
for i in range(2):
    start = 112 - (2 - i) * 16
    batch["vl_token_ids"][0, start:start+16] = 1  # IMG_TOKEN

# 5. 추론
with torch.inference_mode():
    output = model.get_action(batch)

pred = output["action_pred"][0]  # shape: (16, 32)
# pred[:, :6] → SO-100의 6관절에 해당하는 정규화된 action
```

#### 데이터셋에서 추론

```python
import torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
from gr00t.model.idm import IDM
from gr00t.model.transforms_idm import DefaultDataCollatorGR00TIDM

# 1. 모델 로드
model = IDM.from_pretrained("/home/lambda/claude/idm_output_so100/checkpoint-5000")
model.eval(); model.requires_grad_(False); model.to("cuda")

# 2. 데이터셋 로드 (data_config: "so100", "omx", "franka", "gr1_arms_only" 등)
cfg = DATA_CONFIG_MAP["so100"]
dataset = LeRobotSingleDataset(
    dataset_path="/home/lambda/claude/datasets/so100/so100_combined",
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag=EmbodimentTag("new_embodiment"),
    video_backend="decord",
)
dataset.transforms.eval()  # augmentation 비활성화

# 3. 추론
collator = DefaultDataCollatorGR00TIDM()
batch = collator([dataset[0]])
batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.inference_mode():
    output = model.get_action(batch)

pred_actions = output["action_pred"]  # shape: (1, 16, 32)
```

### 5.3 예측값을 실제 값으로 변환 (역정규화)

모델 출력은 **정규화된 [-1, 1]** 범위입니다. 실제 각도/위치로 변환하려면:

#### 정규화 방식: q99 (기본값)

```
정규화:   norm = 2 * (x - q01) / (q99 - q01) - 1      (결과를 [-1, 1]로 clip)
역정규화: x    = (norm + 1) / 2 * (q99 - q01) + q01
```

#### 변환 코드

```python
import json
import numpy as np

# stats.json에서 q01, q99 로드
with open("/path/to/dataset/meta/stats.json") as f:
    stats = json.load(f)

q01 = np.array(stats["action"]["q01"])[:6]  # 로봇 관절 수만큼
q99 = np.array(stats["action"]["q99"])[:6]

def denormalize_q99(normalized, q01, q99):
    """정규화된 [-1,1] 값을 실제 값으로 변환"""
    return (normalized + 1) / 2 * (q99 - q01) + q01

# 사용
pred_normalized = output["action_pred"][0, 0, :6].cpu().numpy()  # 첫 timestep, 6관절
pred_degrees = denormalize_q99(pred_normalized, q01, q99)
print(f"예측된 관절 각도: {pred_degrees}")
# 예: [1.5, 130.2, 129.5, 65.3, 87.9, 10.1] (도 단위)
```

**주의**: DataConfig에 따라 정규화 방식이 다를 수 있습니다.
- So100DataConfig, OmxDataConfig: `min_max` 정규화 사용
- Gr1ArmsOnlyDataConfig: `q99` 정규화 사용
- stats.json에 `q01`/`q99`가 없으면 `min`/`max`로 대체

### 5.4 추론 성능 참고값

| 배치 크기 | 추론 시간 (RTX 5090) |
|-----------|---------------------|
| 1 | ~61ms |
| 4 | ~100ms |
| 16 | ~250ms |

---

## 6. Action Dump (대량 추출)

비디오 데이터셋의 **모든 에피소드에 대해 Action을 자동 생성**하여 새 데이터셋을 만듭니다.
VLA 학습 데이터 생성의 핵심 파이프라인입니다.

### 6.1 dump_idm_actions.py 사용법

```bash
cd /home/lambda/claude/GR00T-Dreams-IDM
source .venv/bin/activate

python IDM_dump/dump_idm_actions.py \
    --checkpoint /home/lambda/claude/idm_output_so100/checkpoint-5000 \
    --dataset /path/to/input_dataset \
    --output_dir /path/to/output_dataset \
    --num_gpus 1 \
    --batch_size 16 \
    --video_indices "0 16"
```

**옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | (필수) | IDM 체크포인트 경로 또는 HuggingFace ID |
| `--dataset` | (필수) | 입력 LeRobot v2 데이터셋 경로 |
| `--output_dir` | (필수) | 출력 데이터셋 경로 |
| `--num_gpus` | 8 | 사용할 GPU 수 (0=CPU) |
| `--batch_size` | 32 | 배치 크기 |
| `--max_episodes` | None | 처리할 최대 에피소드 수 (테스트용) |
| `--num_workers` | 16 | GPU당 데이터 로딩 스레드 수 |
| `--video_indices` | None | 비디오 프레임 인덱스 (예: "0 16") |

### 6.2 동작 원리

```
입력 데이터셋                         IDM 모델                    출력 데이터셋
─────────────                    ──────────                 ─────────────
data/chunk-000/                                            data/chunk-000/
  episode_000000.parquet ──┐                               episode_000000.parquet (action 업데이트됨)
                           │
videos/observation.images.top/   IDM.get_action()
  episode_000000.mp4 ─────┤──→ 프레임 추출 → 추론 ──→     각 timestep의 action을
                           │     (frame t, t+16)           평균하여 parquet에 기록
meta/ ────────────────────┘
  modality.json (키 매핑)                                   meta/ (복사됨, tasks에 <DREAM> 접두사)
  stats.json (역정규화)                                     videos/ (복사됨)
```

핵심 로직:
1. 에피소드의 모든 프레임에 대해 IDM 추론 실행
2. 각 timestep에서 action_horizon(16)만큼의 예측값 생성
3. **동일 timestep에 대한 여러 예측을 평균**하여 최종 action 결정
4. 역정규화(unapply) 후 parquet의 action 컬럼에 기록
5. meta/와 videos/ 복사, tasks.jsonl에 `<DREAM>` 접두사 추가

### 6.3 주의사항

- `dump_idm_actions.py`는 `experiment_cfg/conf.yaml`이 있는 공식 체크포인트 기준으로 설계됨
- 로컬 학습 체크포인트 (conf.yaml 없음) 사용 시 기본 `gr1_arms_only` DataConfig로 폴백됨
- **SO-100/OMX 로컬 체크포인트 사용 시**: `load_dataset_and_config()` 함수에서 올바른 DataConfig를 매핑하도록 수정 필요

### 6.4 전체 파이프라인 (비디오 → VLA 학습 데이터)

```
비디오 수집          →  전처리        →  LeRobot 변환  →  IDM Action Dump  →  VLA 학습
(mp4 파일들)           (크롭/리사이즈)   (데이터셋 구성)   (action 자동생성)    (GR00T-N1)

IDM_dump/scripts/preprocess/so100.sh 순서:
1. split_video_instruction.py  → 비디오 분할 + 지시문 추출
2. preprocess_video.py         → 비디오 크롭/리사이즈
3. raw_to_lerobot.py           → LeRobot v2 형식 변환
4. dump_idm_actions.py         → Action 자동 생성
```

---

## 7. 파인튜닝

### 7.1 언제 파인튜닝이 필요한가?

| 상황 | 파인튜닝 필요? |
|------|-------------|
| 기존 로봇(SO-100/OMX/GR1/Franka)으로 같은 종류의 작업 | **아니오** - 기존 체크포인트 사용 |
| 기존 로봇으로 새로운 환경/카메라 앵글 | **예** - 환경 적응 필요 |
| 기존 로봇으로 완전히 다른 작업 | **선택** - 기존 모델로 테스트 후 결정 |
| 새로운 로봇 (관절 구조 다름) | **예** - 새 embodiment 학습 필요 |

### 7.2 학습 명령어

```bash
cd /home/lambda/claude/GR00T-Dreams-IDM
source .venv/bin/activate

# 기본: base 모델에서 학습
python scripts/idm_training.py \
    --dataset-path /home/lambda/claude/datasets/so100/so100_combined \
    --data-config so100 \
    --output-dir /home/lambda/claude/idm_output_so100 \
    --num-gpus 1 \
    --batch-size 16 \
    --max-steps 5000 \
    --save-steps 1000 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 8 \
    --report-to tensorboard \
    --embodiment-tag new_embodiment

# 체크포인트에서 이어서 학습 (transfer learning)
python scripts/idm_training.py \
    --base-model-path /home/lambda/claude/idm_output_so100/checkpoint-5000 \
    --dataset-path /path/to/new_task_dataset \
    --data-config so100 \
    --output-dir /path/to/output \
    --max-steps 3000
```

### 7.3 주요 학습 옵션

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--base-model-path` | `IDM_dump/base.yaml` | base 모델 경로 (YAML=초기화, 디렉토리=from_pretrained) |
| `--dataset-path` | (필수) | LeRobot v2 데이터셋 경로 |
| `--data-config` | `gr1_arms_only` | 데이터 설정 (so100, omx, franka 등) |
| `--batch-size` | 16 | GPU당 배치 크기 |
| `--max-steps` | 10000 | 총 학습 step |
| `--learning-rate` | 1e-4 | 학습률 |
| `--gradient-checkpointing` | False | VRAM 절약 (속도 ~30% 감소) |
| `--gradient-accumulation-steps` | 1 | 그래디언트 누적 |
| `--no-tune-vision-tower` | - | SiGLIP2 비전 타워 고정 (~3GB VRAM 절약) |
| `--dataloader-num-workers` | 8 | 데이터 로딩 워커 수 (**0이면 20배 느림**) |
| `--embodiment-tag` | `new_embodiment` | 로봇 유형 태그 |
| `--report-to` | `wandb` | 로깅 (wandb, tensorboard, none) |
| `--resume` | False | 동일 output-dir에서 학습 재개 |

### 7.4 지원하는 data-config

| Config 이름 | 로봇 | 관절 수 | Embodiment Tag |
|------------|------|--------|----------------|
| `so100` | SO-100 | 6 | `so100` (26) 또는 `new_embodiment` (31) |
| `omx` | OMX | 6 | `omx` (30) 또는 `new_embodiment` (31) |
| `franka` | Franka Emika Panda | 7+1 | `franka` (17) |
| `gr1_arms_only` | GR1 양팔 | 14 | `gr1` (24) |
| `gr1_arms_waist` | GR1 양팔+허리 | 17 | `gr1` (24) |
| `gr1_full_upper_body` | GR1 상체 | 22 | `gr1` (24) |
| `bimanual_panda_gripper` | 양팔 Panda | 14+2 | - |
| `single_panda_gripper` | 단일 Panda | 7+1 | - |

### 7.5 VRAM별 권장 설정

| GPU VRAM | batch_size | vision_tower | gradient_ckpt | 예상 속도 |
|----------|-----------|-------------|---------------|----------|
| 8GB | 2 | 고정 | 활성화 | ~0.5 step/s |
| 16GB | 8 | 고정 | 비활성화 | ~2 step/s |
| 24GB | 16 | 학습 | 비활성화 | ~3 step/s |
| **32GB** | **16** | **학습** | **비활성화** | **~3.4 step/s** |

### 7.6 학습 모니터링

```bash
# TensorBoard
tensorboard --logdir /path/to/output/runs --port 6006
```

핵심 메트릭:
- `train/loss`: 0.5 이하 → 수렴 시작, 0.2 이하 → 좋은 수렴
- `train/grad_norm`: 2.0 이하가 안정적
- `train/learning_rate`: warmup 후 cosine decay

### 7.7 학습 결과 참고값

| 데이터셋 | Steps | 최종 Loss | 소요시간 (RTX 5090) |
|---------|-------|----------|-------------------|
| OMX 158ep / 78K frames | 10,000 | 0.381 | 48분 |
| SO-100 158ep / 78K frames | 5,000 | 0.170 | 24분 |

---

## 8. 핵심 파라미터 레퍼런스

### 8.1 EmbodimentTag → ID 매핑

| Tag | ID | 설명 |
|-----|-----|------|
| `gr1` | 24 | NVIDIA GR1 휴머노이드 |
| `franka` | 17 | Franka Emika Panda |
| `so100` | 26 | SO-100 로봇팔 |
| `omx` | 30 | OMX (SO-100 변환) |
| `new_embodiment` | 31 | **새 로봇용 (기본 권장)** |

### 8.2 모델 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| action_horizon | 16 | 예측 타임스텝 수 |
| action_dim | 32 | 최대 action 차원 |
| max_state_dim | 64 | 최대 state 차원 |
| num_inference_timesteps | 16 | Flow Matching 디노이징 스텝 |
| max_sequence_length | 112 | VL 시퀀스 길이 |
| dropout | 0.2 | DiT 드롭아웃 |

### 8.3 체크포인트 디렉토리 구조

```
checkpoint-5000/
├── config.json              ← 모델 설정 (from_pretrained에 필수)
├── model.safetensors        ← 모델 가중치 (2.4GB)
├── optimizer.pt             ← 옵티마이저 상태 (학습 재개용)
├── scheduler.pt             ← LR 스케줄러
├── trainer_state.json       ← 학습 상태 (loss 기록 등)
└── experiment_cfg/          ← (공식 체크포인트만) Hydra 설정
    └── conf.yaml
```

### 8.4 현재 보유 체크포인트

| 체크포인트 | 경로 | 학습 데이터 |
|-----------|------|-----------|
| SO-100 5K | `/home/lambda/claude/idm_output_so100/checkpoint-5000/` | so100_combined 158ep |
| OMX 10K | `/home/lambda/claude/idm_output/checkpoint-10000/` | omx_combined 158ep |

### 8.5 현재 보유 데이터셋

| 데이터셋 | 경로 | 에피소드 | 프레임 |
|---------|------|---------|--------|
| SO-100 combined | `/home/lambda/claude/datasets/so100/so100_combined/` | 158 | 78,300 |
| OMX combined | `/home/lambda/claude/datasets/omx/omx_combined/` | 158 | - |
| SO-100 h264 원본 | `/home/lambda/claude/datasets/so100/so100_h264/` | 158 (3 서브셋) | - |

---

## 9. 트러블슈팅

### 9.1 자주 만나는 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| `persistent_workers needs num_workers > 0` | `num_workers=0`과 `persistent_workers=True` 충돌 | 이미 수정됨 (idm_training.py) |
| `IndexError: index N out of bounds size 0` | 빈 에피소드 parquet 파일 | parquet 행 수 확인, 빈 파일 재생성 |
| `IndexError: index N out of bounds size M` | episodes.jsonl의 length와 실제 parquet 불일치 | episodes.jsonl 재생성 |
| `KeyError: 'video_path'` | info.json에 `video_path` 누락 | `video_path` 필드 추가 |
| `KeyError: 'episode_chunk'` | data_path에서 `chunk_index` 사용 | `episode_chunk`로 변경 (v2 형식) |
| `Action dim not found in action_dims` | modality.json 키 매핑 오류 | DataConfig 키와 일치하는지 확인 |
| 학습 매우 느림 (~0.18 step/s) | `num_workers=0`으로 인한 CPU 병목 | `--dataloader-num-workers 8` 사용 |

### 9.2 episodes.jsonl 재생성 스크립트

```python
import json
import pandas as pd
from pathlib import Path

dataset_dir = "/path/to/dataset"
data_dir = Path(dataset_dir) / "data" / "chunk-000"

# 기존 tasks 정보 로드
tasks_map = {}
tasks_file = Path(dataset_dir) / "meta" / "tasks.jsonl"
if tasks_file.exists():
    with open(tasks_file) as f:
        for line in f:
            t = json.loads(line)
            tasks_map[t["task_index"]] = t["task"]

# 실제 parquet에서 episodes.jsonl 재생성
episodes = []
for ep_file in sorted(data_dir.glob("episode_*.parquet")):
    ep_idx = int(ep_file.stem.split("_")[1])
    df = pd.read_parquet(ep_file)
    task_idx = int(df["task_index"].iloc[0])
    task_name = tasks_map.get(task_idx, f"task_{task_idx}")
    episodes.append({
        "episode_index": ep_idx,
        "tasks": [task_name],
        "length": len(df),
    })

with open(Path(dataset_dir) / "meta" / "episodes.jsonl", "w") as f:
    for ep in episodes:
        f.write(json.dumps(ep) + "\n")

print(f"Regenerated {len(episodes)} episodes")
```

### 9.3 비디오 인코딩 문제

```bash
# 모든 비디오를 h264로 일괄 변환
for f in videos/observation.images.top/chunk-000/*.mp4; do
    ffmpeg -i "$f" -c:v libx264 -pix_fmt yuv420p -y "${f%.mp4}_h264.mp4"
    mv "${f%.mp4}_h264.mp4" "$f"
done
```

---

## 부록: 주요 파일 위치 요약

```
GR00T-Dreams-IDM/
├── scripts/
│   └── idm_training.py              ← 학습 스크립트 (수정됨: base_model_path 추가)
├── examples/
│   ├── idm_inference_example.py     ← 추론 스크립트 (4가지 모드)
│   └── idm_preprocessing_example.py ← 전처리/검증 스크립트
├── IDM_dump/
│   ├── dump_idm_actions.py          ← Action 대량 추출
│   ├── base.yaml                    ← Hydra 모델 설정
│   └── scripts/preprocess/          ← 로봇별 전처리 파이프라인
├── gr00t/
│   ├── model/idm.py                 ← IDM 모델 클래스
│   ├── data/dataset.py              ← LeRobotSingleDataset
│   ├── experiment/data_config_idm.py← DataConfig 정의 + DATA_CONFIG_MAP
│   └── data/transform/              ← 정규화/역정규화 변환
└── .venv/                           ← Python 가상환경
```
