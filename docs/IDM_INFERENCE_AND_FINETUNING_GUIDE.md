# GR00T-Dreams-IDM: 추론 & 파인튜닝 가이드

> 환경: RTX 5090 32GB, Python 3.12, PyTorch nightly+cu128
> 작성일: 2026-03-03

---

## 목차
1. [모델 개요](#1-모델-개요)
2. [환경 설정](#2-환경-설정)
3. [IDM 추론 방법](#3-idm-추론-방법)
4. [새 데이터셋으로 파인튜닝](#4-새-데이터셋으로-파인튜닝)
5. [핵심 파라미터 레퍼런스](#5-핵심-파라미터-레퍼런스)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 모델 개요

### IDM (Inverse Dynamics Model) 아키텍처

```
[Video Frame t] → SiGLIP2 Vision Tower (400M) → Vision Features
[Video Frame t+1] → SiGLIP2 Vision Tower        ↓
                                          DiT (Flow Matching)
                                                 ↓
                                     Action Prediction (16 timesteps)
```

| 구성 요소 | 파라미터 | 설명 |
|-----------|---------|------|
| SiGLIP2 Vision Tower | ~400M | 이미지 인코딩 (google/siglip2-large-patch16-256) |
| DiT (Diffusion Transformer) | ~122M | Flow Matching으로 action 생성 |
| Self-Attention Transformer | ~50M | VL token 처리 |
| CategorySpecificLinear | ~40M | 로봇별 action 인코딩/디코딩 |
| **총계** | **~612M** | |

### 입출력 형식

- **입력**: 2개 비디오 프레임 (256x256, SiGLIP 전처리)
- **출력**: `action_pred` shape `(batch, 16, 32)` — 16 timestep × 32 action dim
- 실제 사용 시 첫 6개 dim만 사용 (SO-100 6관절)

---

## 2. 환경 설정

### 2.1 가상환경 생성 (uv 사용)

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

# 수동으로 핵심 패키지 설치 (tensorflow, diffsynth 제외)
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

## 3. IDM 추론 방법

### 3.1 체크포인트에서 모델 로드

```python
from gr00t.model.idm import IDM

# 로컬 체크포인트에서 로드
model = IDM.from_pretrained("/path/to/checkpoint/")
model.eval()
model.requires_grad_(False)
model.to("cuda")

print(f"Action Horizon: {model.action_horizon}")  # 16
print(f"Action Dim: {model.action_dim}")           # 32
```

체크포인트 디렉토리에는 `config.json` + `model.safetensors`가 필요합니다.

### 3.2 데이터셋에서 추론

```python
import torch
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
from gr00t.model.idm import IDM
from gr00t.model.transforms_idm import DefaultDataCollatorGR00TIDM

# 1. 모델 로드
model = IDM.from_pretrained("/path/to/checkpoint/")
model.eval(); model.requires_grad_(False); model.to("cuda")

# 2. 데이터셋 로드
#    data_config: "so100", "omx", "franka", "gr1_arms_only" 등
cfg = DATA_CONFIG_MAP["so100"]
dataset = LeRobotSingleDataset(
    dataset_path="/path/to/dataset",
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag=EmbodimentTag("new_embodiment"),
    video_backend="decord",
)
dataset.transforms.eval()  # augmentation 비활성화

# 3. 추론
collator = DefaultDataCollatorGR00TIDM()
sample = dataset[0]
batch = collator([sample])
batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}

with torch.inference_mode():
    output = model.get_action(batch)

pred_actions = output["action_pred"]  # shape: (1, 16, 32)
print(f"Predicted actions: {pred_actions.shape}")
```

### 3.3 단일 비디오에서 추론

```python
import decord
import PIL.Image
import torch
import numpy as np
from gr00t.model.idm import IDM
from gr00t.model.action_head.siglip import SiglipProcessor

model = IDM.from_pretrained("/path/to/checkpoint/")
model.eval(); model.requires_grad_(False); model.to("cuda")

# 비디오에서 2개 프레임 추출
vr = decord.VideoReader("video.mp4")
frames = vr.get_batch([0, 16]).asnumpy()  # frame 0과 16

# SiGLIP 전처리
processor = SiglipProcessor.from_pretrained("google/siglip2-large-patch16-256")
processed = []
for frame in frames:
    img = PIL.Image.fromarray(frame).resize((256, 256))
    p = processor.image_processor(images=[img])["pixel_values"]
    processed.append(p)
images = np.concatenate(processed, axis=0)

# Embodiment ID 매핑
#   gr1=24, franka=17, so100=26, omx=30, new_embodiment=31
EMBODIMENT_ID = 31  # new_embodiment

# 배치 구성
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

with torch.inference_mode():
    output = model.get_action(batch)

pred = output["action_pred"][0]  # shape: (16, 32)
```

### 3.4 예측값 역정규화 (Denormalization)

모델 출력은 **정규화된 [-1, 1]** 범위입니다. 실제 값으로 변환하려면:

```python
import json
import numpy as np

# stats.json에서 정규화 파라미터 로드
with open("/path/to/dataset/meta/stats.json") as f:
    stats = json.load(f)

q01 = np.array(stats["action"]["q01"])[:6]  # 6 joints
q99 = np.array(stats["action"]["q99"])[:6]

def denormalize_q99(normalized, q01, q99):
    """q99 정규화 역변환: norm ∈ [-1,1] → 실제 값"""
    return (normalized + 1) / 2 * (q99 - q01) + q01

# 사용 예
pred_normalized = output["action_pred"][0, 0, :6].cpu().numpy()
pred_degrees = denormalize_q99(pred_normalized, q01, q99)
```

### 3.5 추론 성능

| 배치 크기 | 추론 시간 (RTX 5090) |
|-----------|---------------------|
| 1 | ~61ms |
| 4 | ~100ms |
| 16 | ~250ms |

---

## 4. 새 데이터셋으로 파인튜닝

### 4.1 데이터셋 준비 (LeRobot v2 형식)

#### 디렉토리 구조

```
my_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   ├── observation.images.top/
│   │   └── chunk-000/
│   │       ├── episode_000000.mp4    # h264 인코딩
│   │       └── ...
│   └── observation.images.wrist/     # (선택사항)
│       └── chunk-000/
│           └── ...
└── meta/
    ├── info.json
    ├── modality.json          ← 핵심! DataConfig 키 매핑
    ├── stats.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

#### info.json

```json
{
  "codebase_version": "v2.0",
  "robot_type": "my_robot",
  "total_episodes": 100,
  "total_frames": 50000,
  "total_tasks": 1,
  "chunks_size": 1000,
  "fps": 30,
  "splits": { "train": "0:100" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
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

#### modality.json (가장 중요!)

DataConfig의 키 이름을 실제 데이터 키로 매핑합니다:

```json
{
  "state": {
    "joint_1": {"start": 0, "end": 1, "original_key": "observation.state"},
    "joint_2": {"start": 1, "end": 2, "original_key": "observation.state"},
    "...": "..."
  },
  "action": {
    "joint_1": {"start": 0, "end": 1, "absolute": false, "original_key": "action"},
    "...": "...",
    "gripper": {"start": 5, "end": 6, "absolute": true, "original_key": "action"}
  },
  "video": {
    "webcam": {"original_key": "observation.images.top"}
  },
  "annotation": {
    "human.task_description": {"original_key": "task_index"}
  }
}
```

> **`absolute` 필드**: `true`면 그리퍼처럼 절대값, `false`면 상대값/속도

#### stats.json

```json
{
  "observation.state": {
    "mean": [...], "std": [...],
    "min": [...], "max": [...],
    "q01": [...], "q99": [...]
  },
  "action": {
    "mean": [...], "std": [...],
    "min": [...], "max": [...],
    "q01": [...], "q99": [...]
  }
}
```

생성 방법:
```python
import pandas as pd, numpy as np, json
from pathlib import Path

all_actions, all_states = [], []
for ep in range(num_episodes):
    df = pd.read_parquet(f"data/chunk-000/episode_{ep:06d}.parquet")
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
with open("meta/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
```

#### episodes.jsonl

```jsonl
{"episode_index": 0, "tasks": ["pick and place"], "length": 450}
{"episode_index": 1, "tasks": ["pick and place"], "length": 380}
```

#### tasks.jsonl

```jsonl
{"task_index": 0, "task": "pick and place"}
```

#### Parquet 컬럼 형식

각 에피소드 parquet 파일에 필요한 컬럼:

| 컬럼 | dtype | 설명 |
|------|-------|------|
| `action` | list[float32] | 관절 action 값 |
| `observation.state` | list[float32] | 관절 상태 값 |
| `timestamp` | float64 | 프레임 타임스탬프 (초) |
| `frame_index` | int64 | 에피소드 내 프레임 번호 |
| `episode_index` | int64 | 에피소드 번호 |
| `index` | int64 | 전체 데이터셋 내 글로벌 인덱스 |
| `task_index` | int64 | 태스크 번호 |

### 4.2 DataConfig 추가 (선택사항)

기존 `so100` 또는 `omx` config이 맞지 않는 새 로봇의 경우:

```python
# gr00t/experiment/data_config_idm.py에 추가

class MyRobotDataConfig(BaseDataConfig):
    """나의 로봇 데이터 설정"""

    @staticmethod
    def modality_config():
        return [
            VideoModalityConfig(
                delta_indices=np.array([0]),
                video_keys=["video.webcam"],
            ),
            StateModalityConfig(
                delta_indices=np.array(list(range(-1, 16))),
                state_keys=[
                    "state.joint_1", "state.joint_2",
                    # ... 관절 이름 나열
                ],
            ),
            ActionModalityConfig(
                delta_indices=np.array(list(range(0, 16))),
                action_keys=[
                    "action.joint_1", "action.joint_2",
                    # ... action 이름 나열
                ],
            ),
            LanguageModalityConfig(
                delta_indices=np.array([0]),
                language_keys=["annotation.human.task_description"],
            ),
        ]

    @staticmethod
    def transform():
        return ComposedModalityTransform(...)

# DATA_CONFIG_MAP에 등록
DATA_CONFIG_MAP["my_robot"] = MyRobotDataConfig
```

### 4.3 학습 실행

```bash
source .venv/bin/activate

python scripts/idm_training.py \
    --dataset-path /path/to/my_dataset \
    --data-config so100 \
    --output-dir /path/to/output \
    --num-gpus 1 \
    --batch-size 16 \
    --max-steps 5000 \
    --save-steps 1000 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 8 \
    --report-to tensorboard \
    --video-backend decord \
    --embodiment-tag new_embodiment
```

### 4.4 주요 학습 옵션

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--batch-size` | 16 | GPU당 배치 크기 |
| `--max-steps` | 10000 | 총 학습 step |
| `--learning-rate` | 1e-4 | 학습률 |
| `--gradient-checkpointing` | False | VRAM 절약 (속도 감소) |
| `--gradient-accumulation-steps` | 1 | 그래디언트 누적 (effective batch 증가) |
| `--tune-vision-tower` | True | SiGLIP2 비전 타워 학습 여부 |
| `--no-tune-vision-tower` | - | 비전 타워 고정 (~3GB VRAM 절약) |
| `--video-backend` | decord | 비디오 디코더 (decord 또는 torchvision_av) |
| `--embodiment-tag` | new_embodiment | 로봇 유형 태그 |

### 4.5 VRAM별 권장 설정

| GPU VRAM | batch_size | vision_tower | gradient_ckpt | 예상 속도 |
|----------|-----------|-------------|---------------|----------|
| 8GB | 2 | 고정 | 활성화 | ~0.5 step/s |
| 16GB | 8 | 고정 | 비활성화 | ~2 step/s |
| 24GB | 16 | 학습 | 비활성화 | ~3 step/s |
| **32GB** | **16** | **학습** | **비활성화** | **~3.4 step/s** |

### 4.6 학습 모니터링

```bash
# TensorBoard
tensorboard --logdir /path/to/output/runs --port 6006

# 핵심 메트릭
# - train/loss: 0.5 이하로 내려가면 수렴 시작
# - train/grad_norm: 2.0 이하가 안정적
# - train/learning_rate: warmup 후 cosine decay
```

### 4.7 체크포인트 구조

```
output_dir/
├── checkpoint-1000/
│   ├── config.json          # 모델 설정
│   ├── model.safetensors    # 모델 가중치 (2.4GB)
│   ├── optimizer.pt         # 옵티마이저 상태
│   ├── scheduler.pt         # LR 스케줄러
│   └── trainer_state.json   # 학습 상태
├── checkpoint-2000/
├── ...
└── runs/                    # TensorBoard 로그
```

---

## 5. 핵심 파라미터 레퍼런스

### 지원 로봇 (EmbodimentTag)

| Tag | ID | 설명 |
|-----|-----|------|
| `gr1` | 24 | NVIDIA GR1 휴머노이드 |
| `franka` | 17 | Franka Emika Panda |
| `so100` | 26 | SO-100 로봇팔 |
| `omx` | 30 | OMX (SO-100 변환) |
| `new_embodiment` | 31 | **새 로봇용 (기본 권장)** |

### DATA_CONFIG_MAP

| Config 이름 | 로봇 | 관절 수 | 비고 |
|------------|------|--------|------|
| `gr1_arms_only` | GR1 | 14 | 양팔만 |
| `gr1_arms_waist` | GR1 | 17 | 양팔 + 허리 |
| `gr1_full_upper_body` | GR1 | 22 | 상체 전체 |
| `so100` | SO-100 | 6 | 단일 팔 |
| `omx` | OMX | 6 | SO-100 변환 |
| `franka` | Franka | 7+1 | 7관절 + 그리퍼 |

### 모델 설정 (base.yaml)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| action_horizon | 16 | 예측 타임스텝 수 |
| action_dim | 32 | 최대 action 차원 |
| max_state_dim | 64 | 최대 state 차원 |
| num_inference_timesteps | 16 | Flow Matching 디노이징 스텝 |
| max_sequence_length | 112 | VL 시퀀스 길이 |
| dropout | 0.2 | DiT 드롭아웃 |

---

## 6. 트러블슈팅

### 자주 만나는 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| `persistent_workers needs num_workers > 0` | `num_workers=0`과 `persistent_workers=True` 충돌 | `idm_training.py`에서 `dataloader_persistent_workers=config.dataloader_num_workers > 0` |
| `IndexError: index N out of bounds size 0` | 빈 에피소드 parquet | `episodes.jsonl` 길이와 실제 parquet 크기 확인 |
| `KeyError: 'video_path'` | `info.json`에 `video_path` 누락 | `"video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"` 추가 |
| `KeyError: 'episode_chunk'` | `data_path`에서 `chunk_index` 사용 | `episode_chunk`로 변경 |
| `q01/q99 missing in stats` | stats.json에 q01/q99 미포함 | `np.percentile(data, [1, 99])` 로 추가 |
| `Action dim not found in action_dims` | `modality.json` 키 매핑 오류 | DataConfig의 action key 이름과 일치하는지 확인 |

### 학습 결과 참고값

| 데이터셋 | Steps | 최종 Loss | 소요시간 (RTX 5090) |
|---------|-------|----------|-------------------|
| OMX 158ep/78K frames | 10,000 | 0.381 | 48분 |
| SO-100 158ep/78K frames | 5,000 | 0.170 | 24분 |

### 추론 결과 참고값

| 테스트 데이터 | 모델 | MAE | 비고 |
|-------------|------|-----|------|
| OMX (delta action) | OMX_10K | 1.6도 | action 범위 ±3도 |
| OMX (delta action) | SO100_5K | 1.5도 | 추가학습 후에도 유지 |
| SO-100 (absolute) | OMX_10K | 21.4도 | action 범위 ~180도 |
| SO-100 (absolute) | SO100_5K | 23.8도 | 상대오차 ~13% |
