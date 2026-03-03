# IDM 학습 파이프라인 종합 가이드

> **목적**: 새 세션/새 GPU(5090 32GB 등)에서 IDM 학습을 처음부터 재현하거나, 기존 데이터로 재학습할 수 있도록 모든 과정을 기록합니다.

---

## 1. 프로젝트 개요

**GR00T-Dreams-IDM**은 NVIDIA의 Inverse Dynamics Model로, 비디오 프레임 쌍에서 로봇 관절 action을 예측합니다.

- **모델 아키텍처**: SiGLIP2-large (vision) + DiT (action head) + Flow Matching
- **총 파라미터**: 637.4M (612M trainable when all unfrozen)
- **입력**: 듀얼 카메라 비디오 프레임 (256x256) + robot state
- **출력**: action_pred [16, 32] (action_horizon=16, action_dim=32)

### 디렉토리 구조

```
/home/lambda/claude/
├── GR00T-Dreams-IDM/          # IDM 코드 (git repo)
│   ├── scripts/idm_training.py    # 학습 스크립트 (수정됨)
│   ├── gr00t/model/idm.py         # 모델 (수정됨)
│   ├── gr00t/utils/video.py       # 비디오 유틸 (수정됨)
│   └── IDM_dump/base.yaml         # 모델 config
├── pseudo-project/
│   └── scripts/
│       ├── convert_so100_to_omx.py       # SO-100 → OMX 변환
│       ├── postprocess_omx_dataset.py    # 후처리 (scalar→array)
│       └── merge_omx_datasets.py         # 데이터셋 병합
├── datasets/
│   ├── so100/                     # 원본 LeRobot 데이터셋
│   │   ├── svla_so100_pickplace/
│   │   ├── svla_so100_stacking/
│   │   └── svla_so100_sorting/
│   └── omx/                       # 변환된 OMX 데이터셋
│       ├── omx_pickplace/
│       ├── omx_stacking/
│       ├── omx_sorting/
│       └── omx_combined/          # 병합 데이터셋 (학습용)
└── idm_output/                    # 학습 결과
    ├── model.safetensors          # 최종 모델 (2.55GB)
    ├── checkpoint-{500..5000}/    # 체크포인트 8개
    └── runs/                      # TensorBoard 로그
```

---

## 2. 데이터셋 준비

### 2.0 변환 완료 데이터셋 다운로드 (권장 - 빠른 경로)

이미 SO-100 → OMX 변환, 후처리, 병합, H264 변환이 완료된 데이터셋이 HuggingFace에 업로드되어 있습니다.
**새 PC에서 시작할 때는 이 방법을 사용하세요** (섹션 2.1~2.5를 건너뛸 수 있습니다).

```bash
# 1. 다운로드 (Private repo - HuggingFace 인증 필요)
huggingface-cli login  # 토큰 입력
huggingface-cli download Yuseok/omx_combined_idm \
    --repo-type dataset \
    --local-dir datasets/omx/omx_combined

# 2. 에피소드별 비디오 심링크 생성 (필수!)
cd datasets/omx/omx_combined
python setup_symlinks.py
# → "Created 316 symlinks across 2 cameras" 출력 확인

# 3. 검증
python -c "
import decord
vr = decord.VideoReader('videos/observation.images.top/chunk-000/episode_000000.mp4')
print(f'Frames: {len(vr)}, Shape: {vr[0].shape}')
"
# → Frames: 19631, Shape: (480, 640, 3)
```

> **HuggingFace repo 구조**: 6개 고유 비디오 파일(pickplace/stacking/sorting × 2카메라)만 저장되어 있고,
> `setup_symlinks.py`가 158개 에피소드별 심링크를 생성합니다. 이렇게 하면 1.4GB만 다운로드하면 됩니다.

| 항목 | 값 |
|------|-----|
| Repo | `Yuseok/omx_combined_idm` (Private) |
| 크기 | 1.41GB (6 비디오 + 158 parquet + meta) |
| 에피소드 | 158 (pickplace 50 + stacking 56 + sorting 52) |
| 프레임 | 78,300 |
| 포맷 | LeRobot v3, H264 코덱, 30 FPS |
| Action/State | 6-DOF (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) |

**이 경로를 사용하면 바로 [섹션 4. 학습 실행](#4-학습-실행)으로 이동하세요.**

---

### 2.1 원본 다운로드 (LeRobot v3, SO-100) - 변환부터 시작할 경우

```bash
# HuggingFace에서 다운로드 (약 3.7GB 총)
pip install huggingface_hub
huggingface-cli download lerobot/svla_so100_pickplace --repo-type dataset --local-dir datasets/so100/svla_so100_pickplace
huggingface-cli download lerobot/svla_so100_stacking --repo-type dataset --local-dir datasets/so100/svla_so100_stacking
huggingface-cli download lerobot/svla_so100_sorting  --repo-type dataset --local-dir datasets/so100/svla_so100_sorting
```

| 데이터셋 | 에피소드 | 프레임 | 태스크 |
|----------|---------|--------|--------|
| pickplace | 50 | 19,631 | pick and place objects |
| stacking | 56 | 22,956 | stack blocks |
| sorting | 52 | 35,713 | sort objects |
| **합계** | **158** | **78,300** | 3 tasks |

### 2.2 SO-100 → OMX Kinematic 변환

SO-100과 OMX는 kinematic chain이 다르므로 URDF 기반 FK/IK 변환이 필요합니다.

```
SO-100 parquet(deg) → rad 변환 → SO-100 FK → 4x4 EE pose → OMX IK(scipy) → OMX joint(rad)
```

```bash
cd /home/lambda/claude/pseudo-project

# 자체 테스트 (URDF FK/IK 정확도 확인)
python scripts/convert_so100_to_omx.py --self-test

# 변환 실행 (8 워커 권장)
python scripts/convert_so100_to_omx.py \
    --source ../datasets/so100/svla_so100_pickplace \
    --output ../datasets/omx/omx_pickplace \
    --workers 8

python scripts/convert_so100_to_omx.py \
    --source ../datasets/so100/svla_so100_stacking \
    --output ../datasets/omx/omx_stacking \
    --workers 8

python scripts/convert_so100_to_omx.py \
    --source ../datasets/so100/svla_so100_sorting \
    --output ../datasets/omx/omx_sorting \
    --workers 8
```

> **중요**: 변환 스크립트 상단에 `OMP_NUM_THREADS=1` 등이 설정되어 있습니다.
> 이는 scipy의 L-BFGS-B가 OpenBLAS를 호출할 때 multiprocessing과 스레드 폭발을 방지합니다.
> (24코어 × 24스레드 = 576스레드로 load average 216 발생 경험)

### 2.3 후처리 (scalar → array columns)

변환 결과는 scalar columns (`state.shoulder_pan` 등)이지만,
IDM은 array columns (`observation.state[6]`, `action[6]`)을 요구합니다.

```bash
python scripts/postprocess_omx_dataset.py \
    --dataset-dir ../datasets/omx/omx_pickplace \
    --task "pick and place objects" \
    --source-dataset ../datasets/so100/svla_so100_pickplace

python scripts/postprocess_omx_dataset.py \
    --dataset-dir ../datasets/omx/omx_stacking \
    --task "stack blocks" \
    --source-dataset ../datasets/so100/svla_so100_stacking

python scripts/postprocess_omx_dataset.py \
    --dataset-dir ../datasets/omx/omx_sorting \
    --task "sort objects" \
    --source-dataset ../datasets/so100/svla_so100_sorting
```

### 2.4 데이터셋 병합

```bash
python scripts/merge_omx_datasets.py \
    --sources ../datasets/omx/omx_pickplace ../datasets/omx/omx_stacking ../datasets/omx/omx_sorting \
    --output ../datasets/omx/omx_combined
```

결과: `omx_combined/` - 158 에피소드, 78,300 프레임, 3 tasks

### 2.5 비디오 H264 변환 (필수!)

원본 비디오가 **AV1 코덱**인데, **decord가 AV1을 지원하지 않습니다**.
반드시 H264로 재인코딩해야 합니다.

```bash
# 각 소스 비디오를 H264로 변환
for dataset in svla_so100_pickplace svla_so100_stacking svla_so100_sorting; do
    for cam in observation.images.top observation.images.wrist; do
        src="datasets/so100/${dataset}/videos/${cam}/chunk-000/file-000.mp4"
        dst="datasets/so100/${dataset}/videos/${cam}/chunk-000/file-000-h264.mp4"
        ffmpeg -i "$src" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "$dst"
    done
done
```

그 후 omx 데이터셋의 심링크를 H264 파일로 업데이트해야 합니다:

```bash
# omx_combined 및 개별 데이터셋의 심링크를 h264 버전으로 갱신
for omx_dir in datasets/omx/omx_*/; do
    for cam in observation.images.top observation.images.wrist; do
        video_dir="${omx_dir}videos/${cam}/chunk-000/"
        [ -d "$video_dir" ] || continue
        for link in "${video_dir}"episode_*.mp4; do
            target=$(readlink "$link")
            new_target="${target/file-000.mp4/file-000-h264.mp4}"
            if [ -f "$new_target" ] || [ -f "$(dirname "$link")/$new_target" ]; then
                ln -sf "$new_target" "$link"
            fi
        done
    done
done
```

검증:
```bash
python -c "
import decord
vr = decord.VideoReader('datasets/omx/omx_combined/videos/observation.images.top/chunk-000/episode_000000.mp4')
print(f'프레임 수: {len(vr)}, shape: {vr[0].shape}')
"
```

---

## 3. GR00T-Dreams-IDM 코드 수정사항

총 3개 파일이 수정되었습니다:

### 3.1 `scripts/idm_training.py` (핵심)

Config dataclass에 3개 필드 추가:

```python
gradient_checkpointing: bool = False
"""Enable gradient checkpointing to save GPU memory."""

gradient_accumulation_steps: int = 1
"""Number of gradient accumulation steps."""

tune_vision_tower: bool = True
"""Whether to fine-tune SiGLIP2 vision tower. Set False for <16GB VRAM."""
```

TrainingArguments에서 config 값 사용:
```python
gradient_checkpointing=config.gradient_checkpointing,
gradient_accumulation_steps=config.gradient_accumulation_steps,
```

모델 로딩 후 vision tower freeze 로직:
```python
if not config.tune_vision_tower:
    print("Freezing vision tower to save VRAM")
    model.action_head.set_trainable_parameters(
        tune_multi_projector=True, tune_diffusion_model=True,
        tune_vision_tower=False, tune_mm_projector=True, tune_vl_mixing=True,
    )
```

### 3.2 `gr00t/model/idm.py`

`prepare_input`의 `to_device_with_maybe_dtype`에 non-tensor 타입 체크 추가:
```python
def to_device_with_maybe_dtype(x):
    if not isinstance(x, torch.Tensor):
        return x
    if torch.is_floating_point(x):
        return x.to(self.device, dtype=self.action_head.dtype)
    else:
        return x.to(self.device)
```

### 3.3 `gr00t/utils/video.py`

`get_frames_by_timestamps`에 `pyav` 백엔드 추가 (lines 96-112):
- decord 대안으로 AV1 코덱 지원
- H264 변환 후에는 decord를 사용하므로 fallback 용도

---

## 4. 학습 실행

### 4.1 환경 설정 (Blackwell GPU / RTX 50xx 시리즈)

RTX 50xx (Blackwell, compute capability 12.0, sm_120)는 **PyTorch stable에서 미지원**됩니다.
반드시 PyTorch nightly + CUDA 12.8 조합을 사용해야 합니다.

#### 검증된 환경 (RTX 5070 Laptop, 2026-03-03 기준)

| 항목 | 버전 | 비고 |
|------|------|------|
| GPU | RTX 5070 Laptop (sm_120) | Blackwell 아키텍처 |
| NVIDIA Driver | 580.126.09 | 560+ 필요 |
| System CUDA Toolkit | 12.6 | PyTorch가 자체 CUDA 런타임 번들 |
| PyTorch | 2.12.0.dev20260302+cu128 | **nightly 필수** |
| Python | 3.11.14 | 3.11 권장 (3.12도 가능) |
| cuDNN | 9.19.0 | PyTorch에 번들됨 |

#### 새 머신 설정 단계

```bash
cd /home/lambda/claude/GR00T-Dreams-IDM

# 1. Python 3.11 venv 생성 (3.12도 가능하지만 3.11이 호환성 더 좋음)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. PyTorch nightly 설치 (Blackwell sm_120 지원 필수)
#    *** stable 버전 (pip install torch)은 sm_120 미지원 → CUDA error 발생 ***
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. 프로젝트 의존성 설치
pip install -r requirements.txt

# 4. Blackwell 호환성 검증
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
cap = torch.cuda.get_device_capability(0)
print(f'Compute capability: {cap}')
assert cap[0] >= 12, 'Blackwell GPU가 아닙니다'
assert 'cu12' in torch.version.cuda.replace('.',''), 'CUDA 12.x 필요'
# sm_120 커널 테스트
x = torch.randn(2, 2, device='cuda')
y = x @ x.T
print(f'CUDA 연산 테스트: OK ({y.shape})')
print('Blackwell 환경 검증 완료!')
"
```

#### Blackwell 주요 문제 및 해결

| 문제 | 증상 | 해결 |
|------|------|------|
| PyTorch stable 사용 | `CUDA error: no kernel image` 또는 `sm_120 not supported` | `--pre` 플래그로 nightly 설치 |
| cu121 변형 설치 | CUDA 버전 불일치로 런타임 에러 | `cu128` 인덱스 URL 사용 |
| torchvision.io.VideoReader 없음 | `AttributeError: no attribute 'VideoReader'` | nightly에서 제거됨. `--video-backend decord` 사용 |
| CUDA toolkit 버전 낮음 | Driver/toolkit 불일치 경고 | System CUDA는 무관, PyTorch 번들 CUDA 사용 |
| Python 3.13+ | 일부 C 확장 호환 문제 | Python 3.11 또는 3.12 사용 |

> **팁**: NVIDIA Driver 560+ 만 설치되어 있으면, System CUDA Toolkit 버전은 무관합니다.
> PyTorch nightly는 자체 CUDA 12.8 런타임을 번들링합니다.

### 4.2 RTX 5070 Laptop (8GB VRAM)

```bash
cd /home/lambda/claude/GR00T-Dreams-IDM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/idm_training.py \
    --dataset-path /home/lambda/claude/datasets/omx/omx_combined \
    --data-config omx \
    --output-dir /home/lambda/claude/idm_output \
    --num-gpus 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --gradient-checkpointing \
    --no-tune-vision-tower \
    --max-steps 5000 \
    --save-steps 500 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 2 \
    --report-to tensorboard \
    --video-backend decord \
    --embodiment-tag new_embodiment
```

- GPU 메모리: 7.6 / 8.1 GB (94%)
- 속도: ~1.25 it/s
- 소요시간: ~65분
- 최종 loss: 0.45

### 4.3 RTX 5090 (32GB VRAM) - 권장

```bash
cd /home/lambda/claude/GR00T-Dreams-IDM

.venv/bin/python scripts/idm_training.py \
    --dataset-path /path/to/datasets/omx/omx_combined \
    --data-config omx \
    --output-dir /path/to/idm_output_5090 \
    --num-gpus 1 \
    --batch-size 16 \
    --gradient-accumulation-steps 1 \
    --tune-vision-tower \
    --max-steps 10000 \
    --save-steps 1000 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 8 \
    --report-to tensorboard \
    --video-backend decord \
    --embodiment-tag new_embodiment
```

**5090 장점**:
- Vision tower 전체 학습 가능 (도메인 적응 → 성능 향상 기대)
- batch_size 16 → GPU 활용률 극대화
- gradient checkpointing 불필요 → 속도 향상
- 더 긴 학습 (10000 steps) 가능
- 예상 속도: ~5-10x 빠름 (7-15분)

### 4.4 모니터링

```bash
# TensorBoard
tensorboard --logdir /path/to/idm_output/runs --bind_all

# Python으로 loss 확인
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
logdir = glob.glob('/path/to/idm_output/runs/*/')[0]
ea = EventAccumulator(logdir)
ea.Reload()
for e in ea.Scalars('train/loss'):
    if e.step % 500 == 0:
        print(f'step={e.step:5d}  loss={e.value:.4f}')
"
```

---

## 5. 추론 (Inference)

```python
import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP

# 1. 모델 로드
model = instantiate(OmegaConf.load("IDM_dump/base.yaml"))
state_dict = load_file("/path/to/idm_output/model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.eval()
model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"
model = model.to("cuda")

# 2. 데이터 준비 (collator 방식: images/view_ids는 concat, 나머지는 stack)
data_config_cls = DATA_CONFIG_MAP["omx"]
dataset = LeRobotSingleDataset(
    dataset_path="/path/to/datasets/omx/omx_combined",
    modality_configs=data_config_cls.modality_config(),
    transforms=data_config_cls.transform(),
    embodiment_tag=EmbodimentTag("new_embodiment"),
    video_backend="decord",
)

sample = dataset[0]
batch = {}
for k, v in sample.items():
    if k in ["images", "view_ids"]:
        batch[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
    elif isinstance(v, torch.Tensor):
        batch[k] = v.unsqueeze(0)
    elif isinstance(v, np.ndarray):
        batch[k] = torch.from_numpy(np.stack([v]))
    elif isinstance(v, (int, float)):
        batch[k] = torch.tensor([v])

# 3. 추론
with torch.no_grad():
    outputs = model.get_action(batch)
action = outputs["action_pred"]  # shape: [1, 16, 32]
print(f"Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")
```

> **주의**: `images`와 `view_ids`는 `unsqueeze(0)` 하면 안 됩니다.
> DataCollator가 `np.concatenate`를 사용하므로 배치 차원 없이 `[num_views, C, H, W]` 형태여야 합니다.

---

## 6. 트러블슈팅

### 6.1 BLAS 스레드 폭발

**증상**: multiprocessing + scipy 사용 시 load average가 CPU 코어의 10배 이상
**원인**: OpenBLAS가 워커당 MAX_THREADS(24)개 스레드 생성
**해결**: 스크립트 최상단에 환경변수 설정 (numpy/scipy import 전)

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

### 6.2 decord AV1 미지원

**증상**: `DECORDError: cannot find video stream with wanted index: -1`
**원인**: decord는 AV1 코덱 미지원
**해결**: ffmpeg로 H264 재인코딩 (섹션 2.5 참고)
**대안**: `--video-backend pyav` (느리지만 AV1 지원)

### 6.3 CUDA OOM

**증상**: `torch.OutOfMemoryError: CUDA out of memory`
**해결** (8GB VRAM):
1. `--batch-size 1 --gradient-accumulation-steps 8`
2. `--gradient-checkpointing`
3. `--no-tune-vision-tower` (SiGLIP2 freeze → ~3GB 절약)
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 6.4 Blackwell GPU (RTX 50xx) PyTorch 설정 실패

**증상**: `CUDA error: no kernel image is available for execution on the device` 또는 `RuntimeError: sm_120 is not supported`
**원인**: PyTorch stable (2.5, 2.6 등)은 Blackwell(sm_120) 미지원. cu121 변형도 불가.
**해결**:
```bash
# 기존 PyTorch 제거 후 nightly+cu128 설치
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
**검증**: `python -c "import torch; print(torch.cuda.get_device_capability(0))"` → `(12, 0)` 확인

### 6.5 torchvision.io.VideoReader 없음

**증상**: `AttributeError: module 'torchvision.io' has no attribute 'VideoReader'`
**원인**: PyTorch nightly에서 VideoReader 제거됨
**해결**: `--video-backend decord` 또는 `--video-backend pyav` 사용

### 6.6 추론 시 TypeError (non-tensor)

**증상**: `TypeError: is_floating_point(): argument must be Tensor, not int`
**원인**: `embodiment_id`가 int로 전달됨
**해결**: `gr00t/model/idm.py`의 `to_device_with_maybe_dtype`에 타입 체크 추가 (섹션 3.2)

---

## 7. 학습 결과 (RTX 5070, 8GB, 2026-03-03)

### Loss 곡선

```
Step  100: 1.368 (warmup)
Step  500: 0.946
Step 1000: 0.709
Step 1500: 0.616
Step 2000: 0.675
Step 2500: 0.576
Step 3000: 0.546
Step 3500: 0.481
Step 4000: 0.447
Step 4500: 0.449
Step 5000: 0.464 (최종)
```

- Loss 감소율: 66.6% (1.35 → 0.45)
- 분기별 단조 감소 확인
- Gradient norm 안정 (평균 1.62, 최대 5.1)
- 학습 설정: vision tower 동결, batch=1, grad_accum=8

### 검증 결과 (UltraQA 4/4 PASS)

| 항목 | 결과 |
|------|------|
| 모델 무결성 | 2.55GB, 637M params, NaN/Inf 없음 |
| 수렴 분석 | 분기별 단조 감소, grad norm 안정 |
| 추론 테스트 | 5개 샘플 모두 유효한 action 예측 |
| 체크포인트 | 8개 모두 정상 로딩 |

---

## 8. 다른 PC / 새 세션에서 이어 시작하기

### 8.1 빠른 경로: HuggingFace에서 다운로드 후 바로 학습 (권장)

변환 완료된 데이터셋이 HuggingFace에 있으므로, 새 PC에서는 4단계만 수행하면 됩니다.

```bash
# ──── Step 1: 코드 준비 ────
git clone <GR00T-Dreams-IDM repo URL>
cd GR00T-Dreams-IDM

# ──── Step 2: 환경 설정 (Blackwell GPU인 경우) ────
python3.11 -m venv .venv
source .venv/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install decord  # 비디오 백엔드

# 환경 검증
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
x = torch.randn(2,2,device='cuda'); print('CUDA OK')
"

# ──── Step 3: 데이터셋 다운로드 + 심링크 ────
huggingface-cli login
huggingface-cli download Yuseok/omx_combined_idm \
    --repo-type dataset \
    --local-dir ../datasets/omx/omx_combined

cd ../datasets/omx/omx_combined
python setup_symlinks.py   # 316 symlinks 생성
cd -  # GR00T-Dreams-IDM으로 복귀

# 검증
python -c "
import decord
vr = decord.VideoReader('../datasets/omx/omx_combined/videos/observation.images.top/chunk-000/episode_000000.mp4')
print(f'OK: {len(vr)} frames, {vr[0].shape}')
"

# ──── Step 4: 학습 실행 ────
# RTX 5090 (32GB) - 전체 모델 학습
.venv/bin/python scripts/idm_training.py \
    --dataset-path ../datasets/omx/omx_combined \
    --data-config omx \
    --output-dir ../idm_output \
    --num-gpus 1 \
    --batch-size 16 \
    --tune-vision-tower \
    --max-steps 10000 \
    --save-steps 1000 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 8 \
    --report-to tensorboard \
    --video-backend decord \
    --embodiment-tag new_embodiment

# RTX 5070 (8GB) - 메모리 최적화 학습
# --batch-size 1 --gradient-accumulation-steps 8 \
# --gradient-checkpointing --no-tune-vision-tower \
# --dataloader-num-workers 2
```

### 8.2 전체 파이프라인: 원본 데이터부터 변환하는 경우

SO-100 원본 데이터에서부터 변환이 필요한 경우:

1. **섹션 2.1**: SO-100 데이터셋 3개 다운로드 (lerobot HuggingFace)
2. **섹션 2.2**: SO-100 → OMX kinematic 변환 (`convert_so100_to_omx.py`)
3. **섹션 2.3**: 후처리 - scalar → array columns (`postprocess_omx_dataset.py`)
4. **섹션 2.4**: 3개 데이터셋 병합 (`merge_omx_datasets.py`)
5. **섹션 2.5**: AV1 → H264 비디오 변환 (ffmpeg)
6. **섹션 4**: 학습 실행

> 변환 스크립트는 `pseudo-project/scripts/`에 있습니다.

### 8.3 필수 Git 저장소

| 저장소 | 용도 | 핵심 파일 |
|--------|------|-----------|
| GR00T-Dreams-IDM | IDM 모델 학습/추론 코드 | `scripts/idm_training.py`, `gr00t/model/idm.py` |
| pseudo-project | 변환 스크립트, 문서 | `scripts/convert_so100_to_omx.py`, `docs/` |

### 8.4 HuggingFace 리소스

| 리소스 | Repo ID | 접근 |
|--------|---------|------|
| 변환 완료 데이터셋 | `Yuseok/omx_combined_idm` | Private (인증 필요) |
| SO-100 pickplace 원본 | `lerobot/svla_so100_pickplace` | Public |
| SO-100 stacking 원본 | `lerobot/svla_so100_stacking` | Public |
| SO-100 sorting 원본 | `lerobot/svla_so100_sorting` | Public |

### 8.5 Claude에게 전달할 컨텍스트

새 세션에서 다음과 같이 요청하세요:

> `docs/IDM_TRAINING_COMPLETE_GUIDE.md` 파일을 읽어줘.
> HuggingFace에서 `Yuseok/omx_combined_idm` 데이터셋을 다운로드하고,
> IDM 모델을 학습해줘. GPU는 RTX 5090 32GB야.

---

## 9. 향후 개선 방향

1. **더 많은 데이터**: `community_dataset_v1` (100+ 에피소드) 추가 변환
2. **Vision tower 학습**: 5090에서 전체 모델 학습 → 도메인 적응 효과 검증
3. **Longer training**: 10000-20000 steps로 수렴 확인
4. **Learning rate sweep**: 5e-5, 1e-4, 2e-4 비교
5. **Evaluation**: 실제 로봇에서 pseudo-labeling 파이프라인 테스트
