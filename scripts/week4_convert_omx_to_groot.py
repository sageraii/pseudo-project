"""
Week 4: OMX HuggingFace 데이터 → GR00T LeRobot v2 포맷 변환

OMX LeRobot 데이터(HuggingFace datasets)를 GR00T가 요구하는
LeRobot v2 포맷(parquet + mp4 + meta/modality.json)으로 변환합니다.

Usage:
    python scripts/week4_convert_omx_to_groot.py \
        --hf-repo-id $HF_USER/omx_pick \
        --output-dir data/omx_groot_v2/pick \
        --task-description "Pick up the object"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가 (utils 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.omx_constants import OMX_DOF as OMX_STATE_DIM, OMX_JOINT_NAMES

OMX_ACTION_DIM = OMX_STATE_DIM


def create_modality_json(output_dir: Path):
    """GR00T LeRobot v2가 요구하는 meta/modality.json 생성"""
    modality = {
        "video": {"cam1": {"resolution": [224, 224], "num_frames": 1}},
        "state": {"joints": {"dim": OMX_STATE_DIM}},
        "action": {"joints": {"dim": OMX_ACTION_DIM, "action_horizon": 16}},
        "annotation": {"task": {"type": "string"}},
    }

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)
    print(f"  Created: {meta_dir / 'modality.json'}")


def convert_hf_to_groot_v2(hf_repo_id: str, output_dir: Path, task_description: str):
    """HuggingFace 데이터셋을 GR00T LeRobot v2로 변환"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        return

    print(f"Loading dataset: {hf_repo_id}")
    ds = load_dataset(hf_repo_id, split="train")
    print(f"  Episodes: {len(set(ds['episode_index']))}")
    print(f"  Total frames: {len(ds)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. modality.json 생성
    create_modality_json(output_dir)

    # 2. 에피소드별 parquet 변환
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pip install pandas pyarrow")
        return

    episodes = sorted(set(ds["episode_index"]))
    parquet_dir = output_dir / "data"
    parquet_dir.mkdir(exist_ok=True)

    for ep_idx in episodes:
        ep_data = ds.filter(lambda x: x["episode_index"] == ep_idx)

        records = []
        for i, row in enumerate(ep_data):
            # OMX 6-dim state/action 추출
            state = row.get("observation.state", [0.0] * OMX_STATE_DIM)
            action = row.get("action", [0.0] * OMX_ACTION_DIM)

            record = {
                "episode_index": ep_idx,
                "frame_index": i,
                "timestamp": row.get("timestamp", i / 15.0),
                "task_description": task_description,
            }

            # State를 개별 joint로 분리
            for j, name in enumerate(OMX_JOINT_NAMES):
                record[f"state.{name}"] = float(state[j]) if j < len(state) else 0.0
                record[f"action.{name}"] = float(action[j]) if j < len(action) else 0.0

            records.append(record)

        df = pd.DataFrame(records)
        parquet_path = parquet_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

    print(f"  Parquet files: {len(episodes)} episodes -> {parquet_dir}/")

    # 3. 비디오 추출 (mp4)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    has_images = any(
        col.startswith("observation.images") for col in ds.column_names
    )
    if has_images:
        print("  Extracting videos from image columns...")
        try:
            import cv2

            for ep_idx in episodes:
                ep_data = ds.filter(lambda x: x["episode_index"] == ep_idx)
                video_path = video_dir / f"cam1_episode_{ep_idx:06d}.mp4"

                # 이미지 컬럼 찾기
                img_cols = [c for c in ds.column_names if "images" in c and "cam" in c]
                if not img_cols:
                    img_cols = [c for c in ds.column_names if "images" in c]
                if not img_cols:
                    continue

                img_col = img_cols[0]
                first_img = ep_data[0][img_col]
                if hasattr(first_img, "size"):  # PIL Image
                    h, w = first_img.size[1], first_img.size[0]
                else:
                    h, w = first_img.shape[:2]

                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))

                for row in ep_data:
                    img = row[img_col]
                    if hasattr(img, "convert"):  # PIL
                        img = np.array(img.convert("RGB"))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    writer.write(img_bgr)

                writer.release()

            print(f"  Videos: {len(episodes)} episodes -> {video_dir}/")
        except ImportError:
            print("  WARNING: cv2 not available. Skipping video extraction.")
            print("  Install: pip install opencv-python")
    else:
        print("  WARNING: No image columns found. Videos must be added separately.")

    # 4. 메타데이터
    meta = {
        "source": hf_repo_id,
        "robot": "ROBOTIS OMX",
        "dof": OMX_STATE_DIM,
        "fps": 15,
        "num_episodes": len(episodes),
        "num_frames": len(ds),
        "task": task_description,
        "joint_names": OMX_JOINT_NAMES,
        "format": "GR00T LeRobot v2 (parquet + mp4 + modality.json)",
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {output_dir / 'meta' / 'info.json'}")

    print(f"\nConversion complete: {output_dir}")
    print(f"  Structure:")
    print(f"    {output_dir}/data/episode_*.parquet")
    print(f"    {output_dir}/videos/cam1_episode_*.mp4")
    print(f"    {output_dir}/meta/modality.json")
    print(f"    {output_dir}/meta/info.json")


def main():
    parser = argparse.ArgumentParser(description="Convert OMX HF dataset to GR00T LeRobot v2")
    parser.add_argument("--hf-repo-id", required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--task-description", required=True, help="Task description string")
    args = parser.parse_args()

    convert_hf_to_groot_v2(
        hf_repo_id=args.hf_repo_id,
        output_dir=Path(args.output_dir),
        task_description=args.task_description,
    )


if __name__ == "__main__":
    main()
