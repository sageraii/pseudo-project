#!/usr/bin/env python3
"""
OMX 데이터셋 병합 스크립트

기존 OMX 데이터셋 (per-episode format)과 변환된 megamix 데이터셋을
하나의 per-episode format 데이터셋으로 병합합니다.

Pipeline:
    1. 기존 OMX 데이터 복사 (episodes 0-N)
    2. 변환된 megamix 데이터 분할 + 리넘버링 (episodes N+1 ~ N+M)
    3. megamix packed 비디오 → per-episode 비디오 추출 (ffmpeg -c copy)
    4. 통합 메타데이터 생성 (info.json, stats.json, modality.json)

Usage:
    python scripts/merge_omx_datasets.py \
        --base-dataset ~/claude/datasets/omx/omx_combined \
        --new-dataset ~/claude/datasets/omx/omx_megamix_converted \
        --new-source-video ~/claude/datasets/so101_megamix \
        --output-dir ~/claude/datasets/omx/omx_merged
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def copy_base_dataset(base_dir: Path, output_dir: Path):
    """기존 OMX 데이터셋의 parquet과 비디오를 복사."""
    data_src = base_dir / "data" / "chunk-000"
    data_dst = output_dir / "data" / "chunk-000"
    data_dst.mkdir(parents=True, exist_ok=True)

    parquets = sorted(data_src.glob("episode_*.parquet"))
    n_episodes = len(parquets)
    print(f"  기존 데이터: {n_episodes} episodes")

    for pf in parquets:
        shutil.copy2(pf, data_dst / pf.name)

    # Copy videos (observation.images.top only)
    vid_src = base_dir / "videos" / "observation.images.top" / "chunk-000"
    vid_dst = output_dir / "videos" / "observation.images.top" / "chunk-000"
    vid_dst.mkdir(parents=True, exist_ok=True)

    if vid_src.exists():
        vids = sorted(vid_src.glob("episode_*.mp4"))
        print(f"  기존 비디오: {len(vids)} files")
        for vf in vids:
            dst_path = vid_dst / vf.name
            src_real = vf.resolve()
            try:
                os.link(src_real, dst_path)
            except OSError:
                shutil.copy2(src_real, dst_path)

    return n_episodes


def split_and_renumber_megamix(
    converted_dir: Path,
    output_dir: Path,
    episode_offset: int,
):
    """변환된 megamix 단일 parquet을 per-episode parquet으로 분할 + 리넘버링."""
    data_dir = converted_dir / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        print(f"[ERROR] No parquet files in {data_dir}")
        sys.exit(1)

    df_all = pd.concat([pd.read_parquet(pf) for pf in parquet_files], ignore_index=True)

    # Handle column format mismatch: individual columns → aggregated arrays
    if "observation.state" not in df_all.columns and "state.shoulder_pan" in df_all.columns:
        # Correct joint order (must match existing OMX dataset)
        joint_order = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                        "wrist_flex", "wrist_roll", "gripper"]
        state_cols = [f"state.{j}" for j in joint_order]
        action_cols = [f"action.{j}" for j in joint_order]
        print(f"  컬럼 변환: {len(state_cols)} state + {len(action_cols)} action → 집계 배열")

        # Aggregate into array columns (matching existing OMX format)
        df_all["observation.state"] = df_all[state_cols].values.tolist()
        df_all["action"] = df_all[action_cols].values.tolist()
        df_all = df_all.drop(columns=state_cols + action_cols)

    episodes = sorted(df_all["episode_index"].unique())
    n_episodes = len(episodes)
    total_frames = len(df_all)
    print(f"  변환된 데이터: {n_episodes} episodes, {total_frames} frames")

    data_dst = output_dir / "data" / "chunk-000"
    data_dst.mkdir(parents=True, exist_ok=True)

    # Calculate global index offset from existing data
    existing_parquets = sorted(data_dst.glob("episode_*.parquet"))
    global_index_offset = 0
    if existing_parquets:
        last_df = pd.read_parquet(existing_parquets[-1])
        global_index_offset = int(last_df["index"].max()) + 1

    for old_ep in episodes:
        new_ep = old_ep + episode_offset
        df_ep = df_all[df_all["episode_index"] == old_ep].copy()
        n_frames = len(df_ep)

        df_ep["episode_index"] = new_ep
        df_ep["frame_index"] = range(n_frames)
        df_ep["index"] = range(global_index_offset, global_index_offset + n_frames)
        global_index_offset += n_frames

        out_path = data_dst / f"episode_{new_ep:06d}.parquet"
        df_ep.to_parquet(out_path, index=False)

    print(f"  리넘버링: episodes {episode_offset}-{episode_offset + n_episodes - 1}")
    return n_episodes


def extract_per_episode_videos(
    source_video_dir: Path,
    output_dir: Path,
    episode_offset: int,
    n_episodes: int,
):
    """Packed 비디오에서 per-episode 비디오를 ffmpeg -c copy로 추출."""
    vid_dst = output_dir / "videos" / "observation.images.top" / "chunk-000"
    vid_dst.mkdir(parents=True, exist_ok=True)

    packed_video = source_video_dir / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
    if not packed_video.exists():
        print(f"[ERROR] Packed video not found: {packed_video}")
        sys.exit(1)

    # Read episode metadata for timestamps
    ep_meta_path = source_video_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_meta_path.exists():
        print(f"[ERROR] Episode metadata not found: {ep_meta_path}")
        sys.exit(1)

    ep_meta = pd.read_parquet(ep_meta_path)
    ep_meta = ep_meta.drop_duplicates("episode_index").sort_values("episode_index").reset_index(drop=True)

    ts_from_col = "videos/observation.images.top/from_timestamp"
    ts_to_col = "videos/observation.images.top/to_timestamp"

    print(f"  비디오 추출: {n_episodes} episodes from {packed_video.name}")
    t_start = time.time()
    errors = 0

    for i in range(n_episodes):
        row = ep_meta[ep_meta["episode_index"] == i]
        if len(row) == 0:
            print(f"    [WARN] Episode {i} not found in metadata, skipping")
            errors += 1
            continue

        row = row.iloc[0]
        from_ts = float(row[ts_from_col])
        to_ts = float(row[ts_to_col])
        new_ep = i + episode_offset
        out_path = vid_dst / f"episode_{new_ep:06d}.mp4"

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{from_ts:.6f}",
            "-to", f"{to_ts:.6f}",
            "-i", str(packed_video),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            stderr = result.stderr.decode()[:200]
            print(f"    [ERROR] Episode {new_ep}: {stderr}")
            errors += 1

        if (i + 1) % 50 == 0 or i == n_episodes - 1:
            elapsed = time.time() - t_start
            eps = (i + 1) / max(elapsed, 0.01)
            print(f"    [{i+1}/{n_episodes}] {elapsed:.1f}s ({eps:.1f} ep/s)")

    elapsed = time.time() - t_start
    print(f"  비디오 추출 완료: {elapsed:.1f}s, errors: {errors}")
    return errors


def compute_combined_stats(output_dir: Path):
    """통합 데이터셋의 정규화 통계 계산."""
    data_dir = output_dir / "data" / "chunk-000"
    parquets = sorted(data_dir.glob("episode_*.parquet"))

    all_states, all_actions = [], []
    for pf in parquets:
        df = pd.read_parquet(pf)
        all_states.append(np.stack(df["observation.state"].values))
        all_actions.append(np.stack(df["action"].values))

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    return {"observation.state": _stats(states), "action": _stats(actions)}


def generate_metadata(output_dir: Path, total_episodes: int, total_frames: int, n_tasks: int):
    """통합 메타데이터 생성."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ]

    info = {
        "codebase_version": "v3.0",
        "robot_type": "omx",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": n_tasks,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32", "shape": [6],
                "names": joint_names, "fps": 30.0,
            },
            "observation.state": {
                "dtype": "float32", "shape": [6],
                "names": joint_names, "fps": 30.0,
            },
            "observation.images.top": {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": 30.0, "video.height": 480, "video.width": 640,
                    "video.channels": 3, "video.codec": "h264",
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": 30.0},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json — copy from base dataset (same format for all OMX datasets)
    # Intentionally NOT generated from scratch to avoid key naming mismatches

    # trajectory_lengths.json
    data_dir = output_dir / "data" / "chunk-000"
    traj_lengths = []
    for ep in range(total_episodes):
        pf = data_dir / f"episode_{ep:06d}.parquet"
        if pf.exists():
            traj_lengths.append(len(pd.read_parquet(pf)))
        else:
            traj_lengths.append(0)

    with open(meta_dir / "trajectory_lengths.json", "w") as f:
        json.dump(traj_lengths, f)

    # episodes.jsonl (required by LeRobotSingleDataset)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, length in enumerate(traj_lengths):
            f.write(json.dumps({"episode_index": i, "length": length}) + "\n")

    print(f"  메타데이터 생성 완료")


def main():
    parser = argparse.ArgumentParser(description="OMX 데이터셋 병합")
    parser.add_argument("--base-dataset", required=True,
                        help="기존 OMX 데이터셋 경로 (per-episode format)")
    parser.add_argument("--new-dataset", required=True,
                        help="변환된 megamix 데이터셋 경로 (convert_so100_to_omx 출력)")
    parser.add_argument("--new-source-video", required=True,
                        help="megamix 원본 비디오 경로 (packed format)")
    parser.add_argument("--output-dir", required=True,
                        help="통합 데이터셋 출력 경로")
    args = parser.parse_args()

    base_dir = Path(args.base_dataset)
    new_dir = Path(args.new_dataset)
    source_vid_dir = Path(args.new_source_video)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("OMX 데이터셋 병합")
    print("=" * 60)

    # Step 1: Copy base dataset
    print(f"\n[1/5] 기존 데이터셋 복사: {base_dir}")
    n_base = copy_base_dataset(base_dir, output_dir)

    # Step 2: Split and renumber megamix
    print(f"\n[2/5] 변환된 데이터 분할 + 리넘버링 (offset={n_base})")
    n_new = split_and_renumber_megamix(new_dir, output_dir, n_base)

    # Step 3: Extract per-episode videos
    print(f"\n[3/5] 비디오 추출 (ffmpeg -c copy)")
    errors = extract_per_episode_videos(source_vid_dir, output_dir, n_base, n_new)

    # Step 4: Compute stats
    print(f"\n[4/5] 통계 계산")
    stats = compute_combined_stats(output_dir)
    stats_path = output_dir / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  stats.json 저장 완료")

    # Step 5: Generate metadata
    total_episodes = n_base + n_new
    data_dir = output_dir / "data" / "chunk-000"
    total_frames = sum(
        len(pd.read_parquet(pf))
        for pf in sorted(data_dir.glob("episode_*.parquet"))
    )
    print(f"\n[5/5] 메타데이터 생성")
    base_info = json.loads((base_dir / "meta" / "info.json").read_text())
    n_tasks = base_info.get("total_tasks", 3) + 8  # megamix has 8 tasks
    generate_metadata(output_dir, total_episodes, total_frames, n_tasks)

    # Copy modality.json from base dataset (avoids key naming mismatches)
    import shutil as _shutil
    _shutil.copy2(base_dir / "meta" / "modality.json", output_dir / "meta" / "modality.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"병합 완료")
    print(f"{'='*60}")
    print(f"  기존: {n_base} episodes")
    print(f"  추가: {n_new} episodes")
    print(f"  합계: {total_episodes} episodes, {total_frames} frames")
    print(f"  출력: {output_dir}")
    if errors > 0:
        print(f"  [WARN] 비디오 추출 오류: {errors}건")


if __name__ == "__main__":
    main()
