#!/usr/bin/env python3
"""Merge multiple OMX datasets into a single combined dataset for IDM training.

Renumbers episode indices to be continuous across all source datasets.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd


def merge_datasets(source_dirs: list[str], output_dir: str, task_descriptions: dict[str, str]):
    """Merge multiple post-processed OMX datasets into one."""
    output_dir = os.path.abspath(output_dir)
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    all_episodes = []
    all_tasks = []
    global_ep_idx = 0
    global_frame_idx = 0
    task_idx = 0

    for src_dir in source_dirs:
        src_dir = os.path.abspath(src_dir)
        src_name = os.path.basename(src_dir)
        src_data = os.path.join(src_dir, "data", "chunk-000")
        src_meta = os.path.join(src_dir, "meta")

        # Read episodes.jsonl
        with open(os.path.join(src_meta, "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]

        task_desc = task_descriptions.get(src_name, f"task from {src_name}")
        all_tasks.append({"task_index": task_idx, "task": task_desc})

        print(f"\n--- {src_name}: {len(episodes)} episodes ---")

        for ep in episodes:
            old_ep_idx = ep["episode_index"]
            ep_length = ep["length"]

            # Read episode parquet
            src_parquet = os.path.join(src_data, f"episode_{old_ep_idx:06d}.parquet")
            if not os.path.exists(src_parquet):
                print(f"  [WARN] Missing {src_parquet}, skipping")
                continue

            df = pd.read_parquet(src_parquet)

            # Renumber indices
            df["episode_index"] = global_ep_idx
            df["task_index"] = task_idx
            df["index"] = range(global_frame_idx, global_frame_idx + len(df))

            # Write to output
            dst_parquet = os.path.join(data_dir, f"episode_{global_ep_idx:06d}.parquet")
            df.to_parquet(dst_parquet, index=False)

            all_episodes.append({"episode_index": global_ep_idx, "length": ep_length})
            global_frame_idx += len(df)
            global_ep_idx += 1

        # Setup video symlinks
        src_video_base = os.path.join(src_dir, "videos")
        dst_video_base = os.path.join(output_dir, "videos")
        for video_key in ["observation.images.top", "observation.images.wrist"]:
            src_chunk = os.path.join(src_video_base, video_key, "chunk-000")
            if not os.path.exists(src_chunk):
                continue
            dst_chunk = os.path.join(dst_video_base, video_key, "chunk-000")
            os.makedirs(dst_chunk, exist_ok=True)
            # Find any mp4 file (either file-000.mp4 or episode_*.mp4 symlinks)
            src_mp4s = [f for f in os.listdir(src_chunk) if f.endswith(".mp4")]
            if not src_mp4s:
                continue
            # Resolve to actual file (follow symlinks)
            actual_mp4 = os.path.realpath(os.path.join(src_chunk, src_mp4s[0]))
            # Create symlinks for each episode from this source
            for ep in episodes:
                # Map old episode index to new global index
                new_idx = ep["episode_index"] + (global_ep_idx - len(episodes))
                ep_mp4 = os.path.join(dst_chunk, f"episode_{new_idx:06d}.mp4")
                if not os.path.exists(ep_mp4):
                    os.symlink(actual_mp4, ep_mp4)

        task_idx += 1
        print(f"  Merged {len(episodes)} episodes (global {global_ep_idx - len(episodes)}..{global_ep_idx - 1})")

    n_episodes = len(all_episodes)
    total_frames = global_frame_idx

    # Compute combined stats
    print(f"\nComputing combined statistics...")
    all_dfs = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".parquet"):
            all_dfs.append(pd.read_parquet(os.path.join(data_dir, f)))
    combined = pd.concat(all_dfs, ignore_index=True)

    stats = {}
    for key in ["observation.state", "action"]:
        data = np.vstack([np.asarray(x, dtype=np.float32) for x in combined[key]])
        stats[key] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.percentile(data, 1, axis=0).tolist(),
            "q99": np.percentile(data, 99, axis=0).tolist(),
        }

    # Write meta files
    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")

    # Copy modality.json from first source (same for all OMX datasets)
    shutil.copy2(
        os.path.join(source_dirs[0], "meta", "modality.json"),
        os.path.join(meta_dir, "modality.json"),
    )

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "omx",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": task_idx,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": [f"main_{j}" for j in joint_names], "fps": 30.0},
            "observation.state": {"dtype": "float32", "shape": [6], "names": [f"main_{j}" for j in joint_names], "fps": 30.0},
            "observation.images.top": {
                "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"],
                "info": {"video.fps": 30.0, "video.height": 480, "video.width": 640, "video.channels": 3,
                         "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": 30.0},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 30.0},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n[OK] Merged {len(source_dirs)} datasets -> {output_dir}")
    print(f"     {n_episodes} episodes, {total_frames} frames, {task_idx} tasks")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge OMX datasets for IDM training")
    parser.add_argument("--sources", nargs="+", required=True, help="Source OMX dataset directories")
    parser.add_argument("--output", required=True, help="Output merged dataset directory")
    args = parser.parse_args()

    task_map = {
        "omx_pickplace": "pick and place objects",
        "omx_stacking": "stack blocks",
        "omx_sorting": "sort objects",
    }
    merge_datasets(args.sources, args.output, task_map)
