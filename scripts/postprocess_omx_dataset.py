#!/usr/bin/env python3
"""Post-process OMX conversion output to match LeRobot v3 / GR00T IDM format.

Converts scalar-column parquet files to:
- Per-episode parquet files with array columns (observation.state, action)
- Proper meta/ directory (info.json, stats.json, episodes.jsonl, tasks.jsonl, modality.json)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

STATE_COLS = [
    "state.shoulder_pan", "state.shoulder_lift", "state.elbow_flex",
    "state.wrist_flex", "state.wrist_roll", "state.gripper",
]
ACTION_COLS = [
    "action.shoulder_pan", "action.shoulder_lift", "action.elbow_flex",
    "action.wrist_flex", "action.wrist_roll", "action.gripper",
]


def postprocess_dataset(dataset_dir: str, task_description: str, source_dataset: str):
    """Post-process a single OMX dataset directory."""
    dataset_dir = os.path.abspath(dataset_dir)
    data_dir = os.path.join(dataset_dir, "data", "chunk-000")
    meta_dir = os.path.join(dataset_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # Find the single combined parquet file
    parquet_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    )
    if not parquet_files:
        print(f"[ERROR] No parquet files found in {data_dir}")
        return False

    # Read all data
    all_dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in parquet_files]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Read {len(df)} rows from {len(parquet_files)} file(s)")

    # Check if already in array format
    if "observation.state" in df.columns:
        print("Already in array format, splitting per-episode only")
        needs_reformat = False
    elif STATE_COLS[0] in df.columns:
        needs_reformat = True
    else:
        print(f"[ERROR] Unknown column format: {list(df.columns)}")
        return False

    # Clean up old files
    for f in parquet_files:
        os.remove(os.path.join(data_dir, f))

    # Reformat and split by episode
    episodes_meta = []
    total_frames = 0

    for ep_idx, group in df.groupby("episode_index"):
        ep_idx = int(ep_idx)
        n = len(group)

        if needs_reformat:
            obs_state = group[STATE_COLS].values.astype(np.float32)
            action = group[ACTION_COLS].values.astype(np.float32)
            new_df = pd.DataFrame({
                "observation.state": [row for row in obs_state],
                "action": [row for row in action],
                "timestamp": group["timestamp"].values,
                "frame_index": group["frame_index"].values,
                "episode_index": group["episode_index"].values,
                "index": group["index"].values,
                "task_index": group["task_index"].values,
            })
        else:
            new_df = group

        ep_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
        new_df.to_parquet(ep_path, index=False)
        episodes_meta.append({"episode_index": ep_idx, "length": n})
        total_frames += n

    n_episodes = len(episodes_meta)
    print(f"Split into {n_episodes} episode files, {total_frames} total frames")

    # Read all data for stats (use first chunk's combined data)
    if needs_reformat:
        obs_all = df[STATE_COLS].values.astype(np.float32)
        act_all = df[ACTION_COLS].values.astype(np.float32)
    else:
        obs_all = np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
        act_all = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]])

    # Generate meta/stats.json
    stats = {}
    for key, data in [("observation.state", obs_all), ("action", act_all)]:
        stats[key] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.percentile(data, 1, axis=0).tolist(),
            "q99": np.percentile(data, 99, axis=0).tolist(),
        }
    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Wrote meta/stats.json")

    # Generate meta/episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    print("Wrote meta/episodes.jsonl")

    # Generate meta/tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")
    print("Wrote meta/tasks.jsonl")

    # Generate meta/modality.json
    modality = {
        "state": {},
        "action": {},
        "video": {"webcam": {"original_key": "observation.images.top"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for i, name in enumerate(joint_names):
        modality["state"][name] = {"start": i, "end": i + 1, "original_key": "observation.state"}
        is_gripper = name == "gripper"
        modality["action"][name] = {
            "start": i, "end": i + 1,
            "absolute": is_gripper,
            "original_key": "action",
        }
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)
    print("Wrote meta/modality.json")

    # Generate meta/info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "omx",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32", "shape": [6],
                "names": [f"main_{j}" for j in joint_names],
                "fps": 30.0,
            },
            "observation.state": {
                "dtype": "float32", "shape": [6],
                "names": [f"main_{j}" for j in joint_names],
                "fps": 30.0,
            },
            "observation.images.top": {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": 30.0, "video.height": 480, "video.width": 640,
                    "video.channels": 3, "video.codec": "av1",
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
        "_conversion_source": "so100",
        "_conversion_method": "URDF FK->IK (method B)",
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Wrote meta/info.json")

    # Setup video directory with per-episode symlinks
    # Original LeRobot v3 stores one video per chunk (file-000.mp4).
    # GR00T IDM code expects per-episode video files, so we create symlinks.
    if source_dataset:
        src_videos = os.path.join(os.path.abspath(source_dataset), "videos")
        videos_dir = os.path.join(dataset_dir, "videos")
        # Remove old symlink if it points to source videos dir
        if os.path.islink(videos_dir):
            os.unlink(videos_dir)
        for video_key in ["observation.images.top", "observation.images.wrist"]:
            src_chunk = os.path.join(src_videos, video_key, "chunk-000")
            if not os.path.exists(src_chunk):
                continue
            dst_chunk = os.path.join(videos_dir, video_key, "chunk-000")
            os.makedirs(dst_chunk, exist_ok=True)
            # Find the source video file (file-000.mp4)
            src_mp4 = os.path.join(src_chunk, "file-000.mp4")
            if not os.path.exists(src_mp4):
                src_files = [f for f in os.listdir(src_chunk) if f.endswith(".mp4")]
                if src_files:
                    src_mp4 = os.path.join(src_chunk, src_files[0])
                else:
                    continue
            # Create per-episode symlinks pointing to the single video file
            for ep in episodes_meta:
                ep_mp4 = os.path.join(dst_chunk, f"episode_{ep['episode_index']:06d}.mp4")
                if not os.path.exists(ep_mp4):
                    os.symlink(src_mp4, ep_mp4)
            print(f"Created {n_episodes} video symlinks for {video_key}")

    print(f"\n[OK] Post-processed {dataset_dir}: {n_episodes} episodes, {total_frames} frames")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process OMX dataset for IDM training")
    parser.add_argument("--dataset-dir", required=True, help="OMX dataset directory")
    parser.add_argument("--task", default="robot manipulation", help="Task description")
    parser.add_argument("--source-dataset", default="", help="Source SO-100 dataset for video symlink")
    args = parser.parse_args()
    success = postprocess_dataset(args.dataset_dir, args.task, args.source_dataset)
    sys.exit(0 if success else 1)
