"""
OMX Modality Configuration for GR00T N1.6 Fine-tuning.

ROBOTIS OMX manipulator: 5-DOF + 1 Gripper = 6-dim action/state space.
This file is loaded by launch_finetune.py via --modality-config-path.

Usage:
    uv run python gr00t/experiment/launch_finetune.py \
        --modality-config-path configs/omx_modality_config.py \
        ...
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# OMX: 5-DOF joints (joint1~joint5) + 1 gripper = 6-dim
OMX_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam1"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # action_horizon = 16
        modality_keys=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "gripper",
        ],
        action_configs=[
            # joint1~joint5: relative joint position
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: absolute position (open/close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.GRIPPER,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(OMX_MODALITY_CONFIG, EmbodimentTag.NEW_EMBODIMENT)
