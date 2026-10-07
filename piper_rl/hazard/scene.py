"""Paths and config for the knife variant of the human-aware PiPER scene."""
from __future__ import annotations

from pathlib import Path

from piper_rl.human_config import HumanAwareEnvConfig

KNIFE_HUMAN_MODEL_PATH = Path(__file__).resolve().parents[2] / "agilex_piper" / "piper_human_knife_task.xml"


def knife_human_config(**overrides) -> HumanAwareEnvConfig:
    """HumanAwareEnvConfig pointing at the scene with the knife in the gripper."""
    cfg = HumanAwareEnvConfig(model_path=str(KNIFE_HUMAN_MODEL_PATH))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
