"""AlienGo hardware deployment runtime.

Thin wrapper around go1.py infrastructure with AlienGo-specific constants.
Default obs builder is build_obs_proprio (45-dim, no height scan) — AlienGo
direction policies are proprioception-only.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Projects/unitree_legged_sdk/lib/python/amd64"))
import numpy as np
import robot_interface as sdk  # pyright: ignore[reportMissingImports]
import torch

from legged_obstacle_rl.deployment.go1 import build_obs_proprio, stop
from legged_obstacle_rl.deployment.go1 import run as _go1_run

KP = 0.01
KD = 2.0
LOW_IP = "192.168.123.10"
LEGGED_TYPE = sdk.LeggedType.Aliengo


def act_rsl_rl(policy, obs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions = policy(obs_t).squeeze(0).numpy()
    actions = np.zeros_like(actions)
    return actions


# TODO: unhardcode act_rsl_rl
def run(agent, *, build_obs_fn=build_obs_proprio, act_fn=act_rsl_rl) -> None:
    """Start teleop, hardware, ramp, then run policy until stop or error."""
    _go1_run(
        agent,
        kp=KP,
        kd=KD,
        low_ip=LOW_IP,
        legged_type=LEGGED_TYPE,
        build_obs_fn=build_obs_fn,
        act_fn=act_fn,
    )


__all__ = ["run", "stop", "KP", "KD", "LOW_IP", "LEGGED_TYPE"]
