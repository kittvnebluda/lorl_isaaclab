"""Event terms specific to the direction locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["push_by_setting_velocity_buffered", "reset_push_buffer"]


def push_by_setting_velocity_buffered(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by adding a random root-velocity delta, recording the delta.

    Same effect as ``isaaclab.envs.mdp.push_by_setting_velocity`` but stores the applied
    delta on ``env._last_push_velocity`` so the ``last_push_velocity`` observation can
    expose it to the critic. The buffer is lazily created and persists between pushes.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # lazily allocate the push buffer
    if getattr(env, "_last_push_velocity", None) is None:
        env._last_push_velocity = torch.zeros(env.num_envs, 6, device=asset.device)

    # sample the velocity delta
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    delta = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # apply and record
    vel_w = asset.data.root_vel_w[env_ids]
    vel_w += delta
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
    env._last_push_velocity[env_ids] = delta


def reset_push_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Zero the recorded push delta for the given envs (use on episode reset)."""
    if getattr(env, "_last_push_velocity", None) is None:
        return
    if env_ids is None:
        env._last_push_velocity.zero_()
    else:
        env._last_push_velocity[env_ids] = 0.0
