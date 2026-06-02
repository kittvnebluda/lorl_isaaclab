"""Privileged observations that expose domain-randomization events to the critic.

These read back the quantities written by the events in the env cfg:

* ``actuator_gains``    <- ``randomize_actuator_gains`` (startup)
* ``external_force_b``  <- ``apply_external_force_torque`` (reset)
* ``external_torque_b`` <- ``apply_external_force_torque`` (reset)
* ``last_push_velocity``<- ``push_by_setting_velocity_buffered`` (interval, below)

Privileged-only: feed to the critic / teacher, not the deployed actor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "actuator_gains",
    "external_force_b",
    "external_torque_b",
    "foot_contact_states",
    "last_push_velocity",
]


def actuator_gains(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    relative: bool = True,
) -> torch.Tensor:
    """Per-joint PD gains set into the sim, as randomized by ``randomize_actuator_gains``.

    Returns the concatenation of stiffness and damping over the selected joints.
    With ``relative=True`` (default) each gain is divided by its default value, so the
    observation is the dimensionless randomization factor (~1.0) — better conditioned
    for a network than raw N·m/rad gains. Zero-default joints fall back to the raw value.

    Shape: ``(num_envs, 2 * num_selected_joints)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    stiffness = asset.data.joint_stiffness[:, joint_ids]
    damping = asset.data.joint_damping[:, joint_ids]
    if relative:
        default_stiffness = asset.data.default_joint_stiffness[:, joint_ids]
        default_damping = asset.data.default_joint_damping[:, joint_ids]
        stiffness = torch.where(default_stiffness != 0, stiffness / default_stiffness, stiffness)
        damping = torch.where(default_damping != 0, damping / default_damping, damping)
    return torch.cat([stiffness, damping], dim=-1)


def external_force_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """External force (link frame) on the selected bodies, set by ``apply_external_force_torque``.

    Shape: ``(num_envs, 3 * num_selected_bodies)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    force = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids]
    return force.reshape(env.num_envs, -1)


def external_torque_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """External torque (link frame) on the selected bodies, set by ``apply_external_force_torque``.

    Shape: ``(num_envs, 3 * num_selected_bodies)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids]
    return torque.reshape(env.num_envs, -1)


def foot_contact_states(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state per selected foot (1.0 = in contact, 0.0 = swing).

    A foot counts as in contact if the max net contact-force magnitude over the
    sensor history exceeds ``threshold`` (N). Reading the history (rather than the
    instantaneous force) debounces brief force dropouts during stance, matching the
    contact signal Lee et al feed to the teacher.

    Shape: ``(num_envs, num_selected_bodies)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > threshold
    return in_contact.float()


def last_push_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    """The most recent root-velocity delta injected by ``push_by_setting_velocity_buffered``.

    Holds the (vx, vy, vz, wx, wy, wz) delta of the last push until the next one, so the
    critic sees the disturbance. Zeros before the first push. Shape: ``(num_envs, 6)``.
    """
    buf = getattr(env, "_last_push_velocity", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, 6, device=env.device)
        env._last_push_velocity = buf
    return buf
