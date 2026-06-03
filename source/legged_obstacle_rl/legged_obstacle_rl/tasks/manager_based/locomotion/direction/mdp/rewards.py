from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_linear_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Command shape: (num_envs, 3) -> [cos(yaw), sin(yaw), turn_dir]
    command = env.command_manager.get_command("base_direction")
    cmd_dir = command[:, :2]  # Extract horizontal direction vector
    cmd_norm = torch.norm(cmd_dir, dim=1)
    is_standing_cmd = cmd_norm < 0.1

    vel_xy_b = asset.data.root_lin_vel_b[:, :2]
    v_pr = torch.sum(vel_xy_b * cmd_dir, dim=1)

    rew = torch.where(v_pr >= 0.6, 1.0, 0.0)
    rew = torch.where(v_pr < 0.6, torch.exp(-2.0 * torch.square(v_pr - 0.6)), rew)
    rew = torch.where(is_standing_cmd, 0.0, rew)

    return rew


def track_angular_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command("base_direction")
    cmd_turn = command[:, 2]  # Discrete turning command: -1, 0, 1
    vel_yaw_b = asset.data.root_ang_vel_b[:, 2]

    w_pr = cmd_turn * vel_yaw_b

    return torch.where(w_pr >= 0.6, 1.0, torch.exp(-1.5 * torch.square(w_pr - 0.6)))


def base_motion_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Command shape: (num_envs, 3) -> [cos(yaw), sin(yaw), turn_dir]
    command = env.command_manager.get_command("base_direction")
    cmd_dir = command[:, :2]

    lv_xy = asset.data.root_lin_vel_b[:, :2]
    av_xy = asset.data.root_ang_vel_b[:, :2]

    v_pr = torch.sum(lv_xy * cmd_dir, dim=1)
    v_o = torch.norm(lv_xy - v_pr.unsqueeze(1) * cmd_dir, dim=1)

    return torch.exp(-1.5 * torch.square(v_o)) + torch.exp(-1.5 * torch.sum(torch.square(av_xy), dim=1))


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    is_commanded_to_move = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= is_commanded_to_move

    return reward


def track_linear_velocity_ramp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward-progress reward with no floor. Linear ramp 0 -> 1 for v_pr in [0, 0.6].

    Unlike :func:`track_linear_velocity`, v_pr <= 0 (standing / backwards) yields 0, so a robot
    that spins or hops in place cannot farm this term. Survival is provided by a dedicated alive
    bonus instead of a velocity-reward floor.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command("base_direction")
    cmd_dir = command[:, :2]
    is_standing_cmd = torch.norm(cmd_dir, dim=1) < 0.1

    v_pr = torch.sum(asset.data.root_lin_vel_b[:, :2] * cmd_dir, dim=1)

    rew = torch.clamp(v_pr, min=0.0, max=0.6) / 0.6
    return torch.where(is_standing_cmd, 0.0, rew)


def track_angular_velocity_still(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Turn-tracking reward that also penalizes yaw spin when no turn is commanded.

    Unlike :func:`track_angular_velocity`, when cmd_turn == 0 the yaw velocity is no longer
    unconstrained: the reward peaks at vel_yaw == 0 and decays with any spin. Removes the flat
    ~0.583 floor that let robots spin freely while no turn was commanded.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command("base_direction")
    cmd_turn = command[:, 2]  # Discrete turning command: -1, 0, 1
    vel_yaw_b = asset.data.root_ang_vel_b[:, 2]

    no_turn = cmd_turn.abs() < 0.1
    w_pr = cmd_turn * vel_yaw_b
    turn_rew = torch.where(w_pr >= 0.6, 1.0, torch.exp(-1.5 * torch.square(w_pr - 0.6)))
    still_rew = torch.exp(-1.5 * torch.square(vel_yaw_b))  # peak at vel_yaw == 0

    return torch.where(no_turn, still_rew, turn_rew)


def feet_air_time_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Air-time reward gated on real forward progress, with capped flight time.

    Unlike :func:`feet_air_time`, the per-foot bonus is clamped (discourages big jumps) and the
    term is gated on v_pr > 0.1 (actual forward motion) instead of merely being commanded to move,
    so hopping in place earns nothing.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    reward = torch.sum(torch.clamp(last_air_time - threshold, max=0.3) * first_contact, dim=1)

    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_dir = env.command_manager.get_command(command_name)[:, :2]
    v_pr = torch.sum(asset.data.root_lin_vel_b[:, :2] * cmd_dir, dim=1)

    return reward * (v_pr > 0.1)


def flight_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty (1.0 per step) when all specified feet are simultaneously airborne.

    Discourages jumping: a full flight phase with no ground contact returns 1.0, which combined
    with a negative weight punishes leaping while leaving normal stance/swing untouched.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history  # (N, T, B, 3)
    in_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    num_contacts = torch.sum(in_contact.int(), dim=1)

    return (num_contacts == 0).float()
