from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from legged_obstacle_rl.teleop import TeleopState

    from .commands_cfg import UniformDirectionCommandCfg


class UniformDirectionCommand(CommandTerm):
    r"""Command generator that produces a directional command vector as described in Lee et al. (2020).

    The command comprises a target horizontal direction in the robot's base frame
    and a discrete turning direction. Mathematically, it is defined as:
        command = < cos(ψ_T), sin(ψ_T), ω̂_T >
    where ψ_T is the commanded yaw angle and ω̂_T ∈ {-1, 0, 1} is the turning direction.
    A standing command is represented as <0.0, 0.0, 0.0>.

    Unlike velocity tracking commands, this generator only prescribes a heading and turning
    intent, allowing the policy to autonomously determine an appropriate speed based on terrain.
    """

    cfg: UniformDirectionCommandCfg

    def __init__(self, cfg: UniformDirectionCommandCfg, env):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffer: [cos_yaw, sin_yaw, turn_dir]
        self.dir_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # standing env tracking
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics
        self.metrics["cmd_angle_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["turn_sign_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v_pr"] = torch.zeros(self.num_envs, device=self.device)  # velocity projection metric from paper

        # command-aligned progress integrated per step against the command active that step.
        self.command_progress = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformDirectionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tYaw range: {self.cfg.ranges.yaw}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired direction command in the base frame. Shape is (num_envs, 3)."""
        return self.dir_command_b

    def reset(self, env_ids=None):
        """Reset per-episode accumulators, then defer to the base resample/metric logic.

        Called at episode boundaries. The terrain curriculum reads ``command_progress`` earlier
        in the same reset cycle (curriculum_manager.compute runs before command_manager.reset),
        so zeroing it here clears it for the next episode without losing the value.
        """
        if env_ids is None:
            self.command_progress[:] = 0.0
        else:
            self.command_progress[env_ids] = 0.0
        return super().reset(env_ids)

    def _update_metrics(self):
        """Log tracking metrics and the velocity projection (v_pr) used in the paper's reward."""
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # Actual velocity direction in base frame
        vel_xy_b = self.robot.data.root_lin_vel_b[:, :2]
        vel_norm = torch.norm(vel_xy_b, dim=-1, keepdim=True)
        vel_dir_b = torch.where(vel_norm > 1e-5, vel_xy_b / vel_norm, torch.zeros_like(vel_xy_b))

        # Commanded direction
        cmd_dir = self.dir_command_b[:, :2]

        # 1. Direction alignment error (angle between commanded and actual direction)
        dot_prod = torch.sum(cmd_dir * vel_dir_b, dim=-1).clamp(-1.0, 1.0)
        dir_error = torch.acos(dot_prod)
        self.metrics["cmd_angle_error"] += dir_error / max_command_step

        # 2. Turning alignment error
        # We compare commanded turn dir with sign of actual angular velocity
        actual_turn_dir = torch.sign(self.robot.data.root_ang_vel_b[:, 2])
        turn_error = torch.abs(self.dir_command_b[:, 2] - actual_turn_dir)
        self.metrics["turn_sign_error"] += turn_error / max_command_step

        # 3. Velocity projection (v_pr)
        v_pr = torch.sum(self.robot.data.root_lin_vel_b[:, :2] * cmd_dir, dim=-1)
        self.metrics["v_pr"] += v_pr / max_command_step

        # integrate command-aligned distance (v_pr * dt) against the command active this step.
        # frame-consistent (both velocity and command are base-frame), so it measures how far the
        # robot advanced along whatever it was told, summed over all commands in the episode.
        self.command_progress += v_pr * self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample direction commands for the specified environments."""
        r = torch.empty(len(env_ids), device=self.device)

        # Sample yaw angle uniformly from range
        yaw = r.uniform_(*self.cfg.ranges.yaw)
        self.dir_command_b[env_ids, 0] = torch.cos(yaw)
        self.dir_command_b[env_ids, 1] = torch.sin(yaw)

        # Sample discrete turning direction: {-1, 0, 1}.
        # With probability turn_prob issue a turn (+/-1, equal chance); otherwise 0 (no rotation).
        do_turn = r.uniform_(0.0, 1.0) <= self.cfg.turn_prob
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
        self.dir_command_b[env_ids, 2] = torch.where(do_turn, sign, torch.zeros_like(sign))

        # Update standing environments
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the command. Enforces zero command for standing environments."""
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.dir_command_b[standing_env_ids, :] = 0.0

    def inject_teleop(self, state: TeleopState) -> None:
        """Override the command from a keyboard teleop state (see ``legged_obstacle_rl.teleop``).

        Maps the (lin_x, lin_y) keys to a base-frame heading and the ang_z key to a discrete
        turn direction: command = <cos psi, sin psi, sign(ang_z)>. A near-zero heading vector
        yields a stand (and optional turn-in-place) command. Freezes periodic resampling on the
        first call and clears standing flags so the teleop command holds across steps.
        """
        # freeze auto-resampling so the teleop command persists (idempotent)
        self.cfg.resampling_time_range = (1.0e9, 1.0e9)

        heading = torch.tensor([state.lin_x, state.lin_y], device=self.device)
        norm = torch.linalg.norm(heading)
        if state.ang_z == 0.0:
            turn = 0
        else:
            turn = float(torch.sign(torch.tensor(state.ang_z)))
        if norm < 1e-3:
            self.dir_command_b[:, 0] = 0.0
            self.dir_command_b[:, 1] = 0.0
        else:
            self.dir_command_b[:, 0] = heading[0] / norm
            self.dir_command_b[:, 1] = heading[1] / norm
        self.dir_command_b[:, 2] = turn
        self.is_standing_env[:] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set visibility of debug visualization markers."""
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                self.turn_visualizer = VisualizationMarkers(self.cfg.turn_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.turn_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
                self.turn_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback to update debug visualization markers."""
        if not self.robot.is_initialized:
            return

        # Base position slightly above the robot for visibility
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # Resolve goal direction and current velocity direction to arrow visuals
        goal_dir = self.command[:, :2]
        goal_scale, goal_quat = self._resolve_xy_direction_to_arrow(goal_dir)

        curr_vel = self.robot.data.root_lin_vel_b[:, :2]
        curr_scale, curr_quat = self._resolve_xy_direction_to_arrow(curr_vel)

        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(base_pos_w, curr_quat, curr_scale)

        # Turn-command marker, placed higher than the direction arrows: up arrow = CCW (turn=+1),
        # down arrow = CW (turn=-1), sphere = no turn (turn=0).
        turn = self.command[:, 2]
        turn_pos_w = self.robot.data.root_pos_w.clone()
        turn_pos_w[:, 2] += 0.9

        # prototype index: 0 = arrow (turning), 1 = sphere (no turn)
        marker_indices = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        marker_indices[turn.abs() > 0.5] = 0

        # rotate the +X arrow about Y: -pi/2 -> world +Z (up, CCW), +pi/2 -> world -Z (down, CW)
        zeros = torch.zeros(self.num_envs, device=self.device)
        pitch = torch.zeros_like(zeros)
        pitch[turn > 0.5] = -math.pi / 2
        pitch[turn < -0.5] = math.pi / 2
        turn_quat = quat_from_euler_xyz(zeros, pitch, zeros)

        self.turn_visualizer.visualize(turn_pos_w, turn_quat, marker_indices=marker_indices)

    def _resolve_xy_direction_to_arrow(self, xy_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts an XY direction vector to arrow scale and orientation in world frame."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_vec.shape[0], 1)

        norm = torch.linalg.norm(xy_vec, dim=1)
        # Scale arrow by vector magnitude, clamped to prevent vanishing arrows on stop commands
        arrow_scale[:, 0] *= torch.clamp(norm * 3.0, min=0.15, max=1.5)

        # Compute heading angle from XY components
        heading_angle = torch.atan2(xy_vec[:, 1], xy_vec[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Rotate from base frame to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
