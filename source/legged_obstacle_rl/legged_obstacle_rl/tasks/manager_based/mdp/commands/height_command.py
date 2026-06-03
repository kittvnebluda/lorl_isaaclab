from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from legged_obstacle_rl.teleop import TeleopState

    from .commands_cfg import UniformBodyHeightCommandCfg


class UniformBodyHeightCommand(CommandTerm):
    cfg: UniformBodyHeightCommandCfg

    def __init__(self, cfg: UniformBodyHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "HeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired height. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        actual_height = self.robot.data.root_pos_w[:, 2]
        cmd_height = self.command.squeeze(-1)
        self.metrics["error_height"] = torch.abs(cmd_height - actual_height)

    def _resample_command(self, env_ids: Sequence[int]):
        min_h, max_h = self.cfg.ranges.height
        self.height_command[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (max_h - min_h) + min_h

    def _update_command(self):
        pass

    def inject_teleop(self, state: TeleopState) -> None:
        """Override the height command from a teleop state. Freezes periodic resampling."""
        self.cfg.resampling_time_range = (1.0e9, 1.0e9)
        self.height_command[:, 0] = state.height

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set visibility of markers"""
        if debug_vis:
            if not hasattr(self, "height_visualizer"):
                self.height_visualizer = VisualizationMarkers(self.cfg.marker_cfg)
            self.height_visualizer.set_visibility(True)
        else:
            if hasattr(self, "height_visualizer"):
                self.height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return

        base_pos = self.robot.data.root_pos_w.clone()
        base_pos[:, 2] = self.height_command.squeeze(-1)
        base_pos[:, 2] += 0.5

        self.height_visualizer.visualize(base_pos)
