from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommand as _UniformVelocityCommand

if TYPE_CHECKING:
    from legged_obstacle_rl.teleop import TeleopState


class UniformVelocityCommand(_UniformVelocityCommand):
    """``UniformVelocityCommand`` that can be overridden by a keyboard teleop state.

    Identical to the base command during training/play; only adds ``inject_teleop`` so the
    play scripts can drive the velocity command directly (see ``legged_obstacle_rl.teleop``).
    """

    def inject_teleop(self, state: TeleopState) -> None:
        """Override the velocity command from a teleop state: command = <lin_x, lin_y, ang_z>.

        Freezes periodic resampling on the first call and clears standing flags so the teleop
        command holds across steps.
        """
        # freeze auto-resampling so the teleop command persists (idempotent)
        self.cfg.resampling_time_range = (1.0e9, 1.0e9)

        self.vel_command_b[:, 0] = state.lin_x
        self.vel_command_b[:, 1] = state.lin_y
        self.vel_command_b[:, 2] = state.ang_z
        self.is_standing_env[:] = False
