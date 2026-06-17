from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from legged_obstacle_rl.teleop import TeleopState


class VelocityCommandable(ABC):
    def __init__(self) -> None:
        self.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # vx,vy,wz

    def inject_teleop(self, state: TeleopState) -> None:
        self.vel_cmd[0] = state.lin_x
        self.vel_cmd[1] = state.lin_y
        self.vel_cmd[2] = state.ang_z


class ArgoCommandable(ABC):
    def __init__(self) -> None:
        self.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # vx,vy,wz
        self.z_cmd = 0.3

    def inject_teleop(self, state: TeleopState) -> None:
        self.vel_cmd[0] = state.lin_x
        self.vel_cmd[1] = state.lin_y
        self.vel_cmd[2] = state.ang_z
        self.z_cmd = state.height


class DirectionCommandable(ABC):
    def __init__(self) -> None:
        self.dir_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # vx,vy,wz

    def inject_teleop(self, state: TeleopState) -> None:
        heading = np.array([state.lin_x, state.lin_y], dtype=np.float32)
        norm = float(np.linalg.norm(heading))
        if norm < 1e-3:
            self.dir_cmd[0] = 0.0
            self.dir_cmd[1] = 0.0
        else:
            self.dir_cmd[0] = heading[0] / norm
            self.dir_cmd[1] = heading[1] / norm
        self.dir_cmd[2] = float(np.sign(state.ang_z))
