from abc import ABC
from importlib.resources import files
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..locomotion_env import LocomotionEnv


class AliengoEnv(LocomotionEnv, ABC):
    init_height: float = 0.5

    def __init__(self, xml_file: str | None, frame_skip: int, device: Literal["cpu", "cuda"], obs_size: int, **kwargs):
        if xml_file is None:
            xml_file = str(files("legged_obstacle_rl").joinpath("tasks/mujoco/aliengo/unitree_aliengo/scene.xml"))
        super().__init__(xml_file, frame_skip, device, obs_size, **kwargs)

    def compute_ctrl(
        self, ctrl_isaac: NDArray[np.float64], q_isaac: NDArray[np.float64], v_isaac: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return ctrl_isaac
