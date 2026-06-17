from abc import ABC
from importlib.resources import files
from typing import Literal

import numpy as np
import torch
from isaaclab.actuators import ActuatorNetMLP
from isaaclab.utils.types import ArticulationActions
from isaaclab_assets.robots.unitree import GO1_ACTUATOR_CFG
from numpy.typing import NDArray

from ..joints import isaac_joint_names
from ..locomotion_env import LocomotionEnv


class Go1Env(LocomotionEnv, ABC):
    def __init__(self, xml_file: str | None, frame_skip: int, device: Literal["cpu", "cuda"], obs_size: int, **kwargs):
        if xml_file is None:
            xml_file = str(files("legged_obstacle_rl").joinpath("tasks/mujoco/go1/unitree_go1/scene.xml"))
        super().__init__(xml_file, frame_skip, device, obs_size, **kwargs)

        self.actuators = ActuatorNetMLP(
            GO1_ACTUATOR_CFG, joint_names=isaac_joint_names, joint_ids=slice(None), num_envs=1, device=self.device
        )

    def compute_ctrl(
        self, ctrl_isaac: NDArray[np.float64], q_isaac: NDArray[np.float64], v_isaac: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map a target joint configuration to joint efforts, in IsaacLab joint order.

        Default uses the GO1 ``ActuatorNetMLP``. Override for a different actuator model.
        All arguments and the return value are in IsaacLab joint order.
        """
        target_articulation = self.actuators.compute(
            ArticulationActions(joint_positions=torch.from_numpy(ctrl_isaac).float().unsqueeze(0)),
            torch.from_numpy(q_isaac).float().unsqueeze(0),
            torch.from_numpy(v_isaac).float().unsqueeze(0),
        )

        if target_articulation.joint_efforts is None:
            raise ValueError("ActuatorNetMLP returned None in joint_efforts")

        return target_articulation.joint_efforts.squeeze(0).detach().numpy()
