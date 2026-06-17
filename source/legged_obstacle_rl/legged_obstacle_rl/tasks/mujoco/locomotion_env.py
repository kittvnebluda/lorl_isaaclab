from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray

from .joints import isaac_home_jpos, isaac_to_mujoco_joints, mujoco_to_isaac_joints
from .utils import normalize, quat_apply_inverse

if TYPE_CHECKING:
    from legged_obstacle_rl.teleop import TeleopState

ZERO_ACTION = np.zeros(12, dtype=np.float64)
GRAVITY_VEC = normalize(np.array([[0.0, 0.0, -9.81]], dtype=np.float64)).squeeze(0)


class LocomotionEnv(MujocoEnv, ABC):
    metadata = {"render_modes": ["human"]}
    init_height: float = 0.4

    def __init__(
        self,
        xml_file: str,
        frame_skip: int,
        device: Literal["cpu", "cuda"],
        obs_size: int,
        **kwargs,
    ):
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None, **kwargs)

        self.metadata = {"render_modes": ["human"], "render_fps": int(np.round(1.0 / self.dt))}

        self.device = device
        self.obs_size = obs_size

        self.action_scale = 0.25
        self.max_actions_len = 15

        self._main_body = 1
        self._step_counter = 0

        self.action_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64)

    @abstractmethod
    def get_obs(self) -> NDArray[np.float64]: ...

    def get_info(self) -> dict[str, np.float64]:
        return {}

    @abstractmethod
    def compute_ctrl(
        self, ctrl_isaac: NDArray[np.float64], q_isaac: NDArray[np.float64], v_isaac: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map a target joint configuration to joint efforts, in IsaacLab joint order."""

    @abstractmethod
    def inject_teleop(self, state: TeleopState) -> None: ...

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:

        self.do_simulation(isaac_home_jpos + action * self.action_scale, self.frame_skip)

        self.actions.append(action.copy())
        if len(self.actions) > self.max_actions_len:
            del self.actions[0]

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), np.float64(0.0), False, False, self.get_info()

    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")

        for _ in range(n_frames):
            q = self.data.qpos[7:]
            v = self.data.qvel[6:]

            efforts_isaac = self.compute_ctrl(ctrl, q[mujoco_to_isaac_joints], v[mujoco_to_isaac_joints])
            efforts_mj = efforts_isaac[isaac_to_mujoco_joints]

            self.data.ctrl[:] = efforts_mj
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset_model(self):
        self.actions = [ZERO_ACTION.copy()] * self.max_actions_len
        self._ep_start_time = copy(self.data.time)

        qpos = np.concatenate([np.array([0, 0, self.init_height, 1, 0, 0, 0]), isaac_home_jpos[isaac_to_mujoco_joints]])
        qvel = np.zeros(len(qpos) - 1)
        self.set_state(qpos, qvel)

        return self.get_obs()

    def base_lin_vel(self):
        base_quat = self.data.qpos[3:7]
        return quat_apply_inverse(base_quat, self.data.qvel[:3])

    def projected_gravity(self):
        q = self.data.qpos[3:7]  # (w, x, y, z)
        return quat_apply_inverse(q, GRAVITY_VEC)
