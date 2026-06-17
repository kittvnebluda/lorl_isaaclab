import numpy as np
from numpy.typing import NDArray

from ..commandables import VelocityCommandable
from ..joints import isaac_home_jpos, mujoco_to_isaac_joints
from ..utils import mujoco_height_scan
from .go1_env import Go1Env

ISAAC_OFFSET = 0.5
HS_RESOLUTION = 0.1
HS_SIZE = (1.6, 1.0)
HS_OFFSET_Z = 20.0


class Go1VelocityEnv(VelocityCommandable, Go1Env):
    def __init__(self, xml_file: str | None = None, **kwargs):
        VelocityCommandable.__init__(self)
        Go1Env.__init__(self, xml_file, frame_skip=4, device="cpu", obs_size=235, **kwargs)

        x_range = np.arange(-HS_SIZE[0] / 2, HS_SIZE[0] / 2 + HS_RESOLUTION, HS_RESOLUTION)
        y_range = np.arange(-HS_SIZE[1] / 2, HS_SIZE[1] / 2 + HS_RESOLUTION, HS_RESOLUTION)
        self.hs_xv, self.hs_yv = np.meshgrid(x_range, y_range)

    def get_obs(self) -> NDArray[np.float64]:
        """Construct and return the observation vector for the current simulation state.

        Assembles a flat numpy array of shape ``(obs_size,)`` from multiple
        state sources in a fixed order.  The result is cast to ``float64`` and
        validated against ``self.obs_size`` before being returned.

        Returns
        -------
        obs : np.ndarray, dtype=float64
            Concatenated observation vector of length ``self.obs_size``.

        Raises
        ------
        AssertionError
            If the assembled vector length does not match ``self.obs_size``.

        Observation layout
        ------------------
        +----+----------------------------------------------+--------------------------------------------------+
        | ID | Description                                  | Details                                          |
        +====+==============================================+==================================================+
        |  1 | Joint positions (relative to home pose)      | ``qpos[7:][mujoco_to_isaac_joints]``             |
        |    |                                              | minus ``isaac_home_jpos``; shape ``(12,)``       |
        +----+----------------------------------------------+--------------------------------------------------+
        |  2 | Base linear velocity                         | Robot-frame linear velocity of the base link;    |
        |    |                                              | shape ``(3,)``                                   |
        +----+----------------------------------------------+--------------------------------------------------+
        |  3 | Base angular velocity                        | Robot-frame ``qvel[3:6]``; shape ``(3,)``        |
        +----+----------------------------------------------+--------------------------------------------------+
        |  4 | Joint velocities                             | ``qvel[6:][mujoco_to_isaac_joints]``;            |
        |    |                                              | shape ``(12,)``                                  |
        +----+----------------------------------------------+--------------------------------------------------+
        |  5 | Projected gravity vector                     | Gravity vector projected into the base frame;    |
        |    |                                              | shape ``(3,)``                                   |
        +----+----------------------------------------------+--------------------------------------------------+
        |  6 | Velocity command                             | ``self.vel_cmd`` (vx, vy, yaw rate);             |
        |    |                                              | shape ``(3,)``                                   |
        +----+----------------------------------------------+--------------------------------------------------+
        |  7 | Previous actions                             | ``self.actions[-1]``; shape ``(12,)``            |
        +----+----------------------------------------------+--------------------------------------------------+
        |  8 | Height scan                                  | Terrain height samples around the base;          |
        |    |                                              | shape ``(187,)``                                 |
        +----+----------------------------------------------+--------------------------------------------------+
        """
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        base_ang_vel = qvel[3:6]

        obs = np.concatenate(
            (
                qpos[7:][mujoco_to_isaac_joints] - isaac_home_jpos,
                self.base_lin_vel(),
                base_ang_vel,
                qvel[6:][mujoco_to_isaac_joints],
                self.projected_gravity(),
                self.vel_cmd,
                self.actions[-1],
                self._height_scan(),
            )
        ).astype(np.float64)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs

    def _height_scan(self):
        return mujoco_height_scan(
            self.model,
            self.data,
            self._main_body,
            self.hs_xv,
            self.hs_yv,
            offset_z=HS_OFFSET_Z,
            base_offset=ISAAC_OFFSET,
        )
