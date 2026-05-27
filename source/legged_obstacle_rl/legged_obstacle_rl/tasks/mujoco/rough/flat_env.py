from importlib.resources import files

from numpy.typing import NDArray

import numpy as np

from legged_obstacle_rl.tasks.mujoco.utils import mujoco_height_scan
from legged_obstacle_rl.tasks.mujoco.velocity_env import VelocityEnv, isaac_home_jpos, mujoco_to_isaac_joints

ISAAC_OFFSET = 0.5
HS_RESOLUTION = 0.1
HS_SIZE = (1.6, 1.0)
HS_OFFSET_Z = 20.0


class Go1RoughFlatEnv(VelocityEnv):
    def __init__(self, xml_file: str | None = None, **kwargs):
        if xml_file is None:
            xml_file = str(files("legged_obstacle_rl").joinpath("tasks/mujoco/unitree_go1/scene.xml"))
        super().__init__(xml_file, frame_skip=4, device="cpu", obs_size=235, **kwargs)

        x_range = np.arange(-HS_SIZE[0] / 2, HS_SIZE[0] / 2 + HS_RESOLUTION, HS_RESOLUTION)
        y_range = np.arange(-HS_SIZE[1] / 2, HS_SIZE[1] / 2 + HS_RESOLUTION, HS_RESOLUTION)
        self.hs_xv, self.hs_yv = np.meshgrid(x_range, y_range)

    def get_obs(self) -> NDArray[np.float32]:
        """Construct and return the observation vector for the current simulation state.

        Assembles a flat numpy array of shape ``(obs_size,)`` from multiple
        state sources in a fixed order.  The result is cast to ``float32`` and
        validated against ``self.obs_size`` before being returned.

        Returns
        -------
        obs : np.ndarray, dtype=float32
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
                self.height_scan(),
            )
        ).astype(np.float32)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs

    def print_debug(self):
        lv = self.base_lin_vel()
        lines = [
            "------------ DEBUG INFO ------------",
            f"Time  : {self.data.time:8.3f} s",
            "-------",
            f"CMD VX: {self.vel_cmd[0]:8.3f} m/s    ACTUAL VX: {lv[0]:8.3f} m/s",
            f"CMD VY: {self.vel_cmd[1]:8.3f} m/s    ACTUAL VY: {lv[1]:8.3f} m/s",
            f"CMD WZ: {self.vel_cmd[2]:8.3f} rad/s  ACTUAL WZ: {lv[2]:8.3f} rad/s",
            "------------------------------------",
            "",
        ]
        print("\n".join(lines))

    def height_scan(self):
        return mujoco_height_scan(
            self.model,
            self.data,
            self._main_body,
            self.hs_xv,
            self.hs_yv,
            offset_z=HS_OFFSET_Z,
            base_offset=ISAAC_OFFSET,
        )
