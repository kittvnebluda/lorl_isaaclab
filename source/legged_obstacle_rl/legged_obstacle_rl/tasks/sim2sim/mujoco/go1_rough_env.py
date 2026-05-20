from importlib.resources import files

import mujoco
import numpy as np

from legged_obstacle_rl.tasks.sim2sim.mujoco.velocity_env import VelocityEnv, isaac_home_jpos, mujoco_to_isaac_joints

ISAAC_OFFSET = 0.5
HS_RESOLUTION = 0.1
HS_SIZE = (1.6, 1.0)
HS_OFFSET_Z = 20.0


class Go1RoughEnv(VelocityEnv):
    def __init__(self, **kwargs):
        xml_file = str(files("legged_obstacle_rl").joinpath("tasks/sim2sim/mujoco/unitree_go1/scene.xml"))
        super().__init__(xml_file, frame_skip=4, device="cpu", obs_size=235, **kwargs)

        x_range = np.arange(-HS_SIZE[0] / 2, HS_SIZE[0] / 2 + HS_RESOLUTION, HS_RESOLUTION)
        y_range = np.arange(-HS_SIZE[1] / 2, HS_SIZE[1] / 2 + HS_RESOLUTION, HS_RESOLUTION)
        self.hs_xv, self.hs_yv = np.meshgrid(x_range, y_range)

    def _get_obs(self):
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
        body_pos = self.data.xpos[self._main_body]
        body_mat = self.data.xmat[self._main_body].reshape(3, 3)

        yaw = np.arctan2(body_mat[1, 0], body_mat[0, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        yaw_mat = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

        local_origins = np.stack(
            [
                self.hs_xv.flatten(),
                self.hs_yv.flatten(),
                np.full_like(self.hs_xv.flatten(), HS_OFFSET_Z),
            ],
            axis=-1,
        )

        world_origins = body_pos + local_origins @ yaw_mat.T

        world_direction = np.array([0, 0, -1.0])

        distances = []
        geom_id = np.zeros(1, dtype=np.int32)
        for origin in world_origins:
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                pnt=origin,
                vec=world_direction,
                geomgroup=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                flg_static=1,
                bodyexclude=-1,
                geomid=geom_id,
            )

            ground = origin[2] - dist
            val = body_pos[2] + ground - ISAAC_OFFSET
            distances.append(np.clip(val, -1.0, 1.0))

        return np.array(distances)
