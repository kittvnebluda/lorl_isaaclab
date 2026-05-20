from importlib.resources import files

import numpy as np
from gymnasium.spaces import Box

from legged_obstacle_rl.tasks.mujoco.velocity_env import VelocityEnv, isaac_home_jpos, mujoco_to_isaac_joints


class Go1ArgoEnv(VelocityEnv):
    def __init__(self, **kwargs):
        xml_file = str(files("legged_obstacle_rl").joinpath("tasks/mujoco/unitree_go1/scene.xml"))
        super().__init__(xml_file, frame_skip=4, device="cpu", obs_size=49, **kwargs)

        self.z_cmd = 0.3

    def get_obs(self):
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
                [self.z_cmd],
                self.actions[-1],
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
            f"CMD Z : {self.z_cmd:8.3f} m      ACTUAL Z : {self.data.qpos[2]:8.3f} m",
            "------------------------------------",
            "",
        ]
        print("\n".join(lines))


class Go1ArgoHEnv(Go1ArgoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_size = 217
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

    def get_obs(self):
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
                [self.z_cmd],
                np.concatenate(self.actions[-15:]),  # TODO: is the order right?
            )
        ).astype(np.float32)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs
