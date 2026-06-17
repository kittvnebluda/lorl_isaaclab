import numpy as np

from ..commandables import ArgoCommandable
from ..joints import isaac_home_jpos, mujoco_to_isaac_joints
from .go1_env import Go1Env


class Go1ArgoEnv(ArgoCommandable, Go1Env):
    def __init__(self, **kwargs):
        ArgoCommandable.__init__(self)
        Go1Env.__init__(self, frame_skip=4, device="cpu", obs_size=49, **kwargs)

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
        ).astype(np.float64)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs


class Go1ArgoHEnv(ArgoCommandable, Go1Env):
    def __init__(self, **kwargs):
        Go1Env.__init__(self, frame_skip=4, device="cpu", obs_size=217, **kwargs)
        ArgoCommandable.__init__(self)

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
                np.concatenate(self.actions[-15:]),
            )
        ).astype(np.float64)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs
