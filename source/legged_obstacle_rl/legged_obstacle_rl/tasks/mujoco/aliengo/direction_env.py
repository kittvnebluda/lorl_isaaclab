from importlib.resources import files

import numpy as np
from numpy.typing import NDArray

from ..commandables import DirectionCommandable
from ..joints import isaac_home_jpos, mujoco_to_isaac_joints
from .aliengo_env import AliengoEnv


class AliengoDirectionProprioEnv(DirectionCommandable, AliengoEnv):
    def __init__(self, xml_file: str | None = None, **kwargs):
        DirectionCommandable.__init__(self)
        AliengoEnv.__init__(self, xml_file, frame_skip=4, device="cpu", obs_size=45, **kwargs)

    def get_obs(self) -> NDArray[np.float64]:
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        base_ang_vel = qvel[3:6]

        obs = np.concatenate(
            (
                qpos[7:][mujoco_to_isaac_joints] - isaac_home_jpos,  # 12: joint pos (rel)
                base_ang_vel,  # 3:  base ang vel
                qvel[6:][mujoco_to_isaac_joints],  # 12: joint vel
                self.projected_gravity(),  # 3:  proj gravity
                self.dir_cmd,  # 3:  direction command
                self.actions[-1],  # 12: last action
            )
        ).astype(np.float64)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs


class AliengoDirectionProprioIcraFlatEnv(AliengoDirectionProprioEnv):
    def __init__(self, **kwargs):
        xml = "tasks/mujoco/aliengo/unitree_aliengo/scene_icra_flat.xml"
        xml_file = str(files("legged_obstacle_rl").joinpath(xml))
        super().__init__(xml_file=xml_file, **kwargs)


class AliengoDirectionProprioIcraSlopedEnv(AliengoDirectionProprioEnv):
    def __init__(self, **kwargs):
        xml = "tasks/mujoco/aliengo/unitree_aliengo/scene_icra_sloped.xml"
        xml_file = str(files("legged_obstacle_rl").joinpath(xml))
        super().__init__(xml_file=xml_file, **kwargs)
