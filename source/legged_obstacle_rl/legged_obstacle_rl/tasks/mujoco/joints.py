import numpy as np

from legged_obstacle_rl.tasks.mujoco.utils import map_indexes

# fmt: off
# Joint Order in MuJoCo
mujoco_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# Joint Order in IsaacLab
isaac_joint_names = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]
# Home joint positions in IsaacLab
isaac_home_jpos = np.array([
     0.1, -0.1,  0.1, -0.1, # hips
     0.8,  0.8,  1.0,  1.0, # thighs
    -1.5, -1.5, -1.5, -1.5, # calves
])
# fmt: on
isaac_to_mujoco_joints = map_indexes(target=mujoco_joint_names, source=isaac_joint_names)
mujoco_to_isaac_joints = map_indexes(target=isaac_joint_names, source=mujoco_joint_names)
