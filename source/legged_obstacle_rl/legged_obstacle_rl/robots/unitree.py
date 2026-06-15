import copy

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

from .utils import resolve_usd

ALIENGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=resolve_usd(f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/aliengo/aliengo.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # min/max_delay are in sim steps (sim.dt=0.005 -> max_delay=20 ~= 100 ms ~ one control period).
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=44.4,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
            min_delay=0,
            max_delay=20,
        ),
    },
)


GO1_CFG = copy.deepcopy(UNITREE_GO1_CFG)
GO1_CFG.spawn.usd_path = resolve_usd(f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd")
