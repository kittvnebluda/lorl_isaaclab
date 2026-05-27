import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from legged_obstacle_rl.tasks.manager_based import mdp
from legged_obstacle_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import VelocityRLEnvCfg

ALIENGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/aliengo/aliengo.usd",
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
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=44.4,
            saturation_effort=44.4,
            velocity_limit=21.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)


@configclass
class AlienGoRoughEnvCfg_v0(VelocityRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.2)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.02

        self.events.physics_material.params["static_friction_range"] = (0.6, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_com.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"

        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.rewards.track_height = None
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_calf"
        self.rewards.flat_orientation_l2.weight = -0.15

        self.observations.policy.height_command = None

        self.commands.base_height = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class AlienGoRoughEnvCfg_v0_PLAY(AlienGoRoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 7
            self.scene.terrain.terrain_generator.curriculum = True

        self.observations.policy.enable_corruption = False

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
