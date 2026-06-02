import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from legged_obstacle_rl.robots.unitree import GO1_CFG as UNITREE_GO1_CFG

from legged_obstacle_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import VelocityRLEnvCfg


@configclass
class Go1RoughEnvCfg_v0(VelocityRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)

        self.rewards.track_height = None
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.flat_orientation_l2.weight = -0.05

        self.observations.policy.height_command = None

        self.commands.base_height = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class Go1RoughEnvCfg_v0_PLAY(Go1RoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 7
            self.scene.terrain.terrain_generator.curriculum = True

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)


@configclass
class Go1RoughEnvCfg_v0_PLAY_ICRA(Go1RoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # disable reset, pushes and CoM change
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # set good frictions
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        # turn off curriculum
        self.curriculum.terrain_levels = None
        self.rewards.terrain_levels_mean = None
        # change map
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/litt/Projects/legged-rl-isaaclab/source/legged_obstacle_rl/legged_obstacle_rl/assets/icra_map_flat.usd",
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.1),
            ),
        )
        self.scene.robot.init_state.pos = (6.0, -4.3, 0.4)
        self.sim.physx.enable_ccd = True
        # visuals
        self.scene.height_scanner.debug_vis = True
        self.commands.base_velocity.debug_vis = False
