from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_obstacle_rl.robots.unitree import ALIENGO_CFG
from legged_obstacle_rl.sensors import CircularPatternCfg

from .. import mdp
from ..direction_env_cfg import DirectionRLEnvCfg

FEET = ("FL_calf/FL_foot", "FR_calf/FR_foot", "RL_calf/RL_foot", "RR_calf/RR_foot")


def _foot_scanner(foot: str) -> RayCasterCfg:
    """One downward concentric-ring height scanner attached to a single foot."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + foot,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=CircularPatternCfg(radii=(0.08, 0.16, 0.24), num_points=(6, 12, 18)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        direction_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_direction"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilligedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        fl_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("fl_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        fr_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("fr_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        rl_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("rl_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        rr_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("rr_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        actuator_gains = ObsTerm(
            func=mdp.actuator_gains,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        forces = ObsTerm(
            func=mdp.external_force_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        torques = ObsTerm(
            func=mdp.external_torque_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_states,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()
    privilliged = PrivilligedCfg()


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_linear_velocity, weight=0.8)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_angular_velocity, weight=0.5)

    base_motion_exp = RewTerm(func=mdp.base_motion_reward, weight=0.06)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-8)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={
            "command_name": "base_direction",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-3.0e-3)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


@configclass
class AlienGoRoughEnvCfg_v0(DirectionRLEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None

        self.scene.fl_foot_scanner = _foot_scanner(FEET[0])
        self.scene.fr_foot_scanner = _foot_scanner(FEET[1])
        self.scene.rl_foot_scanner = _foot_scanner(FEET[2])
        self.scene.rr_foot_scanner = _foot_scanner(FEET[3])
        for _name in ("fl_foot_scanner", "fr_foot_scanner", "rl_foot_scanner", "rr_foot_scanner"):
            getattr(self.scene, _name).update_period = self.decimation * self.sim.dt

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


@configclass
class AlienGoRoughEnvCfg_v0_DISTILL(AlienGoRoughEnvCfg_v0):
    """Teacher-student distillation variant: corrupt the student's proprioception.

    Same dynamics/randomization as the teacher env, but the ``policy`` (proprio) group
    is noisy so the DAgger student learns robustness to real sensor noise. The
    ``priviliged`` group stays clean — it is the teacher's (deterministic) input.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
