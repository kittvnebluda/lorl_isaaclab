from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from ..locomotion_env_cfg import LocomotionRLEnvCfg
from . import mdp


@configclass
class CommandsCfg:
    base_direction = mdp.UniformDirectionCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        turn_prob=0.3,
        debug_vis=True,
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_dir)


##
# Environment configuration
##


@configclass
class DirectionRLEnvCfg(LocomotionRLEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
