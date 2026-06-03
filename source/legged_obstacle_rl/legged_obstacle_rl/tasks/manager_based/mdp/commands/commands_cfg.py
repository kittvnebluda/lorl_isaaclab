from dataclasses import MISSING
from math import pi

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import UniformVelocityCommandCfg as _UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils import configclass

from .direction_command import UniformDirectionCommand
from .height_command import UniformBodyHeightCommand
from .velocity_command import UniformVelocityCommand


@configclass
class UniformVelocityCommandCfg(_UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` whose term supports keyboard teleop injection."""

    class_type: type = UniformVelocityCommand


@configclass
class UniformDirectionCommandCfg(CommandTermCfg):
    """Configuration for the uniform direction command generator."""

    class_type: type = UniformDirectionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    turn_prob: float = 0.3
    """Probability of sampling a non-zero turn command. With probability ``turn_prob`` the turn
    direction is +/-1 (each with equal chance); otherwise it is 0 (no rotation). Lower values yield
    fewer rotation commands. ``turn_prob=2/3`` reproduces the old uniform {-1, 0, 1} sampling."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the direction commands."""

        yaw: tuple[float, float] = (-pi, pi)
        """Range for the yaw command (in rad)."""

    ranges: Ranges = Ranges()
    """Distribution ranges for the direction commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/direction_goal"
    )
    """The configuration for the goal direction visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/direction_current"
    )
    """The configuration for the current direction visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    turn_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/direction_turn",
        markers={"arrow": GREEN_ARROW_X_MARKER_CFG.markers["arrow"], "sphere": SPHERE_MARKER_CFG.markers["sphere"]},
    )
    """Turn-command marker: up arrow = counter-clockwise, down arrow = clockwise, sphere = no turn."""

    turn_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    turn_visualizer_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))
    turn_visualizer_cfg.markers["sphere"].scale = (0.15, 0.15, 0.15)


@configclass
class UniformBodyHeightCommandCfg(CommandTermCfg):
    class_type: type = UniformBodyHeightCommand

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    debug_vis: bool = True

    marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/height_goal")

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """Range for the height (in m)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the height command."""
