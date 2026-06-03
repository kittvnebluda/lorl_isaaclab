from typing import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels_dir(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_direction",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Terrain curriculum driven by command-aligned progress over the whole episode.

    Uses ``UniformDirectionCommand.command_progress`` — the per-step integral of base-frame
    velocity projected onto the command active that step. Unlike net world displacement, this
    does not cancel when an opposite command is resampled mid-episode: a robot that tracks every
    command keeps accumulating distance. Promote when it advanced far along its commands, demote
    when it barely moved. Standing/turning-dominated episodes are guarded against demotion.

    Thresholds scale with the sub-terrain size (one terrain tile), matching the original logic.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command_term = env.command_manager.get_term(command_name)

    tile = terrain.cfg.terrain_generator.size[0]

    command = command_term.command[env_ids]

    progress = command_term.command_progress[env_ids]
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    v_pr: torch.Tensor = command_term.metrics["v_pr"][env_ids]

    is_turning = torch.abs(command[:, 2]) >= 0.1
    is_standing = command_term.is_standing_env[env_ids]

    move_up = distance > tile * 0.5
    move_down = (progress < tile * 0.2) | (v_pr < 0.2)
    move_down = move_down & ~is_turning & ~is_standing

    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    return {"min": levels.min(), "mean": levels.mean(), "max": levels.max()}
