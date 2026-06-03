import gymnasium as gym

from . import direction, velocity


def register_manager_based_env(id: str, env_cfg: str, skrl_cfg: str):
    gym.register(
        id=id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "skrl_cfg_entry_point": skrl_cfg},
    )


##
# Register Gym environments.
# ARGO.
##


def register_argo(id_postfix: str, cfg_name: str):
    register_manager_based_env(
        id="LORL-Go1Argo-RL-" + id_postfix,
        env_cfg=f"{velocity.__name__}.go1.argo_rl_env_cfg:" + cfg_name,
        skrl_cfg=f"{velocity.__name__}.go1.agents:skrl_argo_ppo_cfg.yaml",
    )


register_argo("v0", "Go1ArgoEnvCfg")
register_argo("Play-v0", "Go1ArgoEnvCfg_PLAY")

register_argo("H15-v0", "Go1ArgoHEnvCfg")
register_argo("H15-Play-v0", "Go1ArgoHEnvCfg_PLAY")


##
# Velocity based rough.
##


def register_rough(id_postfix: str, env_cfg: str, skrl_cfg: str):
    register_manager_based_env(
        id="LORL-Go1Rough-RL-" + id_postfix,
        env_cfg=f"{velocity.__name__}.go1.rough.env_cfg:" + env_cfg,
        skrl_cfg=f"{velocity.__name__}.go1.rough.agents:" + skrl_cfg,
    )


register_rough("v0", "Go1RoughEnvCfg_v0", "skrl_rough_ppo_cfg.yaml")
register_rough("Play-v0", "Go1RoughEnvCfg_v0_PLAY", "skrl_rough_ppo_cfg.yaml")
register_rough("Play-ICRA-v0", "Go1RoughEnvCfg_v0_PLAY_ICRA", "skrl_rough_ppo_cfg.yaml")

register_rough("LongArch-v0", "Go1RoughEnvCfg_v0", "skrl_rough_ppo_cfg_long.yaml")
register_rough("LongArch-Play-v0", "Go1RoughEnvCfg_v0_PLAY", "skrl_rough_ppo_cfg_long.yaml")
register_rough("LongArch-Play-ICRA-v0", "Go1RoughEnvCfg_v0_PLAY_ICRA", "skrl_rough_ppo_cfg_long.yaml")

register_rough("WideArch-v0", "Go1RoughEnvCfg_v0", "skrl_rough_ppo_cfg_wide.yaml")
register_rough("WideArch-Play-v0", "Go1RoughEnvCfg_v0_PLAY", "skrl_rough_ppo_cfg_wide.yaml")
register_rough("WideArch-Play-ICRA-v0", "Go1RoughEnvCfg_v0_PLAY_ICRA", "skrl_rough_ppo_cfg_wide.yaml")


##
# Velocity based rough with long observation history.
##


def register_rough_lh(id_postfix: str, env_cfg: str, skrl_cfg: str):
    register_manager_based_env(
        id="LORL-Go1RoughLongHistory-RL-" + id_postfix,
        env_cfg=f"{velocity.__name__}.go1.rough_long_history.env_cfg:" + env_cfg,
        skrl_cfg=f"{velocity.__name__}.go1.rough_long_history.agents:" + skrl_cfg,
    )


register_rough_lh("v0", "Go1RoughLongHistoryEnvCfg_v0", "skrl_rough_ppo_cfg.yaml")
register_rough_lh("Play-v0", "Go1RoughLongHistoryEnvCfg_v0_PLAY", "skrl_rough_ppo_cfg.yaml")
register_rough_lh("Play-ICRA-v0", "Go1RoughLongHistoryEnvCfg_v0_PLAY_ICRA", "skrl_rough_ppo_cfg.yaml")


##
# Go1 direction based rough.
##


def register_go1_direction(id_postfix: str, env_cfg: str):
    base = f"{direction.__name__}.go1"
    gym.register(
        id="LORL-Go1Direction-RL-" + id_postfix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{base}.rough_rl_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{base}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{base}.agents.rsl_rl_distillation_cfg:Go1DirectionDistillationRunnerCfg"
            ),
        },
    )


register_go1_direction("v0", "Go1RoughEnvCfg_v0")
register_go1_direction("Play-v0", "Go1RoughEnvCfg_v0_PLAY")
register_go1_direction("Distill-v0", "Go1RoughEnvCfg_v0_DISTILL")
register_go1_direction("Play-ICRA-v0", "Go1RoughEnvCfg_v0_PLAY_ICRA")


##
# AlienGo velocity based rough.
##


def register_aliengo_rough(id_postfix: str, env_cfg: str, skrl_cfg: str):
    register_manager_based_env(
        id="LORL-AlienGoRough-RL-" + id_postfix,
        env_cfg=f"{velocity.__name__}.aliengo.rough.env_cfg:" + env_cfg,
        skrl_cfg=f"{velocity.__name__}.aliengo.rough.agents:" + skrl_cfg,
    )


register_aliengo_rough("v0", "AlienGoRoughEnvCfg_v0", "skrl_rough_ppo_cfg.yaml")
register_aliengo_rough("Play-v0", "AlienGoRoughEnvCfg_v0_PLAY", "skrl_rough_ppo_cfg.yaml")


##
# AlienGo direction based rough.
##


def register_aliengo_direction(id_postfix: str, env_cfg: str):
    base = f"{direction.__name__}.aliengo"
    gym.register(
        id="LORL-AlienGoDirection-RL-" + id_postfix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{base}.rough_rl_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{base}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{base}.agents.rsl_rl_distillation_cfg:AlienGoDirectionDistillationRunnerCfg"
            ),
        },
    )


register_aliengo_direction("v0", "AlienGoRoughEnvCfg_v0")
register_aliengo_direction("Play-v0", "AlienGoRoughEnvCfg_v0_PLAY")
register_aliengo_direction("Play-ICRA-v0", "AlienGoRoughEnvCfg_v0_PLAY_ICRA")
register_aliengo_direction("Distill-v0", "AlienGoRoughEnvCfg_v0_DISTILL")
