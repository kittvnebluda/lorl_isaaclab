import gymnasium as gym


gym.register(
    id="LORL-Go1Argo-MJC-v0",
    entry_point=f"{__name__}.go1_argo_env:Go1ArgoEnv",
    disable_env_checker=True,
)

gym.register(
    id="LORL-Go1ArgoH-MJC-v0",
    entry_point=f"{__name__}.go1_argo_env:Go1ArgoHEnv",
    disable_env_checker=True,
)

gym.register(
    id="LORL-Go1Rough-MJC-v0",
    entry_point=f"{__name__}.go1_rough_env:Go1RoughEnv",
    disable_env_checker=True,
)
