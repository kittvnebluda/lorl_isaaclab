import gymnasium as gym


def register(id: str, cls_path: str):
    gym.register(id=id, entry_point=f"{__name__}.{cls_path}", disable_env_checker=True)


# Argo Policy
register("LORL-Go1Argo-MJ-v0", "go1_argo_env:Go1ArgoEnv")
register("LORL-Go1ArgoH-MJ-v0", "go1_argo_env:Go1ArgoHEnv")

# Rough Policy
register("LORL-Go1Rough-Flat-MJ-v0", "rough.flat_env:Go1RoughFlatEnv")
register("LORL-Go1Rough-HField-MJ-v0", "rough.hfield_env:Go1RoughHFieldEnv")
