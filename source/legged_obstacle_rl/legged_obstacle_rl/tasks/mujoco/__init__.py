import gymnasium as gym


def register(id: str, cls_path: str):
    gym.register(id=id, entry_point=f"{__name__}.{cls_path}", disable_env_checker=True)


# Argo Policy
register("LORL-Go1-Argo-MJ-v0", "go1.argo_env:Go1ArgoEnv")
register("LORL-Go1-ArgoH-MJ-v0", "go1.argo_env:Go1ArgoHEnv")

# Velocity Policy
register("LORL-Go1-Velocity-Flat-MJ-v0", "go1.velocity_env:Go1VelocityFlatEnv")
register("LORL-Go1-Velocity-HField-MJ-v0", "go1.velocity_env:Go1VelocityHFieldEnv")

# Direction Policy
register("LORL-Aliengo-Direction-MJ-v0", "aliengo.direction_env:AliengoDirectionProprioEnv")
register("LORL-Aliengo-Direction-ICRA-Flat-MJ-v0", "aliengo.direction_env:AliengoDirectionProprioIcraFlatEnv")
register("LORL-Aliengo-Direction-ICRA-Sloped-MJ-v0", "aliengo.direction_env:AliengoDirectionProprioIcraSlopedEnv")
