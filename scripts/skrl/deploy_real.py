"""Script to deploy a policy checkpoint on the real Unitree Go1 via Unitree SDK."""

"""Parse args first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy a policy on the real Unitree Go1.")
parser.add_argument("--task", type=str, default="LORL-Go1Rough-Flat-MJ-v0", help="Mujoco task name for model init.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument(
    "--config",
    type=str,
    default="source/legged_obstacle_rl/legged_obstacle_rl/tasks/manager_based/locomotion/velocity/go1/rough/agents/skrl_rough_ppo_cfg.yaml",
    required=True,
    help="Path to model config YAML.",
)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])

args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import skrl
from packaging import version

MIN_SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(MIN_SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={MIN_SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.agents.torch.base import Agent
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.agents.jax.base import Agent
    from skrl.utils.runner.jax import Runner

import legged_obstacle_rl.tasks.mujoco  # noqa: F401 — registers gymnasium envs
from legged_obstacle_rl.deployment import go1
from legged_obstacle_rl.utils import load_yaml
from skrl.envs.wrappers.torch import wrap_env


def load_agent(cfg: dict) -> Agent:
    resume_path = os.path.abspath(args_cli.checkpoint)
    if not ("agent" in resume_path or "policy" in resume_path):
        raise ValueError(f"Checkpoint path must contain 'agent' or 'policy': {resume_path}")

    with gym.make(args_cli.task) as env:
        env = wrap_env(env, "gymnasium")

        cfg["trainer"]["close_environment_at_exit"] = False
        cfg["agent"]["experiment"]["write_interval"] = 0
        cfg["agent"]["experiment"]["checkpoint_interval"] = 0

        runner = Runner(env, cfg)
        runner.agent.load(resume_path)
        runner.agent.set_running_mode("eval")

        return runner.agent


def main():
    cfg = load_yaml(args_cli.config)

    print("Loading policy.")
    agent = load_agent(cfg)

    try:
        print("Running Go1.")
        go1.run(agent)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping Go1.")
        go1.stop()


if __name__ == "__main__":
    main()
