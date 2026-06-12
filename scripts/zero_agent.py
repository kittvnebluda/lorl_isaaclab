# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--print_obs", action="store_true", help="Print observations.")
parser.add_argument(
    "--log_obs",
    action="store_true",
    help="Log env-0 policy obs + action to this .npz (StepLogger format) for sim2real compare.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import legged_obstacle_rl.tasks  # noqa: F401
import torch
from legged_obstacle_rl.deployment.obs_log import flush_step_log, log_step

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    with gym.make(args_cli.task, cfg=env_cfg) as env:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        step = 0
        obs, _ = env.reset()
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs, _, _, _, _ = env.step(actions)
            if args_cli.print_obs:
                print(obs)
            if args_cli.log_obs:
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                log_step(policy_obs[0].cpu().numpy(), actions[0].cpu().numpy())
            if step % 500 == 0:
                flush_step_log()
            step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
