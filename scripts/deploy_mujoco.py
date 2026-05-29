"""Script to deploy a checkpoint of an RL agent from skrl in MuJoCo."""

"""Parse args first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy a checkpoint of an RL agent from skrl in MuJoCo.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--config", type=str, default=None, help="Path to model's config.")
parser.add_argument("--teleop", action="store_true", help="Keyboard teleoperation")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

MIN_SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(MIN_SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={MIN_SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import legged_obstacle_rl.tasks  # noqa: F401
import numpy as np
from legged_obstacle_rl import teleop
from skrl.envs.wrappers.torch import wrap_env

from isaaclab.utils.io.yaml import load_yaml

actions_list = []


def main():
    cfg = load_yaml(args_cli.config)
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    args_cli.seed = random.randint(0, 10000)

    cfg["seed"] = args_cli.seed if args_cli.seed is not None else cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint and ("agent" in args_cli.checkpoint or "policy" in args_cli.checkpoint):
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        print("[ERROR] Cannot load provided checkpoint")

    # Teleoperation
    if args_cli.teleop:
        print("[INFO] Teleoperation enabled. Use WASD/QE/RF keys.")
        teleop.start()

    with gym.make(args_cli.task, render_mode="human") as env:
        try:
            dt = env.dt
        except AttributeError:
            dt = env.unwrapped.dt

        env = wrap_env(env, "gymnasium")

        cfg["trainer"]["close_environment_at_exit"] = False
        cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
        cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        runner = Runner(env, cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        print(f"[INFO] Joint names: {[env.model.joint(i).name for i in range(13)]}")
        runner.agent.load(resume_path)
        runner.agent.set_running_mode("eval")

        print("[DEBUG] State preprocessor in config: ", cfg["agent"]["state_preprocessor"])
        print("[DEBUG] State preprocessor:", getattr(runner.agent, "_state_preprocessor", None))
        print("[DEBUG] Time delta:", dt)

        obs, _ = env.reset()
        while not teleop.state.stop:
            start_time = time.time()

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # actions = torch.round(actions, decimals=2)
                obs, _, _, _, _ = env.step(actions)

                actions_list.append(actions)

            if args_cli.teleop:
                env._unwrapped.vel_cmd[0] = teleop.state.lin_x
                env._unwrapped.vel_cmd[1] = teleop.state.lin_y
                env._unwrapped.vel_cmd[2] = teleop.state.ang_z
                env._unwrapped.z_cmd = teleop.state.height

            env._unwrapped.print_debug()

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)


def plot_actions(action_history, title="Action Commands Over Time"):
    """
    Plots a list of action tensors/arrays.
    Args:
        action_history: List of tensors or numpy arrays [batch_size, num_actions]
    """
    # Convert list of tensors to a single numpy array [timesteps, num_actions]
    # We assume batch size 1, so we take index 0
    if isinstance(action_history[0], torch.Tensor):
        data = torch.stack(action_history).detach().cpu().numpy()
    else:
        data = np.array(action_history)

    # If the data has a batch dimension (e.g., [T, 1, N]), squeeze it
    if data.ndim == 3:
        data = data.squeeze(1)

    timesteps, num_joints = data.shape

    plt.figure(figsize=(12, 6))
    for i in range(num_joints):
        plt.plot(data[:, i], label=f"Joint {i}", alpha=0.8)

    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Action Value (Normalized or Rad)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Only show legend if joint count is manageable
    if num_joints <= 12:
        plt.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()

    import matplotlib.pyplot as plt

    plot_actions(actions_list)
