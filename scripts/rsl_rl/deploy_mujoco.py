"""Deploy an rsl_rl direction policy in MuJoCo (sim2sim).

Loads the TorchScript policy exported by ``scripts/rsl_rl/play.py`` (``exported/policy.pt``)
and drives one of the MuJoCo direction envs (``LORL-Go1Direction-MJ-v0`` /
``LORL-AlienGoDirection-MJ-v0``). With ``--teleop`` the keyboard sets the direction command.
"""

"""Parse args first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy an rsl_rl direction policy in MuJoCo.")
parser.add_argument("--task", type=str, default=None, help="Name of the MuJoCo task (e.g. LORL-Go1Direction-MJ-v0).")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to the exported TorchScript policy (exported/policy.pt)."
)
parser.add_argument("--teleop", action="store_true", help="Keyboard teleoperation.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

args_cli = parser.parse_args()

# ActuatorNetMLP (imported transitively by the MuJoCo envs) needs the Isaac app context.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import time

import gymnasium as gym
import legged_obstacle_rl.tasks  # noqa: F401
import torch
from legged_obstacle_rl import teleop


def main():
    if args_cli.task is None or args_cli.checkpoint is None:
        raise ValueError("--task and --checkpoint are required")

    print(f"[INFO] Loading TorchScript policy from: {args_cli.checkpoint}")
    policy = torch.jit.load(args_cli.checkpoint)
    policy.eval()

    if args_cli.teleop:
        print("[INFO] Teleoperation enabled. Move: I/K (Vx), J/L (Vy), U/O (Wz). Quit: ESC.")
        teleop.start()

    with gym.make(args_cli.task, render_mode="human") as env:
        unwrapped = env.unwrapped
        try:
            dt = env.dt
        except AttributeError:
            dt = unwrapped.dt

        obs, _ = env.reset()
        while not teleop.state.stop:
            start_time = time.time()

            if args_cli.teleop:
                unwrapped.inject_teleop(teleop.state)

            with torch.inference_mode():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                actions = policy(obs_t).squeeze(0).numpy()
                obs, _, _, _, _ = env.step(actions)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
    simulation_app.close()
