"""Deploy a distilled RSL-RL student policy on real Unitree hardware.

Loads a TorchScript policy.pt (exported by play.py). No IsaacLab / Isaac Sim required.

Usage:
    python scripts/rsl_rl/deploy_real.py \\
        --checkpoint logs/.../exported/policy.pt \\
        --hardware go1
"""

import argparse

parser = argparse.ArgumentParser(description="Deploy distilled RSL-RL student on Unitree hardware.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to exported policy.pt (TorchScript).")
parser.add_argument("--hardware", type=str, required=True, choices=["go1", "aliengo"], help="Target hardware module.")
args_cli = parser.parse_args()

import contextlib
import importlib
import os

import torch

torch.set_num_threads(1)


def main():
    checkpoint = os.path.abspath(args_cli.checkpoint)
    hw_mod = importlib.import_module(f"legged_obstacle_rl.deployment.{args_cli.hardware}")

    print(f"Loading policy from {checkpoint}")
    policy = torch.jit.load(checkpoint, map_location="cpu")
    policy.eval()
    policy.reset()

    with contextlib.suppress(KeyboardInterrupt):
        hw_mod.run(policy)


if __name__ == "__main__":
    main()
