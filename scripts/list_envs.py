# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `legged_obstacle_rl` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")

args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import legged_obstacle_rl.tasks  # noqa: F401
from prettytable import PrettyTable


def main():
    """Print all environments registered in `legged_obstacle_rl` extension."""

    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "LORL-" in task_spec.id and (args_cli.keyword is None or args_cli.keyword in task_spec.id):
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs.get("env_cfg_entry_point", "N/A"),
                ]
            )
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
