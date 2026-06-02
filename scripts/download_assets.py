"""Download robot USDs from Nucleus to a local cache so training starts offline.

Recursively copies the robot asset folders (with all referenced sub-layers, materials, and
meshes) from the remote Nucleus server into the local asset root used by
``legged_obstacle_rl.robots.assets`` — by default ``~/.cache/legged_obstacle_rl/assets`` or
``$LORL_ASSETS_DIR``. Run once while online; subsequent launches resolve to the local copies
and avoid the network fetch (and its timeouts).

Usage:
    python scripts/download_assets.py [--dest DIR] [--force]
"""

"""Launch Isaac Sim Simulator first (needed for omni.client + Nucleus settings)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Download robot USD assets from Nucleus to a local cache.")
parser.add_argument(
    "--dest",
    type=str,
    default=None,
    help="Local asset root. Defaults to $LORL_ASSETS_DIR or ~/.cache/legged_obstacle_rl/assets.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite local copies if they already exist.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

from legged_obstacle_rl.robots.utils import local_path

import omni.client

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Remote Nucleus folders to mirror recursively. Each URL contains an "/Isaac/" segment; the
# local mirror keeps everything from "Isaac/" onward (same convention as robots/assets.py).
ASSET_FOLDERS = [
    f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/aliengo",
    f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1",
]


def main():
    for src in ASSET_FOLDERS:
        dst = local_path(src)

        if os.path.isdir(dst) and not args_cli.force:
            print(f"[SKIP] Already present (use --force to overwrite): {dst}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"[INFO] Copying:\n    from {src}\n    to   {dst}")
        result = omni.client.copy(src.replace(os.sep, "/"), dst, omni.client.CopyBehavior.OVERWRITE)
        if result != omni.client.Result.OK:
            print(f"[ERROR] Failed to copy {src} (result={result}). Is the Nucleus server reachable?")
        else:
            print(f"[OK] {dst}")

    print("[INFO] Done. Training will now resolve these USDs locally (set $LORL_ASSETS_DIR if you used --dest).")


if __name__ == "__main__":
    main()
    simulation_app.close()
