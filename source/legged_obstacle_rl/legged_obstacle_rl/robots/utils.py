"""Local asset resolution for robot USDs.

Robot USDs default to NVIDIA's remote Nucleus server, so every launch streams them over the
network (slow, prone to timeouts). This module redirects to a local copy when one exists.

Layout: a local asset root mirrors the Nucleus tree, e.g.
``<root>/Isaac/Robots/Unitree/aliengo/aliengo.usd``. The root is, in order of precedence:

1. ``$LORL_ASSETS_DIR`` if set,
2. otherwise ``~/.cache/legged_obstacle_rl/assets``.

Populate it once (online) with ``scripts/download_assets.py``. After that, training starts
offline. If a file is missing locally, the resolver falls back to the original Nucleus URL,
so nothing breaks before the assets are downloaded.
"""

from __future__ import annotations

import os

MARKER = "/Isaac/"


def local_assets_root() -> str:
    """The local asset root directory (created lazily by the downloader)."""
    env = os.environ.get("LORL_ASSETS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".cache", "legged_obstacle_rl", "assets")


def local_path(src: str) -> str:
    """Map a Nucleus folder URL to its local mirror path (from the 'Isaac/' segment)."""
    src = src[src.find(MARKER) + 1 :]
    return os.path.join(local_assets_root(), src)


def resolve_usd(nucleus_url: str) -> str:
    """Return a local USD path if present, else the original Nucleus URL.

    Args:
        nucleus_url: The full remote URL, e.g.
            ``{ISAAC_NUCLEUS_DIR}/Robots/Unitree/aliengo/aliengo.usd`` or
            ``{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd``.

    Returns:
        Local absolute path if the file exists on disk, otherwise ``nucleus_url`` unchanged.
    """
    idx = nucleus_url.find(MARKER)
    if idx == -1:
        print(f"[WARNING] NUCLEUS URL does not contain '{MARKER}', unable to resolve local path")
        return nucleus_url

    file_path = local_path(nucleus_url)
    if os.path.isfile(file_path):
        return os.path.abspath(file_path)
    return nucleus_url
