# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import importlib
import pkgutil
import sys
from typing import Callable


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    package = importlib.import_module(package_name)
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: Callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p: str, m: dict[str, bool] = {}) -> bool:
        """Check if a package has been seen before."""
        if p in m:
            return True
        m[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)


_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
