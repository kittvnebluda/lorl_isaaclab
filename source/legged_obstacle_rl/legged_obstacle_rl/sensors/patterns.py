"""Custom ray-cast patterns for legged_obstacle_rl.

Concentric-ring (circular) pattern for egocentric foot height scans, as used in
robust locomotion architectures (Miki et al, Lee et al). Lives in this package so
IsaacLab source stays untouched on upgrades. A ``RayCasterCfg`` accepts any
``PatternBaseCfg`` subclass via its ``pattern_cfg`` field, so this works directly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


def circular_pattern(cfg: "CircularPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Concentric rings of rays around the sensor origin.

    One ray at the center plus ``num_points[i]`` rays evenly spaced on a ring of
    radius ``radii[i]``. Rays are defined in the sensor's local frame; with the
    sensor's ``ray_alignment="yaw"`` the ring stays flat and tracks only the
    body yaw, so terrain height is read correctly regardless of foot pitch/roll.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays. Total ray count is
        ``1 + sum(num_points)``.

    Raises:
        ValueError: If ``radii`` and ``num_points`` differ in length.
        ValueError: If any radius or point count is non-positive.
    """
    if len(cfg.radii) != len(cfg.num_points):
        raise ValueError(f"radii ({len(cfg.radii)}) must match num_points ({len(cfg.num_points)}).")
    if any(r <= 0 for r in cfg.radii):
        raise ValueError(f"All radii must be > 0. Received: {cfg.radii}.")
    if any(n <= 0 for n in cfg.num_points):
        raise ValueError(f"All num_points must be > 0. Received: {cfg.num_points}.")

    # center ray
    points = [torch.zeros(1, 3, device=device)]
    for r, n in zip(cfg.radii, cfg.num_points):
        # drop the duplicated endpoint at 2*pi
        theta = torch.linspace(0.0, 2.0 * torch.pi, n + 1, device=device)[:-1]
        ring = torch.zeros(n, 3, device=device)
        ring[:, 0] = r * torch.cos(theta)
        ring[:, 1] = r * torch.sin(theta)
        points.append(ring)

    ray_starts = torch.cat(points, dim=0)
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, :] = torch.tensor(list(cfg.direction), device=device)
    return ray_starts, ray_directions


@configclass
class CircularPatternCfg(PatternBaseCfg):
    """Configuration for a concentric-ring ray-cast pattern (foot scan)."""

    func: Callable = circular_pattern

    radii: tuple[float, ...] = (0.08, 0.16, 0.24)
    """Radius of each ring (in meters)."""

    num_points: tuple[int, ...] = (6, 12, 18)
    """Number of rays per ring. Must match the length of :attr:`radii`."""

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Ray direction. Defaults to straight down (0.0, 0.0, -1.0)."""
