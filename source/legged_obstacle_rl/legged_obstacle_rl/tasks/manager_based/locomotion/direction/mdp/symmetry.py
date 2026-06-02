"""Left-right (sagittal) symmetry augmentation for the Aliengo direction task.

Used by rsl_rl PPO data augmentation (``RslRlSymmetryCfg``). The augmentation function
mirrors every observation group the actor/critic read (``policy`` and ``privilliged``) plus
the action vector across the robot's sagittal (x-z) plane, returning a 2x batch
``[original, left_right]``.

Index maps are specific to:

* Aliengo Isaac joint order (type-grouped):
  ``[FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh,
     FL_calf, FR_calf, RL_calf, RR_calf]`` (see tasks/mujoco/velocity_env.py).
* Foot scanners ordered ``[FL, FR, RL, RR]``, each a ``CircularPatternCfg`` scan laid out as
  ``[center, ring(6), ring(12), ring(18)]`` CCW from theta=0 (see sensors/patterns.py).

Slices are resolved at call time from the observation manager term dims, so the function does
not hard-code byte offsets. Only the per-term transforms are robot-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# Left<->right joint permutation: swap FL<->FR and RL<->RR within each 4-block (hip/thigh/calf).
_LR_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
# Hip (HAA) joint columns whose sign flips under a sagittal mirror.
_HIP_IDX = [0, 1, 2, 3]
# Foot scanner block order [FL, FR, RL, RR] -> [FR, FL, RR, RL].
_FOOT_PERM = [1, 0, 3, 2]
# Per-foot circular-scan ring sizes (must match CircularPatternCfg.num_points).
_RING_SIZES = (6, 12, 18)


def _scan_ring_perm(ring_sizes=_RING_SIZES) -> list[int]:
    """Index permutation that reflects one foot scan across the sagittal plane (theta -> -theta).

    Layout: ``[center, ring0, ring1, ...]``; center stays, point ``i`` of an ``n``-point ring
    maps to ``(n - i) % n``.
    """
    perm = [0]
    offset = 1
    for n in ring_sizes:
        perm.extend(offset + ((n - i) % n) for i in range(n))
        offset += n
    return perm


_SCAN_PERM = _scan_ring_perm()


def _switch_joints_lr(x: torch.Tensor) -> torch.Tensor:
    """Mirror a length-12 joint vector left<->right and flip hip signs."""
    out = x[..., _LR_PERM].clone()
    out[..., _HIP_IDX] *= -1.0
    return out


def _switch_gains_lr(x: torch.Tensor) -> torch.Tensor:
    """Mirror a length-12 gain vector left<->right (magnitudes, no sign flip)."""
    return x[..., _LR_PERM].clone()


def _switch_foot_scans_lr(x: torch.Tensor, scan_len: int) -> torch.Tensor:
    """Mirror the 4 concatenated foot scans: swap FL/FR & RL/RR blocks, reflect each ring."""
    out = x.clone()
    for dst, src in enumerate(_FOOT_PERM):
        d0, s0 = dst * scan_len, src * scan_len
        block = x[..., s0 : s0 + scan_len]
        out[..., d0 : d0 + scan_len] = block[..., _SCAN_PERM]
    return out


def _vec_sign(x: torch.Tensor, signs: list[float]) -> torch.Tensor:
    """Multiply a 3-vector slice by per-axis signs."""
    return x * torch.tensor(signs, device=x.device, dtype=x.dtype)


# Per-term transforms. Each maps a (B, dim) slice -> mirrored (B, dim).
# Keyed by the ObservationTerm attribute name in the env cfg.
def _t_joint(x):
    return _switch_joints_lr(x)


def _t_ang_vel(x):
    return _vec_sign(x, [-1.0, 1.0, -1.0])


def _t_lin_vel(x):
    return _vec_sign(x, [1.0, -1.0, 1.0])


def _t_gravity(x):
    return _vec_sign(x, [1.0, -1.0, 1.0])


def _t_direction_cmd(x):
    # [cos(yaw), sin(yaw), turn_dir] in base frame -> reflect heading y and turn sign.
    return _vec_sign(x, [1.0, -1.0, -1.0])


def _t_force(x):
    return _vec_sign(x, [1.0, -1.0, 1.0])


def _t_torque(x):
    return _vec_sign(x, [-1.0, 1.0, -1.0])


def _t_gains(x):
    # stiffness(12) + damping(12): permute each half, no sign flip.
    half = x.shape[-1] // 2
    out = x.clone()
    out[..., :half] = _switch_gains_lr(x[..., :half])
    out[..., half:] = _switch_gains_lr(x[..., half:])
    return out


def _t_foot_contacts(x):
    # [FL, FR, RL, RR] -> [FR, FL, RR, RL]
    return x[..., _FOOT_PERM].clone()


_TERM_TRANSFORMS = {
    "joint_pos": _t_joint,
    "joint_vel": _t_joint,
    "actions": _t_joint,
    "base_ang_vel": _t_ang_vel,
    "base_lin_vel": _t_lin_vel,
    "gravity": _t_gravity,
    "direction_commands": _t_direction_cmd,
    "forces": _t_force,
    "torques": _t_torque,
    "actuator_gains": _t_gains,
    "foot_contacts": _t_foot_contacts,
    # foot scans handled separately (need scan_len), see below.
    "fl_foot_scan": None,
    "fr_foot_scan": None,
    "rl_foot_scan": None,
    "rr_foot_scan": None,
}

_FOOT_SCAN_TERMS = ("fl_foot_scan", "fr_foot_scan", "rl_foot_scan", "rr_foot_scan")


def _transform_group(env: "ManagerBasedRLEnv", group: str, obs: torch.Tensor) -> torch.Tensor:
    """Apply the left-right mirror to a concatenated observation group tensor.

    Slices the flat group tensor by the observation manager's term dims and dispatches each
    term to its transform. The 4 foot scans are mirrored jointly (block swap + ring reflect).
    """
    om = env.observation_manager
    term_names = om.active_terms[group]
    term_dims = [int(d[0]) if isinstance(d, (tuple, list)) else int(d) for d in om.group_obs_term_dim[group]]

    out = obs.clone()
    # collect the contiguous foot-scan span so we can mirror it as one unit
    scan_start = scan_len = None
    offset = 0
    for name, dim in zip(term_names, term_dims):
        sl = slice(offset, offset + dim)
        if name in _FOOT_SCAN_TERMS:
            if scan_start is None:
                scan_start = offset
                scan_len = dim
            # else: accumulate, all foot scans share scan_len
        elif name in _TERM_TRANSFORMS and _TERM_TRANSFORMS[name] is not None:
            out[..., sl] = _TERM_TRANSFORMS[name](obs[..., sl])
        # unknown terms left as-is (identity)
        offset += dim

    if scan_start is not None:
        span = slice(scan_start, scan_start + 4 * scan_len)
        out[..., span] = _switch_foot_scans_lr(obs[..., span], scan_len)
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: torch.Tensor | None = None,
):
    """rsl_rl symmetry augmentation: append a sagittally-mirrored copy (2x batch).

    Args:
        env: the (vec) environment, used to read observation-manager term layout.
        obs: observation TensorDict keyed by group name, or None.
        actions: action tensor (num_envs, 12), or None.

    Returns:
        ``(obs_aug, actions_aug)`` each ``[original, left_right]`` stacked along the batch, or
        None for whichever input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][:batch_size] = obs[group][:]
            obs_aug[group][batch_size:] = _transform_group(env.unwrapped, group, obs[group])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _switch_joints_lr(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
