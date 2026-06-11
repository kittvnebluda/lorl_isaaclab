"""Step logger for sim2real / sim2sim observation comparison.

numpy-only — no IsaacSim or Unitree SDK imports — so the *same* logger drops into
both the real deploy loop (``aliengo.act_rsl_rl``) and an IsaacLab/IsaacSim play
script. Each ``log(...)`` call appends one row; arrays are stacked along axis 0
and written to a single ``.npz`` at process exit (and periodically).

Real-side usage is automatic: set ``LORL_OBS_LOG`` to an output path and
``log_step`` (called inside ``act_rsl_rl``) writes there. In IsaacSim, construct a
``StepLogger`` directly and call ``.log(obs=..., action=...)`` each env step with
the *same* obs layout (see ``decode_proprio_dir`` for the 45-dim direction obs).
"""

from __future__ import annotations

import atexit
import os
import time
from typing import Optional

import numpy as np

# 45-dim direction proprio obs layout (must match build_obs_proprio_dir / the
# IsaacLab direction-task policy group order).
PROPRIO_DIR_SLICES = {
    "joint_pos": slice(0, 12),  # q - home, IsaacLab order
    "base_ang_vel": slice(12, 15),  # gyro, body frame
    "joint_vel": slice(15, 27),  # dq, IsaacLab order
    "projected_gravity": slice(27, 30),
    "direction_command": slice(30, 33),  # [cos psi, sin psi, turn]
    "last_action": slice(33, 45),
}


def decode_proprio_dir(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 45-dim direction proprio obs into its named groups (for readable diffs)."""
    obs = np.asarray(obs).ravel()
    assert obs.shape == (45,), f"expected 45-dim proprio obs, got {obs.shape}"
    return {k: obs[s] for k, s in PROPRIO_DIR_SLICES.items()}


class StepLogger:
    """Append one row per ``log`` call; dump a stacked ``.npz`` at exit.

    All kwargs to ``log`` are recorded as float arrays under their keyword name;
    every row must pass the same set of keys. A monotonic ``t`` (seconds since the
    first ``log``) is added automatically.
    """

    def __init__(self, path: str, *, flush_every: int = 500) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._cols: dict[str, list[np.ndarray]] = {}
        self._t: list[float] = []
        self._t0: Optional[float] = None
        self._flush_every = flush_every
        atexit.register(self.save)

    def log(self, **arrays: np.ndarray) -> None:
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self._t.append(now - self._t0)
        for k, v in arrays.items():
            self._cols.setdefault(k, []).append(np.asarray(v, dtype=np.float32).ravel())
        if len(self._t) % self._flush_every == 0:
            self.save()

    def save(self) -> None:
        if not self._t:
            return
        out = {"t": np.asarray(self._t, dtype=np.float64)}
        for k, rows in self._cols.items():
            out[k] = np.stack(rows)
        np.savez(self.path, **out)
        print(f"[obs_log] wrote {len(self._t)} rows -> {self.path}")


# --- real-deploy convenience: env-var gated singleton -----------------------

_LOGGER: Optional[StepLogger] = None


def log_step(obs: np.ndarray, action: np.ndarray) -> None:
    """Log one (obs, action) step iff ``LORL_OBS_LOG`` is set; else no-op.

    Called from ``act_rsl_rl``. The path in ``LORL_OBS_LOG`` is the output ``.npz``.
    Safe to leave wired in production — does nothing unless the env var is present.
    """
    global _LOGGER
    path = os.environ.get("LORL_OBS_LOG")
    if path is None:
        return
    if _LOGGER is None:
        _LOGGER = StepLogger(path)
        print(f"[obs_log] logging obs+action to {_LOGGER.path} (set by LORL_OBS_LOG)")
    _LOGGER.log(obs=obs, action=action)
