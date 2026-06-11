"""AlienGo hardware deployment runtime.

AlienGo-specific SDK spec + gains on top of the shared runtime in ``_common``.

The aliengo ``robot_interface`` (cpython-311 build) carries no ``DT_NEEDED`` link
to ``libunitree_legged_sdk.so``, so the C++ lib is preloaded with ``RTLD_GLOBAL``
before import (see ``_common.load_sdk``). Low-level UDP follows the SDK's own
``example_py/example_position.py``: explicit cmd/state lengths + ``levelFlag``.

Default obs builder is ``build_obs_proprio`` (45-dim, no height scan) — AlienGo
direction policies are proprioception-only. Actions are zeroed for a safe
comms-bringup (robot ramps to home and holds limp); see ``act_rsl_rl``.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _common
from ._common import build_obs_proprio_dir, stop
from .obs_log import log_step

sdk = _common.load_sdk()

# PD regulator
KP = 51.0
KD = 2.0

# Connection
LOWLEVEL = 0xFF
LOW_IP = "192.168.123.10"
LOCAL_PORT = 8082
TARGET_PORT = 8007
LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

LEGGED_TYPE = sdk.LeggedType.Aliengo

# fmt: off
# Joint limits in IsaacLab order: [hip x4, thigh x4, calf x4].
# Source: Unitree aliengo_description URDF (const.xacro), deg->rad:
#   hip +-70 deg, thigh -120..240 deg, calf -159..-37 deg.
Q_LO_ISAAC = np.array([
    -1.2217305, -1.2217305, -1.2217305, -1.2217305,  # hips
    -2.0943951, -2.0943951, -2.0943951, -2.0943951,  # thighs
    -2.7751958, -2.7751958, -2.7751958, -2.7751958,  # calves
], dtype=np.float32)
Q_HI_ISAAC = np.array([
     1.2217305,  1.2217305,  1.2217305,  1.2217305,  # hips
     4.1887902,  4.1887902,  4.1887902,  4.1887902,  # thighs
    -0.6457718, -0.6457718, -0.6457718, -0.6457718,  # calves
], dtype=np.float32)
# fmt: on


def _make_udp(ip: str):
    return sdk.UDP(LOCAL_PORT, ip, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)


def _init_cmd(udp, cmd) -> None:
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL


ALIENGO_SPEC = _common.RobotSpec(
    sdk=sdk,
    legged_type=LEGGED_TYPE,
    make_udp=_make_udp,
    init_cmd=_init_cmd,
    low_ip=LOW_IP,
    kp=KP,
    kd=KD,
    q_lo=Q_LO_ISAAC,
    q_hi=Q_HI_ISAAC,
)


def act_rsl_rl(policy, obs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions = policy(obs_t).squeeze(0).numpy()
    actions = np.zeros_like(actions)  # safe bringup — remove to drive joints
    log_step(obs, actions)  # no-op unless LORL_OBS_LOG is set
    return actions


def run(agent, *, build_obs_fn=build_obs_proprio_dir, act_fn=act_rsl_rl) -> None:
    """Start teleop, hardware, ramp, then run policy until stop or error."""
    _common.run(ALIENGO_SPEC, agent, build_obs_fn=build_obs_fn, act_fn=act_fn)


__all__ = ["run", "stop", "KP", "KD", "LOW_IP", "LEGGED_TYPE"]
