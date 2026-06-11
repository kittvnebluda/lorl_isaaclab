"""Go1 hardware deployment runtime.

Go1-specific SDK spec + gains on top of the shared runtime in ``_common``. The
go1-branch ``robot_interface`` self-resolves its C++ symbols (cpython-38 build
has the ``DT_NEEDED`` link), so no preload is needed here.
"""

from __future__ import annotations

import numpy as np

from . import _common
from ._common import stop

sdk = _common.load_sdk()

# PD regulator
KP = 35.0
KD = 1.5

# Connection
LOWLEVEL = 0xFF
LOW_IP = "192.168.123.10"
LOW_LOCAL_PORT = 8080
LOW_ROBOT_PORT = 8007

LEGGED_TYPE = sdk.LeggedType.Go1

Q_LO_ISAAC = np.array(
    [-0.863, -0.863, -0.863, -0.863, -0.686, -0.686, -0.686, -0.686, -2.818, -2.818, -2.818, -2.818],
    dtype=np.float32,
)
Q_HI_ISAAC = np.array(
    [0.863, 0.863, 0.863, 0.863, 4.501, 4.501, 4.501, 4.501, -0.888, -0.888, -0.888, -0.888],
    dtype=np.float32,
)

GO1_SPEC = _common.RobotSpec(
    sdk=sdk,
    legged_type=LEGGED_TYPE,
    make_udp=lambda ip: sdk.UDP(LOWLEVEL, LOW_LOCAL_PORT, ip, LOW_ROBOT_PORT),
    init_cmd=lambda udp, cmd: udp.InitCmdData(cmd),
    low_ip=LOW_IP,
    kp=KP,
    kd=KD,
    q_lo=Q_LO_ISAAC,
    q_hi=Q_HI_ISAAC,
)


def run(agent, **kwargs) -> None:
    """Start teleop, hardware, ramp, then run the policy until stop or error."""
    _common.run(GO1_SPEC, agent, **kwargs)


__all__ = ["run", "stop", "KP", "KD", "LOW_IP", "LEGGED_TYPE"]
