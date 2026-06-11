"""Go1 deployment runtime.

Three concurrent threads:

- Teleop (started by ``teleop.start()``): keyboard input -> velocity command and
  stop flag.
- Hardware I/O (``Go1Hardware._io_loop``, 500 Hz): single owner of the SDK UDP
  socket. Reads ``LowState``, publishes ``SensorSnapshot``, consumes the latest
  ``MotorTarget``, applies ``PowerProtect``, sends ``LowCmd``.
- Policy (``PolicyRunner._policy_loop``, 50 Hz): reads latest snapshot, updates
  the estimator, builds the observation, calls the agent, publishes a new
  ``MotorTarget``.

Shared state lives in two single-slot publishers on ``Go1Hardware``
(``_snap`` and ``_target``), each protected by its own lock and held only for
the duration of an assignment or read of the slot.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch

from legged_obstacle_rl import teleop
from legged_obstacle_rl.tasks.mujoco.utils import map_indexes, normalize, quat_apply_inverse

sys.path.insert(0, os.path.expanduser("~/Projects/unitree_legged_sdk/lib/python/amd64"))
import robot_interface as sdk  # pyright: ignore[reportMissingImports]

# fmt: off
# Joint Order in MuJoCo
mujoco_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# Joint Order in IsaacLab
isaac_joint_names = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]
# Home joint positions in IsaacLab
isaac_home_jpos = np.array([
     0.1, -0.1,  0.1, -0.1, # hips
     0.8,  0.8,  1.0,  1.0, # thighs
    -1.5, -1.5, -1.5, -1.5, # calves
])
# fmt: on
isaac_to_mujoco_joints = map_indexes(target=mujoco_joint_names, source=isaac_joint_names)
mujoco_to_isaac_joints = map_indexes(target=isaac_joint_names, source=mujoco_joint_names)

ZERO_ACTION = np.zeros(12, dtype=np.float32)
GRAVITY_VEC = normalize(np.array([[0.0, 0.0, -9.81]], dtype=np.float32)).squeeze(0)

# PD regulator
KP = 35.0
KD = 1.5

# Connection
LOWLEVEL = 0xFF
LOW_IP = "192.168.123.10"
LOW_LOCAL_PORT = 8080
LOW_ROBOT_PORT = 8007

# Ramp
DT = 0.02  # 50 Hz policy
RAMP_STEPS = 300  # startup ramp to home position
RAMP_KP = 5.0  # soft gains during ramp
RAMP_KD = 1.0

# Shutdown
DAMPING_KD = 2.0  # servo Kd while robot settles after stop
DAMPING_SETTLE_S = 3.0  # seconds of active damping before motor cut

# Height scanner
ISAAC_OFFSET = 0.5
HS_SIZE = 187

# Loops
COMM_LOOP_DT = 0.002
POLICY_LOOP_DT = 0.02

# Velocity estimator
G_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float32)
VEL_DECAY = 0.98  # ~1 s time constant at DT=0.02

# FK leg geometry (Go1)
_L_THIGH = 0.213
_L_CALF = 0.213
# SDK calf motor indices for legs in isaac order [FL, FR, RL, RR]
_CALF_SDK_IDX = [5, 2, 11, 8]
_CONTACT_TAU_THRESH = 3.0  # Nm; |tauEst| below this -> swing leg

ACTION_SCALE = 0.25
DEFAULT_BASE_HEIGHT = 0.28

# Joint limits in IsaacLab order: [hip x4, thigh x4, calf x4]
# Source: unitree_go1/go1.xml and go1.urdf
_Q_LO_ISAAC = np.array(
    [-0.863, -0.863, -0.863, -0.863, -0.686, -0.686, -0.686, -0.686, -2.818, -2.818, -2.818, -2.818],
    dtype=np.float32,
)
_Q_HI_ISAAC = np.array(
    [0.863, 0.863, 0.863, 0.863, 4.501, 4.501, 4.501, 4.501, -0.888, -0.888, -0.888, -0.888],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorSnapshot:
    """Immutable snapshot of one SDK low-state read, in IsaacLab joint order."""

    t: float
    q_isaac: np.ndarray  # (12,)
    dq_isaac: np.ndarray  # (12,)
    tau_est_sdk: np.ndarray  # (12,) raw SDK order, only used for contact detection
    quat: np.ndarray  # (4,) w,x,y,z
    accel_body: np.ndarray  # (3,) body-frame specific force
    gyro_body: np.ndarray  # (3,) body-frame rad/s


@dataclass(frozen=True)
class MotorTarget:
    """Latest joint target written by the policy thread."""

    q_isaac: np.ndarray  # (12,)
    kp: float
    kd: float


@dataclass
class EstimatorState:
    """Linear velocity (integrated body-frame accel) + base height (leg FK)."""

    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_height: float = DEFAULT_BASE_HEIGHT

    def update(self, s: SensorSnapshot, dt: float) -> None:
        g_body = quat_apply_inverse(s.quat, G_WORLD)
        self.lin_vel = VEL_DECAY * self.lin_vel + (s.accel_body + g_body) * dt
        self.base_height = estimate_base_height(s.q_isaac, s.tau_est_sdk)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def estimate_base_height(q_isaac: np.ndarray, tau_est_sdk: np.ndarray) -> float:
    """Base height from 2-D leg FK, averaging only stance legs.

    Contact detected via calf ``tauEst >= _CONTACT_TAU_THRESH``. Falls back to
    all 4 legs if none in contact (airborne).
    """
    leg_h = [_L_THIGH * np.cos(q_isaac[4 + i]) + _L_CALF * np.cos(q_isaac[4 + i] + q_isaac[8 + i]) for i in range(4)]
    stance = [i for i in range(4) if abs(tau_est_sdk[_CALF_SDK_IDX[i]]) >= _CONTACT_TAU_THRESH]
    legs = stance if stance else range(4)
    return sum(leg_h[i] for i in legs) / len(legs)


def build_obs(
    s: SensorSnapshot,
    est: EstimatorState,
    vel_cmd: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Construct the 235-dim observation vector from sensor + estimator + command."""
    gravity = quat_apply_inverse(s.quat, GRAVITY_VEC)
    height_scan = np.full(HS_SIZE, est.base_height - ISAAC_OFFSET, dtype=np.float32)
    obs = np.concatenate(
        [
            s.q_isaac - isaac_home_jpos,
            est.lin_vel,
            s.gyro_body,
            s.dq_isaac,
            gravity,
            vel_cmd,
            last_action,
            height_scan,
        ]
    )
    assert obs.shape == (235,), f"Obs size mismatch: {obs.shape}"
    return obs


def build_obs_proprio(
    s: SensorSnapshot,
    est: EstimatorState,
    vel_cmd: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Construct 45-dim proprio-only obs for direction-task distilled student (no height scan).

    vel_cmd is [lin_x, lin_y, ang_z] from teleop; converted to [cos θ, sin θ, ω].
    Obs order matches IsaacLab direction-task policy group:
    joint_pos → base_ang_vel → joint_vel → projected_gravity → direction_commands → last_action
    """
    lin_x, lin_y, ang_z = vel_cmd
    mag = np.hypot(lin_x, lin_y)
    yaw = np.arctan2(lin_y, lin_x) if mag > 1e-3 else 0.0
    dir_cmd = np.array([np.cos(yaw), np.sin(yaw), ang_z], dtype=np.float32)
    gravity = quat_apply_inverse(s.quat, GRAVITY_VEC)
    obs = np.concatenate(
        [
            s.q_isaac - isaac_home_jpos,  # 12
            s.gyro_body,  #  3
            s.dq_isaac,  # 12
            gravity,  #  3
            dir_cmd,  #  3
            last_action,  # 12
        ]
    )
    assert obs.shape == (45,), f"Obs size mismatch: {obs.shape}"
    return obs


def pace_loop(dt: float, stop: threading.Event) -> Iterator[float]:
    """Yield once per ``dt`` seconds until ``stop`` is set. Drift-free."""
    next_t = time.perf_counter()
    while not stop.is_set():
        yield next_t
        next_t += dt
        rem = next_t - time.perf_counter()
        if rem > 0:
            stop.wait(rem)


# ---------------------------------------------------------------------------
# Hardware I/O
# ---------------------------------------------------------------------------


class Go1Hardware:
    """Single owner of the Unitree SDK UDP socket.

    One thread runs ``_io_loop`` at ``COMM_LOOP_DT``: ``Recv`` -> publish a
    ``SensorSnapshot`` -> apply the latest ``MotorTarget`` -> ``PowerProtect`` ->
    ``Send``. Snap and target each live in a one-slot publisher under a
    dedicated lock.
    """

    def __init__(self, low_ip: str = LOW_IP, legged_type=sdk.LeggedType.Go1):
        self._udp = sdk.UDP(LOWLEVEL, LOW_LOCAL_PORT, low_ip, LOW_ROBOT_PORT)
        self._safe = sdk.Safety(legged_type)
        self._cmd = sdk.LowCmd()
        self._low_state = sdk.LowState()
        self._udp.InitCmdData(self._cmd)

        self._snap_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._snap: Optional[SensorSnapshot] = None
        self._target: Optional[MotorTarget] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def snap(self) -> Optional[SensorSnapshot]:
        """Access sensor snapshot with a lock"""
        with self._snap_lock:
            return self._snap

    @snap.setter
    def snap(self, s: SensorSnapshot):
        """Access sensor snapshot with a lock"""
        with self._snap_lock:
            self._snap = s

    @property
    def target(self) -> Optional[MotorTarget]:
        """Access motor target with a lock"""
        with self._target_lock:
            return self._target

    @target.setter
    def target(self, target: MotorTarget) -> None:
        """Access motor target with a lock"""
        with self._target_lock:
            self._target = target

    def start(self) -> None:
        self._thread = threading.Thread(target=self._io_loop, name="go1-io")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def _read_snapshot(self) -> SensorSnapshot:
        s = self._low_state
        q_sdk = np.array([s.motorState[i].q for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([s.motorState[i].dq for i in range(12)], dtype=np.float32)
        tau_sdk = np.array([s.motorState[i].tauEst for i in range(12)], dtype=np.float32)
        return SensorSnapshot(
            t=time.perf_counter(),
            q_isaac=q_sdk[mujoco_to_isaac_joints],
            dq_isaac=dq_sdk[mujoco_to_isaac_joints],
            tau_est_sdk=tau_sdk,
            quat=np.array(s.imu.quaternion, dtype=np.float32),
            accel_body=np.array(s.imu.accelerometer, dtype=np.float32),
            gyro_body=np.array(s.imu.gyroscope, dtype=np.float32),
        )

    def _send(self):
        self._safe.PowerProtect(self._cmd, self._low_state, 1)
        self._fill_damping()
        self._udp.SetSend(self._cmd)
        self._udp.Send()

    def _recv(self):
        self._udp.Recv()
        self._udp.GetRecv(self._low_state)

    def _fill_damping(self) -> None:
        """Servo mode, zero Kp, nonzero Kd — robot settles gently under gravity."""
        for i in range(12):
            self._cmd.motorCmd[i].mode = 0x0A  # servo
            self._cmd.motorCmd[i].Kp = 0.0
            self._cmd.motorCmd[i].Kd = DAMPING_KD
            self._cmd.motorCmd[i].tau = 0.0
            self._cmd.motorCmd[i].q = 0.0
            self._cmd.motorCmd[i].dq = 0.0

    def _fill_idle(self) -> None:
        """Cut motor power entirely after robot has settled."""
        for i in range(12):
            self._cmd.motorCmd[i].mode = 0x00  # cut off power
            self._cmd.motorCmd[i].Kp = 0.0
            self._cmd.motorCmd[i].Kd = 0.0
            self._cmd.motorCmd[i].tau = 0.0
            self._cmd.motorCmd[i].q = 0.0
            self._cmd.motorCmd[i].dq = 0.0

    def _fill_cmd(self, target_q_isaac: np.ndarray, kp: float, kd: float) -> None:
        """Write a PD position target to ``cmd``, converting IsaacLab -> SDK joint order."""
        if not np.all(np.isfinite(target_q_isaac)):
            raise RuntimeError(f"NaN/Inf in joint target: {target_q_isaac}")
        q = np.clip(target_q_isaac, _Q_LO_ISAAC, _Q_HI_ISAAC)[isaac_to_mujoco_joints]
        for i in range(12):
            self._cmd.motorCmd[i].mode = 0x0A  # servo
            self._cmd.motorCmd[i].q = float(q[i])
            self._cmd.motorCmd[i].dq = 0.0
            self._cmd.motorCmd[i].Kp = kp
            self._cmd.motorCmd[i].Kd = kd
            self._cmd.motorCmd[i].tau = 0.0

    def _io_loop(self) -> None:
        try:
            print_t = time.perf_counter()
            for tick in pace_loop(COMM_LOOP_DT, self._stop):
                self._recv()
                self.snap = self._read_snapshot()
                tgt = self.target
                if tgt is not None:
                    self._fill_cmd(tgt.q_isaac, tgt.kp, tgt.kd)
                self._send()

                now = time.perf_counter()
                if now - print_t > 1.0:
                    print(f"IO LOOP WORK: {now - tick:.4f}")
                    print_t = now
        finally:
            print(f"Damping for {DAMPING_SETTLE_S:.0f} s, then cutting power...")
            self._fill_damping()
            deadline = time.perf_counter() + DAMPING_SETTLE_S
            while time.perf_counter() < deadline:
                self._recv()
                self._send()
                time.sleep(COMM_LOOP_DT)
            self._fill_idle()
            self._send()
            print("Motors cut.")


# ---------------------------------------------------------------------------
# Policy runner
# ---------------------------------------------------------------------------


def _act(agent, obs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        out = agent.act(torch.from_numpy(obs).float().unsqueeze(0), timestep=0, timesteps=0)  # pyright: ignore
        actions = out[-1].get("mean_actions", out[0])  # (1, 12)
    return actions.squeeze(0).cpu().numpy()


class PolicyRunner:
    """Owns the policy thread and the policy-side state (estimator, last action)."""

    def __init__(
        self,
        hw: Go1Hardware,
        agent,
        teleop_,
        *,
        kp: float = KP,
        kd: float = KD,
        build_obs_fn=None,
        act_fn=None,
    ) -> None:
        self.stop_event = threading.Event()

        self._hw = hw
        self._agent = agent
        self._teleop = teleop_
        self._kp = kp
        self._kd = kd
        self._build_obs = build_obs_fn if build_obs_fn is not None else build_obs
        self._act_fn = act_fn if act_fn is not None else _act
        self._est = EstimatorState()
        self._last_action = np.zeros(12, dtype=np.float32)
        self._thread: Optional[threading.Thread] = None

    def _wait_for_valid_snap(self, timeout_s: float = 5.0) -> SensorSnapshot:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.stop_event.is_set() or self._teleop.state.stop:
                raise RuntimeError("aborted while waiting for SDK state")
            snap = self._hw.snap
            print(snap)
            if snap is not None and np.any(snap.q_isaac != 0.0):
                return snap
            time.sleep(0.01)
        raise TimeoutError(f"no valid SDK state within {timeout_s:.1f} s")

    def ramp_to_home(self) -> None:
        print("Waiting for valid state from robot...")
        snap = self._wait_for_valid_snap()
        q0 = snap.q_isaac.copy()
        print(f"Got valid state: {q0}")
        print(f"Ramping to home position over {RAMP_STEPS * DT:.1f} s ...")
        for step in range(RAMP_STEPS):
            if self.stop_event.is_set() or self._teleop.state.stop:
                raise RuntimeError("aborted during ramp")
            t = step / (RAMP_STEPS - 1)
            target_q = (1.0 - t) * q0 + t * isaac_home_jpos
            self._hw.target = MotorTarget(target_q, RAMP_KP, RAMP_KD)
            time.sleep(DT)
        print("Ramp complete.")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._policy_loop, name="go1-policy")
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _policy_loop(self) -> None:
        try:
            print_t = time.perf_counter()
            for tick in pace_loop(POLICY_LOOP_DT, self.stop_event):
                if self._teleop.state.stop:
                    break

                snap = self._hw.snap
                if snap is None:
                    continue

                vel_cmd = np.array(
                    [self._teleop.state.lin_x, self._teleop.state.lin_y, self._teleop.state.ang_z],
                    dtype=np.float32,
                )
                self._est.update(snap, POLICY_LOOP_DT)
                obs = self._build_obs(snap, self._est, vel_cmd, self._last_action)
                action = self._act_fn(self._agent, obs)

                self._hw.target = MotorTarget(isaac_home_jpos + action * ACTION_SCALE, self._kp, self._kd)
                self._last_action = action.copy()

                now = time.perf_counter()
                if now - print_t > 1.0:
                    print(f"IMU GYRO: {snap.gyro_body}")
                    print(f"LIN_VEL_EST: {self._est.lin_vel}")
                    print(f"HEIGHT_EST: {self._est.base_height}")
                    print(f"POLICY LOOP WORK: {now - tick:.4f}")
                    print_t = now
        finally:
            self.stop_event.set()


# ---------------------------------------------------------------------------
# Module-level facade
# ---------------------------------------------------------------------------


_session_lock = threading.Lock()
_session: Optional[tuple[Go1Hardware, PolicyRunner]] = None


def run(
    agent,
    *,
    kp: float = KP,
    kd: float = KD,
    low_ip: str = LOW_IP,
    legged_type=sdk.LeggedType.Go1,
    build_obs_fn=None,
    act_fn=None,
) -> None:
    """Start teleop, hardware, ramp, then run the policy until stop or error."""
    global _session

    hw = Go1Hardware(low_ip=low_ip, legged_type=legged_type)
    runner = PolicyRunner(hw, agent, teleop, kp=kp, kd=kd, build_obs_fn=build_obs_fn, act_fn=act_fn)
    with _session_lock:
        _session = (hw, runner)

    try:
        teleop.start()
        hw.start()
        runner.ramp_to_home()
        runner.start()
        runner.stop_event.wait()
    finally:
        runner.stop()
        hw.stop()
        with _session_lock:
            _session = None


def stop() -> None:
    """Idempotent shutdown. Safe to call from any thread."""
    with _session_lock:
        sess = _session
    if sess is None:
        return
    hw, runner = sess
    runner.stop()
    hw.stop()


__all__ = [
    "run",
    "stop",
    "Go1Hardware",
    "PolicyRunner",
    "SensorSnapshot",
    "MotorTarget",
    "EstimatorState",
    "build_obs",
    "build_obs_proprio",
    "estimate_base_height",
    "pace_loop",
    "GRAVITY_VEC",
    "isaac_home_jpos",
    "KP",
    "KD",
    "LOW_IP",
]
