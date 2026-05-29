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

from legged_obstacle_rl.tasks.mujoco.velocity_env import (
    GRAVITY_VEC,
    isaac_home_jpos,
    isaac_to_mujoco_joints,
    mujoco_to_isaac_joints,
    quat_apply_inverse,
)
from legged_obstacle_rl import teleop

sys.path.insert(0, os.path.expanduser("~/Projects/unitree_legged_sdk/lib/python/amd64"))
import robot_interface as sdk  # pyright: ignore[reportMissingImports]

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


def _fill_cmd(cmd: sdk.LowCmd, target_q_isaac: np.ndarray, kp: float, kd: float) -> None:
    """Write a PD position target to ``cmd``, converting IsaacLab -> SDK joint order."""
    q = target_q_isaac[isaac_to_mujoco_joints]
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A  # servo
        cmd.motorCmd[i].q = float(q[i])
        cmd.motorCmd[i].dq = 0.0
        cmd.motorCmd[i].Kp = kp
        cmd.motorCmd[i].Kd = kd
        cmd.motorCmd[i].tau = 0.0


def _fill_damping(cmd: sdk.LowCmd) -> None:
    """Put all motors in damping mode so the robot settles gently."""
    for i in range(12):
        cmd.motorCmd[i].mode = 0x00  # damping
        cmd.motorCmd[i].Kp = 0.0
        cmd.motorCmd[i].Kd = 0.0
        cmd.motorCmd[i].tau = 0.0
        cmd.motorCmd[i].q = 0.0
        cmd.motorCmd[i].dq = 0.0


class Go1Hardware:
    """Single owner of the Unitree SDK UDP socket.

    One thread runs ``_io_loop`` at ``COMM_LOOP_DT``: ``Recv`` -> publish a
    ``SensorSnapshot`` -> apply the latest ``MotorTarget`` -> ``PowerProtect`` ->
    ``Send``. Snap and target each live in a one-slot publisher under a
    dedicated lock.
    """

    def __init__(self):
        self._udp = sdk.UDP(LOWLEVEL, LOW_LOCAL_PORT, LOW_IP, LOW_ROBOT_PORT)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
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

    def _io_loop(self) -> None:
        try:
            print_t = time.perf_counter()
            for tick in pace_loop(COMM_LOOP_DT, self._stop):
                self._udp.Recv()
                self._udp.GetRecv(self._low_state)
                self.snap = self._read_snapshot()
                tgt = self.target
                if tgt is not None:
                    _fill_cmd(self._cmd, tgt.q_isaac, tgt.kp, tgt.kd)
                self._safe.PowerProtect(self._cmd, self._low_state, 1)
                self._udp.SetSend(self._cmd)
                self._udp.Send()

                now = time.perf_counter()
                if now - print_t > 1.0:
                    print(f"IO LOOP WORK: {now - tick:.4f}")
                    print_t = now
        finally:
            _fill_damping(self._cmd)
            self._udp.SetSend(self._cmd)
            self._udp.Send()
            print("Motors set to damping mode.")


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

    def __init__(self, hw: Go1Hardware, agent, teleop_) -> None:
        self.stop_event = threading.Event()

        self._hw = hw
        self._agent = agent
        self._teleop = teleop_
        self._est = EstimatorState()
        self._last_action = isaac_home_jpos.copy()
        self._thread: Optional[threading.Thread] = None

    def _wait_for_valid_snap(self, timeout_s: float = 5.0) -> SensorSnapshot:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.stop_event.is_set() or self._teleop.state.stop:
                raise RuntimeError("aborted while waiting for SDK state")
            snap = self._hw.snap
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
                obs = build_obs(snap, self._est, vel_cmd, self._last_action)
                action = _act(self._agent, obs)

                self._hw.target = MotorTarget(isaac_home_jpos + action * ACTION_SCALE, KP, KD)
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


def run(agent) -> None:
    """Start teleop, hardware, ramp, then run the policy until stop or error."""
    global _session

    hw = Go1Hardware()
    runner = PolicyRunner(hw, agent, teleop)
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
    "estimate_base_height",
    "pace_loop",
]
