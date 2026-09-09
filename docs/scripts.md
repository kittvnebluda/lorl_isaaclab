# Scripts

Command-line entry points for training, evaluation and deployment.
Back to the [README](../README.md).

## Train (skrl)

```bash
python scripts/skrl/train.py \
    --task=LORL-Go1Rough-RL-v0 \
    [--num_envs 4096] \
    [--checkpoint PATH] \
    [--max_iterations 1500] \
    [--video] [--video_length 200] \
    [--seed 42] \
    [--algorithm PPO]
```

Logs to `logs/skrl/` and `outputs/`. TensorBoard metrics include velocity
tracking, terrain level, and custom hyperparameters.

## Teacher–Student Training (rsl_rl)

Two-phase privileged-learning + distillation for the AlienGo direction task.
Phase A trains a privileged teacher with PPO; Phase B distills it into
a proprioception-only GRU student via DAgger. Both phases log under `logs/rsl_rl/aliengo_direction/`.

**Phase A — teacher (privileged PPO, symmetry augmentation, clean observations):**

```bash
python scripts/rsl_rl/train.py \
    --task LORL-AlienGoDirection-RL-v0 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --run_name teacher \
    --headless
```

Resume a teacher:

```bash
  python scripts/rsl_rl/train.py \
      --task LORL-AlienGoDirection-RL-v0 \
      --num_envs 8192 \
      --max_iterations 1500 \
      --run_name teacher \
      --headless \
      --resume \
      --load_run 2026-06-16_14-25-03_teacher_kadupul \
      --checkpoint model_1499.pt \
      --seed -1
```

Writes checkpoints to `logs/rsl_rl/aliengo_direction/<timestamp>_teacher/`.

**Phase B — student (GRU DAgger distillation, noisy proprioception):**

```bash
python scripts/rsl_rl/train.py \
    --task LORL-AlienGoDirection-RL-Distill-v0 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --num_envs 4096 \
    --max_iterations 1000 \
    --load_run <timestamp>_teacher \
    --checkpoint model_1499.pt \
    --run_name student \
    --headless
```

`--agent rsl_rl_distillation_cfg_entry_point` selects the distillation runner; `--load_run`
and `--checkpoint` point at the Phase A teacher (resolved within the shared
`aliengo_direction` experiment root). The student imitates the teacher's actions
while acting on corrupted proprioception.

## Play  (skrl)

```bash
python scripts/skrl/play.py \
    --task=LORL-Go1Rough-RL-Play-v0 \
    --checkpoint PATH \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

`--teleop` enables keyboard control (see [Teleop Controls](../README.md#teleop-controls)).

## Play (rsl_rl)

Teacher:

```bash
python scripts/rsl_rl/play.py
    --task LORL-AlienGoDirection-RL-Play-v0 \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_teacher/model_X.pt \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

Student:

```bash
python scripts/rsl_rl/play.py
    --task LORL-AlienGoDirection-RL-Play-v0 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_student/model_X.pt \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

`--teleop` enables keyboard control (see [Teleop Controls](../README.md#teleop-controls)).

## Deploy to MuJoCo

skrl:

```bash
python scripts/skrl/deploy_mujoco.py \
    --task=LORL-Go1Rough-MJ-v0 \
    --checkpoint PATH \
    --teleop \
    --real-time \
    [--config path/to/agent_cfg.yaml]
```

rsl_rl:

```bash
python scripts/rsl_rl/deploy_mujoco.py \
    --task LORL-Aliengo-Direction-MJ-v0 \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_student/exported/policy.pt \
    --real-time \
    --teleop
```

## Deploy to Real Robot (rsl_rl)

Runs a TorchScript policy directly on Unitree hardware over the low-level UDP
interface. No IsaacLab / Isaac Sim needed — only `torch`, `numpy`, `evdev` and the
Unitree SDK python bindings.

### Prerequisites

- `policy.pt` exported by `scripts/rsl_rl/play.py`, which writes
  `<checkpoint-dir>/exported/policy.pt` (TorchScript) and `policy.onnx`.
- [`unitree_legged_sdk`](https://github.com/unitreerobotics/unitree_legged_sdk) built
  with python bindings. `deployment._common.load_sdk` prepends
  `~/Projects/unitree_legged_sdk/lib/python/amd64` to `sys.path` and imports
  `robot_interface` — build the branch matching the robot (go1 branch for Go1,
  aliengo/v3.2 for AlienGo) and the interpreter's cpython ABI.
- Host on the robot's low-level network, able to reach `192.168.123.10`
  (ports: Go1 local 8080, AlienGo local 8082, robot 8007).
- Robot on the ground with the high-level sport controller put into low-level
  mode from the remote, so this script owns the motors.
- evdev keyboard access for teleop (same controls as
  [Teleop Controls](../README.md#teleop-controls)).

### Run

```bash
python scripts/rsl_rl/deploy_real.py \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_student/exported/policy.pt \
    --hardware aliengo        # or: go1
```

`--hardware` selects the module under `legged_obstacle_rl/deployment/`
(`aliengo.py` / `go1.py`), which supplies the SDK spec, PD gains
(AlienGo `kp=46, kd=2`; Go1 `kp=35, kd=1.5`) and joint limits.

`aliengo` is the rsl_rl path: it defaults to the 45-dim proprioception-only
observation (`build_obs_proprio_dir`) and to `act_rsl_rl`, which calls the
TorchScript module directly — matching the distilled direction student.

`go1` still defaults to the shared 235-dim `build_obs` (proprioception + a flat
height scan derived from leg-FK base height) *and* to the skrl action path
(`_common._act_skrl`, which calls `agent.act(...)`).

### Startup sequence

1. Teleop thread starts (keyboard → velocity command + stop flag).
2. Hardware I/O thread starts at 500 Hz — sole owner of the UDP socket, reads
   `LowState`, applies `PowerProtect`, sends `LowCmd`.
3. Ramp: interpolates from the measured joint positions to `isaac_home_jpos`.
4. Policy thread runs at 50 Hz.

Press Ctrl-C to stop: the runner switches to damping
(`kd=8`) for 4 s so the robot settles, then cuts the motors. Shutdown is also
registered via `atexit`.

### Safe bringup

Before the first live run, uncomment the zeroing line in `act_rsl_rl`
(`deployment/aliengo.py`) so actions are dropped — the robot ramps to home and
holds limp, which verifies comms, joint ordering and IMU signs without the policy
driving anything.

### Logging for sim2real debugging

```bash
LORL_OBS_LOG=logs/obs/real.npz python scripts/rsl_rl/deploy_real.py \
    --checkpoint .../exported/policy.pt --hardware aliengo
```

`log_step` inside `act_rsl_rl` writes an `.npz` of `obs`/`action`/`t`. Record the
matching sim run with a `StepLogger` in play, then diff per observation group:

```bash
python scripts/compare_obs.py logs/obs/real.npz logs/obs/sim.npz [--skip 50] [--plot]
```
