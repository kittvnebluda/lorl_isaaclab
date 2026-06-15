# Unitree Go1 — Rough Terrain RL in IsaacLab

## Overview

Reinforcement learning for the Unitree Go1 quadruped on rough terrain, using [IsaacLab](https://isaac-sim.github.io/IsaacLab) for training and [skrl](https://skrl.readthedocs.io) as the RL library. Includes a MuJoCo sim-to-sim transfer pipeline and keyboard teleop for interactive evaluation.

**Features:**

- Velocity-commanded and direction-commanded locomotion policies
- Curriculum over 8 procedurally generated terrain types
- MuJoCo sim2sim deployment with learned actuator model from [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)
- Keyboard teleop in both IsaacLab (play) and MuJoCo (deploy_mujoco)
- TensorBoard logging

**Keywords:** unitree, go1, aliengo, isaaclab, rsl_rl, skrl, legged-robotics, sim2sim, mujoco

## Installation

Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Clone this repository outside the `IsaacLab` directory, then install in editable mode:

```bash
python -m pip install -e source/legged_obstacle_rl
```

Verify by listing available environments:

```bash
python scripts/list_envs.py
```

### Cache Robot Assets Locally (optional, avoids Nucleus timeouts)

Robot USDs default to NVIDIA's remote Nucleus server, so every launch streams
them over the network. Download them once to a local cache to start offline
and avoid server timeouts:

```bash
python scripts/download_assets.py            # downloads Aliengo + Go1 to ~/.cache/legged_obstacle_rl/assets
# or a custom location:
python scripts/download_assets.py --dest /data/lorl_assets
export LORL_ASSETS_DIR=/data/lorl_assets     # if you used --dest, point training at it
```

After this, `ALIENGO_CFG` / `GO1_CFG` resolve to the local copies automatically.
If an asset is missing locally, they fall back to the Nucleus
URL --- nothing breaks before the download. Re-run with `--force` to refresh.

## Scripts

### Train (skrl)

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

### Teacher–Student Training (rsl_rl)

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

### Play  (skrl)

```bash
python scripts/skrl/play.py \
    --task=LORL-Go1Rough-RL-Play-v0 \
    --checkpoint PATH \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

`--teleop` enables keyboard control (see [Teleop Controls](#teleop-controls)).

### Play (rsl_rl)

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

`--teleop` enables keyboard control (see [Teleop Controls](#teleop-controls)).

### Deploy to MuJoCo

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

## Teleop Controls

Available in both `play.py` (`--teleop`) and `deploy_mujoco.py` (`--teleop`).
Requires Linux evdev python package.

| Key | Action | Range |
|-----|--------|-------|
| I / K | Forward velocity +/- | [-1.5, 1.5] m/s |
| J / L | Lateral velocity +/- | [-1.5, 1.5] m/s |
| U / O | Yaw rate +/- | [-1.5, 1.5] rad/s |
| Y / H | Body height +/- | [0.1, 0.5] m |
| Ctrl+L | Toggle command lock | — |
| ESC | Stop | — |

## MuJoCo Setup

```bash
pip install gymnasium[mujoco]
```

## ROS2 / Gazebo

> **Status: work in progress — ROS2 deployment not yet functional.**

A ROS2 workspace is located at `source/legged_obstacle_rl/legged_obstacle_rl/tasks/ros2_ws/`
with packages for bringup, description, and Gazebo simulation.

Install Gazebo Harmonic:

```bash
sudo apt-get update
sudo apt-get install lsb-release wget gnupg

sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update
sudo apt-get install gz-harmonic ros-humble-ros-gzharmonic

echo 'export GZ_VERSION=harmonic' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
gz sim
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

Add the extension path to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/legged_obstacle_rl"
    ]
}
```

### Pylance Crash

If Pylance runs out of memory from indexing too many Omniverse packages, exclude unused ones in `.vscode/settings.json`:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"
"<path-to-isaac-sim>/extscache/omni.kit.*"
"<path-to-isaac-sim>/extscache/omni.graph.*"
"<path-to-isaac-sim>/extscache/omni.services.*"
```
