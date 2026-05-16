# Unitree Go1 — Rough Terrain RL in IsaacLab

## Overview

Reinforcement learning for the Unitree Go1 quadruped on rough terrain, using [IsaacLab](https://isaac-sim.github.io/IsaacLab) for training and [skrl](https://skrl.readthedocs.io) as the RL library. Includes a MuJoCo sim-to-sim transfer pipeline and keyboard teleop for interactive evaluation.

**Features:**

- Velocity-commanded and direction-commanded locomotion policies
- Curriculum over 8 procedurally generated terrain types
- MuJoCo sim2sim deployment with learned actuator model (MLP)
- Keyboard teleop in both IsaacLab (play) and MuJoCo (deploy)
- TensorBoard logging of custom scalars and hyperparameters

**Keywords:** unitree, go1, isaaclab, skrl, legged-robotics, sim2sim, mujoco

## Installation

Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Clone this repository outside the `IsaacLab` directory, then install in editable mode:

```bash
python -m pip install -e source/legged_obstacle_rl
```

Verify by listing available environments:

```bash
python scripts/skrl/list_envs.py
```

## Environments

### IsaacLab — Velocity Commanded

| Environment ID | Description |
|---|---|
| `LORL-Go1Rough-RL-v0` | Training: velocity commands, rough terrain curriculum |
| `LORL-Go1Rough-RL-Play-v0` | Evaluation: reduced envs, no randomization |
| `LORL-Go1Rough-RL-Play-ICRA-v0` | Evaluation: single env on custom ICRA map |
| `LORL-Go1Rough-RL-LongArch-v0` | Training: baseline env, larger network architecture |
| `LORL-Go1Rough-RL-LongArch-Play-v0` | Evaluation: LongArch |
| `LORL-Go1Rough-RL-LongArch-Play-ICRA-v0` | Evaluation: LongArch, ICRA map |
| `LORL-Go1Rough-RL-WideArch-v0` | Training: baseline env, wider network architecture |
| `LORL-Go1Rough-RL-WideArch-Play-v0` | Evaluation: WideArch |
| `LORL-Go1Rough-RL-WideArch-Play-ICRA-v0` | Evaluation: WideArch, ICRA map |
| `LORL-Go1RoughLongHistory-RL-v0` | Training: 15-step observation history |
| `LORL-Go1RoughLongHistory-RL-Play-v0` | Evaluation: 15-step history |
| `LORL-Go1RoughLongHistory-RL-Play-ICRA-v0` | Evaluation: 15-step history, ICRA map |
| `LORL-Go1Argo-RL-v0` | Training: Argo env variant |
| `LORL-Go1Argo-RL-Play-v0` | Evaluation: Argo variant |
| `LORL-Go1Argo-RL-H15-v0` | Training: Argo with 15-step history |
| `LORL-Go1Argo-RL-H15-Play-v0` | Evaluation: Argo with 15-step history |

### IsaacLab — Direction Commanded

| Environment ID | Description |
|---|---|
| `LORL-Go1Direction-RL-v0` | Training: direction vector command |
| `LORL-Go1Direction-RL-Play-v0` | Evaluation |
| `LORL-Go1Direction-RL-Play-ICRA-v0` | Evaluation on ICRA map |

Robot receives a direction unit vector and learns to move toward it. Rewards orientation and velocity alignment; penalizes orthogonal drift.

### MuJoCo — Sim2Sim

| Environment ID | Description |
|---|---|
| `LORL-Go1Rough-MJC-v0` | Go1 rough terrain in MuJoCo |
| `LORL-Go1Argo-MJC-v0` | Argo env in MuJoCo |
| `LORL-Go1ArgoH-MJC-v0` | Argo env with longer history in MuJoCo |

Uses a learned MLP actuator model (`ActuatorNetMLP`) trained to replicate Go1 joint dynamics. Observation space: 235-dim (joint positions, velocities, base state, projected gravity, velocity commands, height scan).

## Terrain

8 procedurally generated sub-terrain types with curriculum difficulty scaling:

| Type | Description |
|---|---|
| Pyramid stairs (ascending) | Steps 0.05–0.23 m high, 0.3 m wide |
| Pyramid stairs (descending) | Inverted pyramid |
| Random boxes | Grid of boxes 0.05–0.2 m high |
| Random rough | Perlin noise 0.01–0.06 m amplitude |
| Sloped pyramid (ascending) | Slope 0–0.4 |
| Sloped pyramid (descending) | Inverted slope |
| Square holes | Flat terrain with 0.3 m square holes |
| Diamond walkway | Beam grid mesh |

## Scripts

### Train (IsaacLab)

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

Logs to `logs/skrl/` and `outputs/`. TensorBoard metrics include velocity tracking, terrain level, and custom hyperparameters.

### Play / Evaluate (IsaacLab)

```bash
python scripts/skrl/play.py \
    --task=LORL-Go1Rough-RL-Play-v0 \
    --checkpoint PATH \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

`--teleop` enables keyboard control (see [Teleop Controls](#teleop-controls)).

### Deploy to MuJoCo (Sim2Sim)

```bash
python scripts/skrl/deploy.py \
    --task=LORL-Go1Rough-MJC-v0 \
    --checkpoint PATH \
    [--config path/to/agent_cfg.yaml] \
    [--teleop] \
    [--real-time]
```

Loads a checkpoint trained in IsaacLab and runs it in MuJoCo. Records action history and plots at end of episode.

### Export Weights

Saves policy and preprocessor weights separately for deployment:

```bash
python scripts/skrl/deploy.py --task=... --checkpoint PATH
```

## Teleop Controls

Available in both `play.py` (`--teleop`) and `deploy.py` (`--teleop`). Requires Linux evdev.

| Key | Action | Range |
|-----|--------|-------|
| I / K | Forward velocity +/- | [-1.5, 1.5] m/s |
| J / L | Lateral velocity +/- | [-1.5, 1.5] m/s |
| U / O | Yaw rate +/- | [-1.5, 1.5] rad/s |
| Y / H | Body height +/- | [0.1, 0.5] m |
| Ctrl+L | Toggle command lock | — |
| ESC | Stop | — |

MuJoCo deploy additionally uses WASD/QE/RF keys.

## MuJoCo Setup

```bash
pip install gymnasium[mujoco] skrl
```

## ROS2 / Gazebo

> **Status: work in progress — ROS2 deployment not yet functional.**

A ROS2 workspace is located at `source/legged_obstacle_rl/legged_obstacle_rl/tasks/sim2sim/ros2_ws/` with packages for bringup, description, and Gazebo simulation (in development).

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
