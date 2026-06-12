import os

import yaml


def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
        return data


def plot_torques(torque_log, joint_names, dt, log_dir):
    """Plot per-joint motor torques over the episode and save to log_dir."""
    if not torque_log:
        print("[WARN] No torque data collected, skipping plot.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    torques = np.asarray(torque_log)  # (steps, num_joints)
    t = np.arange(torques.shape[0]) * dt

    fig, ax = plt.subplots(figsize=(12, 7))
    for j, name in enumerate(joint_names):
        ax.plot(t, torques[:, j], label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("applied torque [Nm]")
    ax.set_title("Motor torques")
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    ax.grid(True)

    out_path = os.path.join(log_dir, "torques.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved torque plot to: {out_path}")
    plt.show()
