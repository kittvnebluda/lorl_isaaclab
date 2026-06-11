"""Compare two observation logs (real vs sim) recorded by ``deployment.obs_log.StepLogger``.

Both ``.npz`` files hold ``obs`` (N, 45) + ``action`` (N, 12) + ``t`` (N,). This aligns
them by step index and reports per-group / per-joint divergence so you can pin which
observation term drives a sim2real gap.

Usage:
    python scripts/compare_obs.py logs/obs/real.npz logs/obs/sim.npz [--skip 50] [--plot]

``--skip`` drops the first N steps of each run (startup/settle transient) before comparing.
``--plot`` draws per-group time series (needs matplotlib; otherwise text-only).
"""

from __future__ import annotations

import argparse

import numpy as np

# 45-dim direction proprio layout — must match build_obs_proprio_dir / StepLogger.
SLICES = {
    "joint_pos": slice(0, 12),
    "base_ang_vel": slice(12, 15),
    "joint_vel": slice(15, 27),
    "projected_gravity": slice(27, 30),
    "direction_command": slice(30, 33),
    "last_action": slice(33, 45),
}
JOINTS = ["FLh", "FRh", "RLh", "RRh", "FLt", "FRt", "RLt", "RRt", "FLc", "FRc", "RLc", "RRc"]


def _labels(group: str, n: int) -> list[str]:
    if n == 12:
        return JOINTS
    if group == "base_ang_vel":
        return ["wx", "wy", "wz"]
    if group == "projected_gravity":
        return ["gx", "gy", "gz"]
    if group == "direction_command":
        return ["cos", "sin", "turn"]
    return [str(i) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real", help="first .npz (e.g. real robot)")
    ap.add_argument("sim", help="second .npz (e.g. IsaacSim)")
    ap.add_argument("--skip", type=int, default=0, help="drop first N steps of each run before comparing")
    ap.add_argument("--plot", action="store_true", help="plot per-group time series (needs matplotlib)")
    args = ap.parse_args()

    a, b = np.load(args.real), np.load(args.sim)
    oa, ob = a["obs"][args.skip :], b["obs"][args.skip :]
    n = min(len(oa), len(ob))
    if n == 0:
        raise SystemExit("no overlapping steps after --skip")
    oa, ob = oa[:n], ob[:n]
    print(f"real={args.real} ({len(a['obs'])} rows)  sim={args.sim} ({len(b['obs'])} rows)")
    print(f"comparing {n} aligned steps (skip={args.skip})\n")

    diff = oa - ob  # (n, 45)
    print(f"{'group':18s} {'real_mean':>10s} {'sim_mean':>10s} {'|diff|_mean':>12s} {'|diff|_max':>11s}")
    print("-" * 66)
    worst_g, worst_v = None, -1.0
    for g, s in SLICES.items():
        d = np.abs(diff[:, s])
        gm = d.mean()
        print(
            f"{g:18s} {np.linalg.norm(oa[:, s].mean(0)):>10.3f} {np.linalg.norm(ob[:, s].mean(0)):>10.3f} "
            f"{gm:>12.4f} {d.max():>11.4f}"
        )
        if gm > worst_v:
            worst_g, worst_v = g, gm
    print(f"\nlargest divergence: {worst_g} (|diff|_mean={worst_v:.4f})\n")

    # per-dim breakdown of the worst group
    s = SLICES[worst_g]
    labels = _labels(worst_g, s.stop - s.start)
    dd = np.abs(diff[:, s]).mean(0)
    print(f"per-dim |diff|_mean for '{worst_g}':")
    for lab, v, ra, sb in zip(labels, dd, oa[:, s].mean(0), ob[:, s].mean(0)):
        print(f"  {lab:5s} diff={v:8.4f}   real={ra:+8.4f}  sim={sb:+8.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[plot] matplotlib not installed — skipping")
            return
        groups = list(SLICES)
        fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2 * len(groups)), sharex=True)
        for ax, g in zip(axes, groups):
            sl = SLICES[g]
            ax.plot(np.linalg.norm(oa[:, sl], axis=1), label="real")
            ax.plot(np.linalg.norm(ob[:, sl], axis=1), label="sim", ls="--")
            ax.set_ylabel(g, fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
