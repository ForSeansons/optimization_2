#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_histories(fold_dir):
    """Load val_rmse histories grouped by method."""
    paths = sorted(glob.glob(os.path.join(fold_dir, "history_fold*_*.json")))
    if not paths:
        raise FileNotFoundError("No history_fold*_*.json found. Run CV first.")

    pat = re.compile(r"history_fold(\d+)_(.+)\.json$")
    by_method = {}
    for p in paths:
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        fold = int(m.group(1))
        method = m.group(2)
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        curve = obj.get("val_rmse", [])
        by_method.setdefault(method, []).append((fold, curve))
    return by_method


def plot_group(by_method, title, out_path):
    """Plot mean ± std convergence curves for a group of methods."""
    fig = plt.figure(figsize=(10.5, 4.4))
    ax = fig.add_subplot(111)

    for method, curves in sorted(by_method.items()):
        max_len = max(len(c) for _, c in curves)
        mat = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for r, (_, c) in enumerate(curves):
            mat[r, :len(c)] = np.array(c, dtype=np.float64)

        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(1, max_len + 1)

        ax.plot(x, mean, marker="o", linewidth=1.5, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12)

    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Validation RMSE (mean ± std across folds)")
    ax.set_title(title)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    set_style()
    by_method = load_histories(args.fold_dir)

    high_key = "c_fista_nuc"
    excluded = {
        "alt_block_sgd",
        "mf_sgd",
        "perturbed_sgd",
        "spec_init_alt",
        "nc_mf_sgd",
        "nc_perturb",
        "nc_spec_alt",
    }

    c_fista_data = {high_key: by_method.pop(high_key)} if high_key in by_method else {}

    # Drop explicitly excluded methods from the shared plot
    for k in list(by_method.keys()):
        if k in excluded:
            by_method.pop(k)

    other_data = dict(by_method)

    if other_data:
        out_path = args.out or os.path.join(args.fold_dir, "convergence_val_rmse.png")
        plot_group(other_data, "Convergence comparison (validation)", out_path)
    else:
        print("Warning: no methods to plot (excluding c_fista_nuc).")

    if c_fista_data:
        out_path_fista = (
            os.path.join(args.fold_dir, "convergence_val_rmse_c_fista_nuc.png")
            if args.out is None
            else os.path.splitext(args.out)[0] + "_c_fista_nuc.png"
        )
        plot_group(c_fista_data, "Convergence (validation) - c_fista_nuc", out_path_fista)


if __name__ == "__main__":
    main()
