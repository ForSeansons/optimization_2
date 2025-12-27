#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
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


def mean_std(x):
    a = np.array(x, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)


def is_convex(name: str) -> bool:
    return name.startswith("c_")


def plot_bar_sorted(methods, rmses, out_path):
    stats = []
    for m in methods:
        mu, sd = mean_std(rmses[m])
        stats.append((m, mu, sd, is_convex(m)))
    stats.sort(key=lambda t: t[1])  # sort by mean rmse

    labels = [s[0] for s in stats]
    means = [s[1] for s in stats]
    stds  = [s[2] for s in stats]
    flags = [s[3] for s in stats]

    x = np.arange(len(labels))
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)

    bars = ax.bar(x, means, yerr=stds, capsize=4)
    for b, cflag in zip(bars, flags):
        if cflag:
            b.set_hatch("//")  # convex
        else:
            b.set_hatch("")    # nonconvex

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test RMSE (mean ± std, 5 folds)")
    ax.set_title("All methods (sorted): hatch='//' indicates convex")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_box_grouped(methods, rmses, out_path):
    nc = [m for m in methods if not is_convex(m)]
    cvx = [m for m in methods if is_convex(m)]

    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)

    data = [rmses[m] for m in nc] + [rmses[m] for m in cvx]
    labels = nc + cvx

    bp = ax.boxplot(data, labels=labels, showmeans=True)
    # visually mark convex group background
    split = len(nc)
    ax.axvline(split + 0.5, linestyle="--", alpha=0.35)

    ax.set_ylabel("Test RMSE")
    ax.set_title("RMSE distribution across folds (left=nonconvex, right=convex)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(methods, rmses, out_path):
    k = len(next(iter(rmses.values())))
    mat = np.zeros((k, len(methods)), dtype=np.float64)
    for j, m in enumerate(methods):
        mat[:, j] = np.array(rmses[m], dtype=np.float64)

    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels([f"fold{i}" for i in range(k)])
    ax.set_title("Per-fold test RMSE heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("RMSE")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rankline(methods, rmses, out_path):
    k = len(next(iter(rmses.values())))
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)

    for fi in range(k):
        pairs = [(m, rmses[m][fi]) for m in methods]
        pairs.sort(key=lambda x: x[1])
        ys = [p[1] for p in pairs]
        ax.plot(np.arange(len(methods)), ys, marker="o", linewidth=1.4, label=f"fold{fi}")

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([str(i + 1) for i in range(len(methods))])
    ax.set_xlabel("Within-fold rank (1=best)")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Stability view: RMSE vs within-fold rank")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--method_order", type=str, nargs="*", default=None)
    args = ap.parse_args()

    set_style()
    with open(args.results_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rmses = obj["results"]

    methods = args.method_order if args.method_order else list(rmses.keys())
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_json))
    os.makedirs(out_dir, exist_ok=True)

    plot_bar_sorted(methods, rmses, os.path.join(out_dir, "rmse_bar_sorted_grouped.png"))
    plot_box_grouped(methods, rmses, os.path.join(out_dir, "rmse_box_grouped.png"))
    plot_heatmap(methods, rmses, os.path.join(out_dir, "rmse_heatmap.png"))
    plot_rankline(methods, rmses, os.path.join(out_dir, "rmse_rankline.png"))

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()
