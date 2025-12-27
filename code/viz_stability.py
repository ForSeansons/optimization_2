#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def mean_std(x):
    a = np.array(x, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    set_style()

    with open(args.results_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rmses = obj["results"]  # dict: method -> [fold0..fold4]

    methods = list(rmses.keys())
    k = len(next(iter(rmses.values())))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_json))
    os.makedirs(out_dir, exist_ok=True)

    # ---------- (a) RMSE mean±std ----------
    stats = []
    for m in methods:
        mu, sd = mean_std(rmses[m])
        stats.append((m, mu, sd))
    stats.sort(key=lambda t: t[1])  # sort by mean rmse

    labels = [t[0] for t in stats]
    means  = [t[1] for t in stats]
    stds   = [t[2] for t in stats]

    x = np.arange(len(labels))
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test RMSE (mean ± std over folds)")
    ax.set_title("Stability view A: RMSE mean±std (fold)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stability_bar_rmse.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------- (b) rank stability ----------
    # For each fold: rank methods by rmse (1=best). Then compute mean/std rank per method
    rank_mat = np.zeros((k, len(methods)), dtype=np.float64)
    for fi in range(k):
        pairs = [(m, rmses[m][fi]) for m in methods]
        pairs.sort(key=lambda t: t[1])
        rank = {m: (idx + 1) for idx, (m, _) in enumerate(pairs)}
        for j, m in enumerate(methods):
            rank_mat[fi, j] = rank[m]

    rank_mean = rank_mat.mean(axis=0)
    rank_std  = rank_mat.std(axis=0, ddof=1) if k > 1 else np.zeros_like(rank_mean)

    order = np.argsort(rank_mean)  # best mean rank first
    labels_r = [methods[i] for i in order]
    mean_r   = rank_mean[order]
    std_r    = rank_std[order]

    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(labels_r)), mean_r, yerr=std_r, capsize=4)
    ax.set_xticks(np.arange(len(labels_r)))
    ax.set_xticklabels(labels_r, rotation=25, ha="right")
    ax.set_ylabel("Fold-wise rank (mean ± std), 1=best")
    ax.set_title("Stability view B: ranking stability across folds")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stability_rank.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------- (c) win-rate matrix ----------
    # winrate[a,b] = fraction of folds where a rmse < b rmse
    win = np.zeros((len(methods), len(methods)), dtype=np.float64)
    for a, ma in enumerate(methods):
        for b, mb in enumerate(methods):
            if a == b:
                win[a, b] = np.nan
            else:
                cnt = 0
                for fi in range(k):
                    if rmses[ma][fi] < rmses[mb][fi]:
                        cnt += 1
                win[a, b] = cnt / k

    fig = plt.figure(figsize=(8.2, 6.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(win, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticklabels(methods)
    ax.set_title("Stability view C: pairwise win-rate (fold fraction)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Win-rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stability_winrate.png"), bbox_inches="tight")
    plt.close(fig)

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
