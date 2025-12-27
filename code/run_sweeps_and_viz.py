#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, subprocess, glob
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

def run_one(cmd, cwd=None):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["results"], obj["meta"]["params"]

def mean_std(x):
    a = np.array(x, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

def viz_rank_sweep(records, method, out_path):
    # records: list of dict with keys rank, mean, std
    records.sort(key=lambda d: d["rank"])
    xs = [d["rank"] for d in records]
    ys = [d["mean"] for d in records]
    es = [d["std"]  for d in records]

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.6, capsize=4)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Test RMSE (mean ± std over folds)")
    ax.set_title(f"Rank sweep: {method}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def viz_heatmap(grid, xs, ys, title, out_path, xlabel, ylabel):
    # grid shape (len(ys), len(xs))
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_yticks(np.arange(len(ys)))
    ax.set_xticklabels([str(x) for x in xs], rotation=20, ha="right")
    ax.set_yticklabels([str(y) for y in ys])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Value")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", type=str, required=True)
    ap.add_argument("--python", type=str, default="python")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=42)

    # sweep configs
    ap.add_argument("--rank_list", type=int, nargs="+", default=[8,16,32,64,96,128])
    ap.add_argument("--lr_list", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--reg_list", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])

    ap.add_argument("--methods", type=str, nargs="+", default=["nc_mf_sgd", "nc_spec_alt", "nc_perturb"])
    ap.add_argument("--out_root", type=str, default=None)

    # keep CV runtime manageable for sweeps
    ap.add_argument("--epochs_mf", type=int, default=4)

    args = ap.parse_args()
    set_style()

    out_root = args.out_root or os.path.join(args.fold_dir, "sweeps")
    os.makedirs(out_root, exist_ok=True)

    # ---------- rank sweep (fix lr/reg) ----------
    rank_root = os.path.join(out_root, "rank_sweep")
    os.makedirs(rank_root, exist_ok=True)

    for r in args.rank_list:
        out_dir = os.path.join(rank_root, f"rank{r}")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            args.python, "code/run_cv_all_gpu.py",
            "--fold_dir", args.fold_dir,
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--epochs_mf", str(args.epochs_mf),
            "--rank_mf", str(r),
            "--methods", *args.methods,
        ]
        # run in-place; outputs results.json under fold_dir; so we copy it into out_dir for sweep bookkeeping
        run_one(cmd)

        # copy results.json & histories into out_dir snapshot
        src = os.path.join(args.fold_dir, "results.json")
        with open(src, "r", encoding="utf-8") as f:
            obj = json.load(f)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # aggregate rank sweep plots per method
    for m in args.methods:
        recs = []
        for p in glob.glob(os.path.join(rank_root, "rank*", "results.json")):
            rmses, params = load_results(p)
            if m not in rmses:
                continue
            rank = int(params["rank_mf"])
            mu, sd = mean_std(rmses[m])
            recs.append({"rank": rank, "mean": mu, "std": sd})
        if recs:
            viz_rank_sweep(recs, m, os.path.join(rank_root, f"rmse_vs_rank_{m}.png"))

    # ---------- lr×reg grid for one nonconvex method (nc_mf_sgd) ----------
    grid_root = os.path.join(out_root, "param_grid_nc_mf_sgd")
    os.makedirs(grid_root, exist_ok=True)

    for lr in args.lr_list:
        for reg in args.reg_list:
            out_dir = os.path.join(grid_root, f"lr{lr}_reg{reg}")
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                args.python, "code/run_cv_all_gpu.py",
                "--fold_dir", args.fold_dir,
                "--device", args.device,
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed),
                "--epochs_mf", str(args.epochs_mf),
                "--lr_mf", str(lr),
                "--reg_mf", str(reg),
                "--methods", "nc_mf_sgd",
            ]
            run_one(cmd)

            src = os.path.join(args.fold_dir, "results.json")
            with open(src, "r", encoding="utf-8") as f:
                obj = json.load(f)
            with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

    # build heatmaps: mean RMSE and std RMSE
    xs = args.lr_list
    ys = args.reg_list
    mean_grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    std_grid  = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)

    for yi, reg in enumerate(ys):
        for xi, lr in enumerate(xs):
            p = os.path.join(grid_root, f"lr{lr}_reg{reg}", "results.json")
            rmses, _ = load_results(p)
            mu, sd = mean_std(rmses["nc_mf_sgd"])
            mean_grid[yi, xi] = mu
            std_grid[yi, xi]  = sd

    viz_heatmap(mean_grid, xs, ys,
                "nc_mf_sgd mean test RMSE (fold mean)",
                os.path.join(grid_root, "heatmap_mean_rmse.png"),
                xlabel="lr", ylabel="reg")

    viz_heatmap(std_grid, xs, ys,
                "nc_mf_sgd std of test RMSE (fold std)  — stability",
                os.path.join(grid_root, "heatmap_std_rmse.png"),
                xlabel="lr", ylabel="reg")

    print("\nSaved sweep plots under:", out_root)

if __name__ == "__main__":
    main()
