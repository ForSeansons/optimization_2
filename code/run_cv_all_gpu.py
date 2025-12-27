#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from all_methods_gpu import (
    set_seed,
    rmse_predictor,
    train_nc_mf_sgd,
    train_nc_mf_nobias,
    train_nc_alt_block,
    train_nc_spec_alt,
    train_nc_perturb,
    train_nc_pgd_rankk,
    train_c_softimpute,
    train_c_ista_nuc,
    train_c_fista_nuc,
    train_c_fw_trace,
)


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize(vals: List[float]) -> dict:
    a = np.array(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()) if a.size else float("nan"),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "folds": int(a.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_chunk", type=int, default=1_000_000)

    # nonconvex MF
    ap.add_argument("--rank_mf", type=int, default=64)
    ap.add_argument("--epochs_mf", type=int, default=30)
    ap.add_argument("--lr_mf", type=float, default=0.02)
    ap.add_argument("--reg_mf", type=float, default=0.02)
    ap.add_argument("--reg_bias", type=float, default=0.005)

    # rank-k PGD
    ap.add_argument("--rank_pgd", type=int, default=32)
    ap.add_argument("--iters_pgd", type=int, default=30)
    ap.add_argument("--eta_pgd", type=float, default=0.2)

    # convex nuclear/softimpute
    ap.add_argument("--rank_cvx", type=int, default=32)
    ap.add_argument("--iters_softimpute", type=int, default=30)
    ap.add_argument("--lam_softimpute", type=float, default=0.5)

    ap.add_argument("--iters_ista", type=int, default=30)
    ap.add_argument("--lam_ista", type=float, default=0.5)
    ap.add_argument("--eta_ista", type=float, default=0.1)

    ap.add_argument("--iters_fista", type=int, default=30)
    ap.add_argument("--lam_fista", type=float, default=0.5)
    ap.add_argument("--eta_fista", type=float, default=0.1)

    # convex FW trace-norm
    ap.add_argument("--iters_fw", type=int, default=30)
    ap.add_argument("--tau_fw", type=float, default=50.0)

    ap.add_argument("--methods", type=str, nargs="+", default=[
        "nc_mf_sgd",
        "nc_mf_nobias",
        "nc_alt_block",
        "nc_spec_alt",
        "nc_perturb",
        "nc_pgd_rankk",
        "c_softimpute",
        "c_ista_nuc",
        "c_fista_nuc",
        "c_fw_trace",
    ])
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA not available. Use --device cpu or install GPU torch."
    set_seed(args.seed)

    meta_path = os.path.join(args.fold_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n_users, n_items = int(meta["n_users"]), int(meta["n_items"])

    folds = [os.path.join(args.fold_dir, f"fold{k}.txt") for k in range(5)]
    for p in folds:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    results: Dict[str, List[float]] = {m: [] for m in args.methods}

    for k in range(5):
        test_paths = [folds[k]]
        val_paths = [folds[(k + 1) % 5]]
        train_paths = [folds[j] for j in range(5) if j not in (k, (k + 1) % 5)]

        print(f"\n================= FOLD {k} =================")
        print(f"train={train_paths}\nval={val_paths[0]}\ntest={test_paths[0]}")
        print(f"users={n_users}, items={n_items}, device={args.device}")

        # --- Nonconvex ---
        if "nc_mf_sgd" in args.methods:
            model, hist = train_nc_mf_sgd(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_mf, epochs=args.epochs_mf,
                batch_size=args.batch_size, lr=args.lr_mf,
                reg=args.reg_mf, reg_bias=args.reg_bias,
                device=args.device, seed=args.seed + 100*k,
                shuffle_chunk=args.shuffle_chunk,
            )
            rmse = rmse_predictor(model, test_paths, args.batch_size, args.device)
            results["nc_mf_sgd"].append(rmse)
            print(f"[nc_mf_sgd] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_nc_mf_sgd.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "nc_mf_nobias" in args.methods:
            model, hist = train_nc_mf_nobias(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_mf, epochs=args.epochs_mf,
                batch_size=args.batch_size, lr=args.lr_mf,
                reg=args.reg_mf,
                device=args.device, seed=args.seed + 100*k + 1,
                shuffle_chunk=args.shuffle_chunk,
            )
            rmse = rmse_predictor(model, test_paths, args.batch_size, args.device)
            results["nc_mf_nobias"].append(rmse)
            print(f"[nc_mf_nobias] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_nc_mf_nobias.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "nc_alt_block" in args.methods:
            model, hist = train_nc_alt_block(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_mf, epochs=args.epochs_mf,
                batch_size=args.batch_size, lr=args.lr_mf,
                reg=args.reg_mf, reg_bias=args.reg_bias,
                device=args.device, seed=args.seed + 100*k + 2,
                shuffle_chunk=args.shuffle_chunk,
            )
            rmse = rmse_predictor(model, test_paths, args.batch_size, args.device)
            results["nc_alt_block"].append(rmse)
            print(f"[nc_alt_block] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_nc_alt_block.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "nc_spec_alt" in args.methods:
            model, hist = train_nc_spec_alt(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_mf, epochs=args.epochs_mf,
                batch_size=args.batch_size, lr=args.lr_mf,
                reg=args.reg_mf, reg_bias=args.reg_bias,
                device=args.device, seed=args.seed + 100*k + 3,
                shuffle_chunk=args.shuffle_chunk,
            )
            rmse = rmse_predictor(model, test_paths, args.batch_size, args.device)
            results["nc_spec_alt"].append(rmse)
            print(f"[nc_spec_alt] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_nc_spec_alt.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "nc_perturb" in args.methods:
            model, hist = train_nc_perturb(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_mf, epochs=args.epochs_mf,
                batch_size=args.batch_size, lr=args.lr_mf,
                reg=args.reg_mf, reg_bias=args.reg_bias,
                device=args.device, seed=args.seed + 100*k + 4,
                shuffle_chunk=args.shuffle_chunk,
                plateau_patience=2, noise_std=1e-3,
            )
            rmse = rmse_predictor(model, test_paths, args.batch_size, args.device)
            results["nc_perturb"].append(rmse)
            print(f"[nc_perturb] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_nc_perturb.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "nc_pgd_rankk" in args.methods:
            X, hist = train_nc_pgd_rankk(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_pgd, iters=args.iters_pgd,
                batch_size=args.batch_size, eta=args.eta_pgd,
                device=args.device, seed=args.seed + 100*k + 5,
                init="spectral",
            )
            rmse = rmse_predictor(X, test_paths, args.batch_size, args.device)
            results["nc_pgd_rankk"].append(rmse)
            print(f"[nc_pgd_rankk] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_nc_pgd_rankk.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        # --- Convex ---
        if "c_softimpute" in args.methods:
            X, hist = train_c_softimpute(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_cvx, iters=args.iters_softimpute,
                batch_size=args.batch_size, lam=args.lam_softimpute,
                device=args.device, seed=args.seed + 100*k + 10,
                init="zeros",
            )
            rmse = rmse_predictor(X, test_paths, args.batch_size, args.device)
            results["c_softimpute"].append(rmse)
            print(f"[c_softimpute] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_c_softimpute.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "c_ista_nuc" in args.methods:
            X, hist = train_c_ista_nuc(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_cvx, iters=args.iters_ista,
                batch_size=args.batch_size, lam=args.lam_ista, eta=args.eta_ista,
                device=args.device, seed=args.seed + 100*k + 11,
                init="spectral",
            )
            rmse = rmse_predictor(X, test_paths, args.batch_size, args.device)
            results["c_ista_nuc"].append(rmse)
            print(f"[c_ista_nuc] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_c_ista_nuc.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "c_fista_nuc" in args.methods:
            X, hist = train_c_fista_nuc(
                train_paths, val_paths, n_users, n_items,
                rank=args.rank_cvx, iters=args.iters_fista,
                batch_size=args.batch_size, lam=args.lam_fista, eta=args.eta_fista,
                device=args.device, seed=args.seed + 100*k + 12,
            )
            rmse = rmse_predictor(X, test_paths, args.batch_size, args.device)
            results["c_fista_nuc"].append(rmse)
            print(f"[c_fista_nuc] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_c_fista_nuc.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

        if "c_fw_trace" in args.methods:
            fw, hist = train_c_fw_trace(
                train_paths, val_paths, n_users, n_items,
                iters=args.iters_fw, tau=args.tau_fw,
                batch_size=args.batch_size, device=args.device,
                seed=args.seed + 100*k + 13,
            )
            rmse = rmse_predictor(fw, test_paths, args.batch_size, args.device)
            results["c_fw_trace"].append(rmse)
            print(f"[c_fw_trace] fold{k} test_rmse={rmse:.4f}")
            save_json(os.path.join(args.fold_dir, f"history_fold{k}_c_fw_trace.json"),
                      {"train_rmse": hist.train_rmse, "val_rmse": hist.val_rmse})

    summary = {m: summarize(v) for m, v in results.items()}
    print("\n================= SUMMARY (5-fold test RMSE) =================")
    for m in results:
        s = summary[m]
        print(f"{m:14s} mean={s['mean']:.4f} std={s['std']:.4f} vals={[f'{x:.4f}' for x in results[m]]}")

    out = {
        "meta": {
            "fold_dir": args.fold_dir,
            "device": args.device,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "methods": args.methods,
            "n_users": n_users,
            "n_items": n_items,
            "params": vars(args),
        },
        "results": results,
        "summary": summary,
    }
    save_path = os.path.join(args.fold_dir, "results.json")
    save_json(save_path, out)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
