#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import os
from typing import Dict, Tuple, Iterator


def fold_id(raw_uid: int, raw_mid: int, ts: int, k: int, seed: int) -> int:
    key = f"{seed}:{raw_uid}:{raw_mid}:{ts}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "little") % k


def iter_ratings_ml10m_dat(path: str) -> Iterator[Tuple[int, int, float, int]]:
    # format: UserID::MovieID::Rating::Timestamp
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 4:
                continue
            yield int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])


def iter_ratings_ml20m_csv(path: str) -> Iterator[Tuple[int, int, float, int]]:
    # format header: userId,movieId,rating,timestamp
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        # tolerate alternative cases
        for row in reader:
            uid = int(row.get("userId") or row.get("userid") or row.get("UserID") or row.get("UserId"))
            mid = int(row.get("movieId") or row.get("movieid") or row.get("MovieID") or row.get("MovieId"))
            r = float(row.get("rating") or row.get("Rating"))
            ts = int(float(row.get("timestamp") or row.get("Timestamp")))
            yield uid, mid, r, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=str, required=True, help="ratings.dat (ml-10m) OR ratings.csv (ml-20m)")
    ap.add_argument("--out_dir", type=str, required=True, help="output dir for fold0..fold4.txt and meta.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", type=str, default="auto", choices=["auto", "ml10m_dat", "ml20m_csv"])
    ap.add_argument("--min_rating", type=float, default=None, help="optional filter")
    ap.add_argument("--max_rating", type=float, default=None, help="optional filter")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    outs = [open(os.path.join(args.out_dir, f"fold{i}.txt"), "w", encoding="utf-8") for i in range(args.k)]

    # choose iterator
    fmt = args.format
    if fmt == "auto":
        if args.ratings.lower().endswith(".csv"):
            fmt = "ml20m_csv"
        else:
            fmt = "ml10m_dat"

    if fmt == "ml20m_csv":
        it = iter_ratings_ml20m_csv(args.ratings)
        dataset_name = "ml-20m"
    else:
        it = iter_ratings_ml10m_dat(args.ratings)
        dataset_name = "ml-10m"

    user2idx: Dict[int, int] = {}
    item2idx: Dict[int, int] = {}

    n_lines = 0
    n_filtered = 0

    for raw_u, raw_i, r, ts in it:
        if args.min_rating is not None and r < args.min_rating:
            n_filtered += 1
            continue
        if args.max_rating is not None and r > args.max_rating:
            n_filtered += 1
            continue

        if raw_u not in user2idx:
            user2idx[raw_u] = len(user2idx)
        if raw_i not in item2idx:
            item2idx[raw_i] = len(item2idx)

        u = user2idx[raw_u]
        i = item2idx[raw_i]
        fi = fold_id(raw_u, raw_i, ts, args.k, args.seed)
        outs[fi].write(f"{u}\t{i}\t{r}\n")
        n_lines += 1

        if n_lines % 2_000_000 == 0:
            print(f"[split] seen={n_lines:,} users={len(user2idx):,} items={len(item2idx):,}")

    for fp in outs:
        fp.close()

    meta = {
        "dataset": dataset_name,
        "ratings_path": os.path.abspath(args.ratings),
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "n_ratings": n_lines,
        "n_filtered": n_filtered,
        "k": args.k,
        "seed": args.seed,
        "format": fmt,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
