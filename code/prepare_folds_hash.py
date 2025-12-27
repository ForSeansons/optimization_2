#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os


def fold_id(uid: int, mid: int, ts: int, k: int, seed: int) -> int:
    key = f"{seed}:{uid}:{mid}:{ts}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "little") % k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=str, required=True, help="path/to/ratings.dat")
    ap.add_argument("--out_dir", type=str, required=True, help="output dir for fold*.txt + meta.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    outs = [open(os.path.join(args.out_dir, f"fold{i}.txt"), "w", encoding="utf-8") for i in range(args.k)]

    user2idx, item2idx = {}, {}
    n_lines = 0

    with open(args.ratings, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 4:
                continue
            uid = int(parts[0])
            mid = int(parts[1])
            r = float(parts[2])
            ts = int(parts[3])

            if uid not in user2idx:
                user2idx[uid] = len(user2idx)
            if mid not in item2idx:
                item2idx[mid] = len(item2idx)

            u = user2idx[uid]
            i = item2idx[mid]
            fi = fold_id(uid, mid, ts, args.k, args.seed)

            # save as: u_idx \t i_idx \t rating
            outs[fi].write(f"{u}\t{i}\t{r}\n")
            n_lines += 1

    for fp in outs:
        fp.close()

    meta = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "n_ratings": n_lines,
        "k": args.k,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
