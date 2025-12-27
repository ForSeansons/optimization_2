# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RMIN, RMAX = 0.5, 5.0


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clip_rating(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, RMIN, RMAX)


def _read_triplets(path: str) -> Iterator[Tuple[int, int, float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) < 3:
                continue
            yield int(a[0]), int(a[1]), float(a[2])


def stream_batches(
    paths: List[str],
    batch_size: int,
    device: str,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    u_buf, i_buf, r_buf = [], [], []
    for p in paths:
        for u, i, r in _read_triplets(p):
            u_buf.append(u)
            i_buf.append(i)
            r_buf.append(r)
            if len(u_buf) >= batch_size:
                uu = torch.tensor(u_buf, device=device, dtype=torch.long)
                ii = torch.tensor(i_buf, device=device, dtype=torch.long)
                rr = torch.tensor(r_buf, device=device, dtype=torch.float32)
                yield uu, ii, rr
                u_buf.clear(); i_buf.clear(); r_buf.clear()
    if u_buf:
        uu = torch.tensor(u_buf, device=device, dtype=torch.long)
        ii = torch.tensor(i_buf, device=device, dtype=torch.long)
        rr = torch.tensor(r_buf, device=device, dtype=torch.float32)
        yield uu, ii, rr


def stream_batches_chunk_shuffled(
    paths: List[str],
    batch_size: int,
    device: str,
    shuffle_chunk: int = 1_000_000,
    seed: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    u_buf: List[int] = []
    i_buf: List[int] = []
    r_buf: List[float] = []

    def flush_chunk():
        if not u_buf:
            return
        u = np.asarray(u_buf, dtype=np.int32)
        i = np.asarray(i_buf, dtype=np.int32)
        r = np.asarray(r_buf, dtype=np.float32)
        perm = rng.permutation(u.shape[0])
        u, i, r = u[perm], i[perm], r[perm]
        n = u.shape[0]
        for s in range(0, n, batch_size):
            uu = torch.from_numpy(u[s:s+batch_size]).to(device=device, dtype=torch.long, non_blocking=True)
            ii = torch.from_numpy(i[s:s+batch_size]).to(device=device, dtype=torch.long, non_blocking=True)
            rr = torch.from_numpy(r[s:s+batch_size]).to(device=device, dtype=torch.float32, non_blocking=True)
            yield uu, ii, rr
        u_buf.clear(); i_buf.clear(); r_buf.clear()

    for p in paths:
        for u, i, r in _read_triplets(p):
            u_buf.append(u); i_buf.append(i); r_buf.append(r)
            if len(u_buf) >= shuffle_chunk:
                yield from flush_chunk()

    yield from flush_chunk()


def compute_global_mean(paths: List[str]) -> float:
    s, n = 0.0, 0
    for p in paths:
        for _, _, r in _read_triplets(p):
            s += float(r)
            n += 1
    return s / max(n, 1)


# =========================
# Predictor interface
# =========================
class Predictor:
    def predict(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@torch.no_grad()
def rmse_predictor(pred: Predictor, paths: List[str], batch_size: int, device: str) -> float:
    ss, n = 0.0, 0
    for u, i, r in stream_batches(paths, batch_size, device):
        y = clip_rating(pred.predict(u, i))
        err = y - r
        ss += float((err * err).sum().item())
        n += int(r.numel())
    return math.sqrt(ss / max(n, 1))


# =========================
# Nonconvex: MF models
# =========================
class BiasedMF(nn.Module, Predictor):
    def __init__(self, n_users: int, n_items: int, rank: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.U = nn.Embedding(n_users, rank)
        self.V = nn.Embedding(n_items, rank)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.mu + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + (self.U(u) * self.V(i)).sum(dim=-1)

    def predict(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.forward(u, i)


class NoBiasMF(nn.Module, Predictor):
    def __init__(self, n_users: int, n_items: int, rank: int, mu_init: float = 0.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(mu_init), dtype=torch.float32))
        self.U = nn.Embedding(n_users, rank)
        self.V = nn.Embedding(n_items, rank)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.mu + (self.U(u) * self.V(i)).sum(dim=-1)

    def predict(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.forward(u, i)


def _mf_reg_loss_biased(model: BiasedMF, u: torch.Tensor, i: torch.Tensor, reg: float, reg_bias: float) -> torch.Tensor:
    return reg * (model.U(u).pow(2).mean() + model.V(i).pow(2).mean()) + reg_bias * (model.bu(u).pow(2).mean() + model.bi(i).pow(2).mean())


def _mf_reg_loss_nobias(model: NoBiasMF, u: torch.Tensor, i: torch.Tensor, reg: float) -> torch.Tensor:
    return reg * (model.U(u).pow(2).mean() + model.V(i).pow(2).mean())


@dataclass
class History:
    train_rmse: List[float]
    val_rmse: List[float]


# =========================
# Randomized SVD for operators (A@O and A^T@Q)
# =========================
def randomized_svd_operator(
    n_users: int,
    n_items: int,
    rank: int,
    matmul: Callable[[torch.Tensor], torch.Tensor],   # (n_items,k)->(n_users,k)
    rmatmul: Callable[[torch.Tensor], torch.Tensor],  # (n_users,k)->(n_items,k) = A^T Q
    device: str,
    oversample: int = 8,
    power: int = 0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    k = rank + oversample
    Omega = torch.randn(n_items, k, device=device) / math.sqrt(k)
    Y = matmul(Omega)  # (n_users,k)
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    for _ in range(power):
        T = rmatmul(Q)      # (n_items,k)
        Y = matmul(T)       # (n_users,k)
        Q, _ = torch.linalg.qr(Y, mode="reduced")

    T = rmatmul(Q)          # (n_items,k) = A^T Q
    B = T.t().contiguous()  # (k,n_items) = (A^T Q)^T = Q^T A
    Ub, Sb, Vbt = torch.linalg.svd(B, full_matrices=False)
    Ub = Ub[:, :rank]
    Sb = Sb[:rank]
    V = Vbt[:rank, :].t().contiguous()  # (n_items, rank)
    U = Q @ Ub                           # (n_users, rank)
    return U, Sb, V


# =========================
# Low-rank predictors (for convex & rank-projected methods)
# =========================
@dataclass
class LowRankSVD(Predictor):
    U: torch.Tensor   # (n_users,r)
    S: torch.Tensor   # (r,)
    V: torch.Tensor   # (n_items,r)
    mu: float = 0.0

    def predict(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # mu + sum_k U[u,k]*S[k]*V[i,k]
        return self.mu + (self.U[u] * self.S.unsqueeze(0) * self.V[i]).sum(dim=-1)

    def matmul_items(self, O: torch.Tensor) -> torch.Tensor:
        # X @ O , O:(n_items,k) -> (n_users,k)
        # U diag(S) V^T O
        return self.U @ (self.S.unsqueeze(1) * (self.V.t() @ O))

    def rmatmul_users(self, Q: torch.Tensor) -> torch.Tensor:
        # X^T @ Q, Q:(n_users,k)->(n_items,k)
        return self.V @ (self.S.unsqueeze(1) * (self.U.t() @ Q))


# =========================
# Convex/Nonconvex operator steps from residuals on 次
# =========================
def svd_of_Z_from_residual(
    X: LowRankSVD,
    data_paths: List[str],
    batch_size: int,
    device: str,
    rank: int,
    coeff: float,
    # delta(u,i) = coeff * (r_resid - pred_resid)
    # where r_resid = r - mu, pred_resid = X.predict(u,i) - mu
    oversample: int,
    power: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def Z_matmul(O: torch.Tensor) -> torch.Tensor:
        Y = X.matmul_items(O)
        for u, i, r in stream_batches(data_paths, batch_size, device):
            rr = r - float(X.mu)
            pred = X.predict(u, i) - float(X.mu)
            delta = coeff * (rr - pred)          # (B,)
            Y.index_add_(0, u, O[i] * delta.unsqueeze(1))
        return Y

    def Z_rmatmul(Q: torch.Tensor) -> torch.Tensor:
        T = X.rmatmul_users(Q)
        for u, i, r in stream_batches(data_paths, batch_size, device):
            rr = r - float(X.mu)
            pred = X.predict(u, i) - float(X.mu)
            delta = coeff * (rr - pred)
            T.index_add_(0, i, Q[u] * delta.unsqueeze(1))
        return T

    return randomized_svd_operator(
        n_users=X.U.shape[0],
        n_items=X.V.shape[0],
        rank=rank,
        matmul=Z_matmul,
        rmatmul=Z_rmatmul,
        device=device,
        oversample=oversample,
        power=power,
        seed=seed,
    )


def spectral_init_from_data(
    data_paths: List[str],
    n_users: int,
    n_items: int,
    rank: int,
    mu: float,
    batch_size: int,
    device: str,
    oversample: int = 8,
    power: int = 1,
    seed: int = 0,
) -> LowRankSVD:
    # A is sparse residual matrix: r - mu on 次
    def A_matmul(O: torch.Tensor) -> torch.Tensor:
        Y = torch.zeros(n_users, O.shape[1], device=device)
        for u, i, r in stream_batches(data_paths, batch_size, device):
            rr = (r - mu)
            Y.index_add_(0, u, O[i] * rr.unsqueeze(1))
        return Y

    def A_rmatmul(Q: torch.Tensor) -> torch.Tensor:
        T = torch.zeros(n_items, Q.shape[1], device=device)
        for u, i, r in stream_batches(data_paths, batch_size, device):
            rr = (r - mu)
            T.index_add_(0, i, Q[u] * rr.unsqueeze(1))
        return T

    U, S, V = randomized_svd_operator(
        n_users=n_users,
        n_items=n_items,
        rank=rank,
        matmul=A_matmul,
        rmatmul=A_rmatmul,
        device=device,
        oversample=oversample,
        power=power,
        seed=seed,
    )
    # small scale to keep stable
    S = 0.05 * S
    return LowRankSVD(U=U, S=S, V=V, mu=mu)


# =========================
# Nonconvex trainings
# =========================
def train_nc_mf_sgd(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, epochs: int,
    batch_size: int, lr: float,
    reg: float, reg_bias: float,
    device: str, seed: int,
    shuffle_chunk: int = 1_000_000,
    early_stop: int = 2,
) -> Tuple[BiasedMF, History]:
    set_seed(seed)
    model = BiasedMF(n_users, n_items, rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = History([], [])
    best, bad = 1e9, 0

    for ep in range(1, epochs + 1):
        model.train()
        ss, n = 0.0, 0
        for u, i, r in stream_batches_chunk_shuffled(train_paths, batch_size, device, shuffle_chunk=shuffle_chunk, seed=seed + ep):
            pred = model(u, i)
            loss = F.mse_loss(pred, r) + _mf_reg_loss_biased(model, u, i, reg, reg_bias)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                err = pred - r
                ss += float((err * err).sum().item()); n += int(r.numel())

        train_rmse = math.sqrt(ss / max(n, 1))
        val_rmse = rmse_predictor(model, val_paths, batch_size, device)
        hist.train_rmse.append(train_rmse); hist.val_rmse.append(val_rmse)
        print(f"[nc_mf_sgd] epoch {ep}/{epochs} train_rmse~{train_rmse:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse + 1e-4 < best:
            best = val_rmse; bad = 0
        else:
            bad += 1
            if bad >= early_stop:
                break

        lr *= 0.95
        for g in opt.param_groups:
            g["lr"] = lr

    return model, hist


def train_nc_mf_nobias(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, epochs: int,
    batch_size: int, lr: float,
    reg: float,
    device: str, seed: int,
    shuffle_chunk: int = 1_000_000,
) -> Tuple[NoBiasMF, History]:
    set_seed(seed)
    mu = compute_global_mean(train_paths)
    model = NoBiasMF(n_users, n_items, rank, mu_init=mu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = History([], [])

    for ep in range(1, epochs + 1):
        model.train()
        ss, n = 0.0, 0
        for u, i, r in stream_batches_chunk_shuffled(train_paths, batch_size, device, shuffle_chunk=shuffle_chunk, seed=seed + ep):
            pred = model(u, i)
            loss = F.mse_loss(pred, r) + _mf_reg_loss_nobias(model, u, i, reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                err = pred - r
                ss += float((err * err).sum().item()); n += int(r.numel())

        train_rmse = math.sqrt(ss / max(n, 1))
        val_rmse = rmse_predictor(model, val_paths, batch_size, device)
        hist.train_rmse.append(train_rmse); hist.val_rmse.append(val_rmse)
        print(f"[nc_mf_nobias] epoch {ep}/{epochs} train_rmse~{train_rmse:.4f} val_rmse={val_rmse:.4f}")

        lr *= 0.95
        for g in opt.param_groups:
            g["lr"] = lr

    return model, hist


def train_nc_alt_block(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, epochs: int,
    batch_size: int, lr: float,
    reg: float, reg_bias: float,
    device: str, seed: int,
    shuffle_chunk: int = 1_000_000,
) -> Tuple[BiasedMF, History]:
    set_seed(seed)
    model = BiasedMF(n_users, n_items, rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = History([], [])

    def set_requires(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad_(flag)

    for ep in range(1, epochs + 1):
        # A: update U,bu,mu
        set_requires(model.V, False); set_requires(model.bi, False)
        set_requires(model.U, True); set_requires(model.bu, True)
        model.mu.requires_grad_(True)

        model.train()
        for u, i, r in stream_batches_chunk_shuffled(train_paths, batch_size, device, shuffle_chunk=shuffle_chunk, seed=seed + 1000 + ep):
            pred = model(u, i)
            loss = F.mse_loss(pred, r) + _mf_reg_loss_biased(model, u, i, reg, reg_bias)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # B: update V,bi,mu
        set_requires(model.U, False); set_requires(model.bu, False)
        set_requires(model.V, True); set_requires(model.bi, True)
        model.mu.requires_grad_(True)

        model.train()
        for u, i, r in stream_batches_chunk_shuffled(train_paths, batch_size, device, shuffle_chunk=shuffle_chunk, seed=seed + 2000 + ep):
            pred = model(u, i)
            loss = F.mse_loss(pred, r) + _mf_reg_loss_biased(model, u, i, reg, reg_bias)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # unfreeze
        set_requires(model.U, True); set_requires(model.V, True)
        set_requires(model.bu, True); set_requires(model.bi, True)

        val_rmse = rmse_predictor(model, val_paths, batch_size, device)
        hist.train_rmse.append(float("nan")); hist.val_rmse.append(val_rmse)
        print(f"[nc_alt_block] epoch {ep}/{epochs} val_rmse={val_rmse:.4f}")

        lr *= 0.95
        for g in opt.param_groups:
            g["lr"] = lr

    return model, hist


def train_nc_spec_alt(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, epochs: int,
    batch_size: int, lr: float,
    reg: float, reg_bias: float,
    device: str, seed: int,
    shuffle_chunk: int = 1_000_000,
) -> Tuple[BiasedMF, History]:
    # spectral init using residual matrix (r - mu), then set U,V weights
    set_seed(seed)
    mu = compute_global_mean(train_paths)
    X0 = spectral_init_from_data(train_paths, n_users, n_items, rank, mu, batch_size, device, power=1, seed=seed)
    model = BiasedMF(n_users, n_items, rank).to(device)
    with torch.no_grad():
        # initialize around mu; biases start 0; factors from spectral
        model.mu.copy_(torch.tensor(mu, device=device))
        model.U.weight.copy_(X0.U)
        model.V.weight.copy_(X0.V)
        # absorb S into U for MF-style
        model.U.weight.mul_(X0.S.sqrt().unsqueeze(0))
        model.V.weight.mul_(X0.S.sqrt().unsqueeze(0))

    print("[nc_spec_alt] spectral init done -> alt-block")
    return train_nc_alt_block(
        train_paths=train_paths, val_paths=val_paths,
        n_users=n_users, n_items=n_items,
        rank=rank, epochs=epochs, batch_size=batch_size, lr=lr,
        reg=reg, reg_bias=reg_bias, device=device, seed=seed + 7,
        shuffle_chunk=shuffle_chunk,
    )


def train_nc_perturb(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, epochs: int,
    batch_size: int, lr: float,
    reg: float, reg_bias: float,
    device: str, seed: int,
    plateau_patience: int = 2,
    noise_std: float = 1e-3,
    shuffle_chunk: int = 1_000_000,
) -> Tuple[BiasedMF, History]:
    set_seed(seed)
    model = BiasedMF(n_users, n_items, rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = History([], [])
    best, bad = 1e9, 0

    for ep in range(1, epochs + 1):
        model.train()
        ss, n = 0.0, 0
        for u, i, r in stream_batches_chunk_shuffled(train_paths, batch_size, device, shuffle_chunk=shuffle_chunk, seed=seed + ep):
            pred = model(u, i)
            loss = F.mse_loss(pred, r) + _mf_reg_loss_biased(model, u, i, reg, reg_bias)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                err = pred - r
                ss += float((err * err).sum().item()); n += int(r.numel())

        train_rmse = math.sqrt(ss / max(n, 1))
        val_rmse = rmse_predictor(model, val_paths, batch_size, device)
        hist.train_rmse.append(train_rmse); hist.val_rmse.append(val_rmse)
        print(f"[nc_perturb] epoch {ep}/{epochs} train_rmse~{train_rmse:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse + 1e-4 < best:
            best = val_rmse; bad = 0
        else:
            bad += 1
            if bad >= plateau_patience:
                with torch.no_grad():
                    model.U.weight.add_(noise_std * torch.randn_like(model.U.weight))
                    model.V.weight.add_(noise_std * torch.randn_like(model.V.weight))
                    model.bu.weight.add_(noise_std * torch.randn_like(model.bu.weight))
                    model.bi.weight.add_(noise_std * torch.randn_like(model.bi.weight))
                print(f"[nc_perturb] plateau -> perturb(std={noise_std})")
                bad = 0

        lr *= 0.95
        for g in opt.param_groups:
            g["lr"] = lr

    return model, hist


def train_nc_pgd_rankk(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, iters: int,
    batch_size: int,
    eta: float,
    device: str,
    seed: int,
    oversample: int = 8,
    power: int = 0,
    init: str = "spectral",
) -> Tuple[LowRankSVD, History]:
    set_seed(seed)
    mu = compute_global_mean(train_paths)

    if init == "spectral":
        X = spectral_init_from_data(train_paths, n_users, n_items, rank, mu, batch_size, device, power=1, seed=seed)
    else:
        U = torch.randn(n_users, rank, device=device) * 0.01
        V = torch.randn(n_items, rank, device=device) * 0.01
        S = torch.ones(rank, device=device) * 0.01
        X = LowRankSVD(U=U, S=S, V=V, mu=mu)

    hist = History([], [])
    for t in range(1, iters + 1):
        # Z = X + eta*(r_resid - pred_resid) on 次, then take top-r SVD (projection)
        U, S, V = svd_of_Z_from_residual(
            X=X, data_paths=train_paths, batch_size=batch_size, device=device,
            rank=rank, coeff=eta, oversample=oversample, power=power, seed=seed + 100 + t
        )
        X = LowRankSVD(U=U, S=S, V=V, mu=mu)

        val_rmse = rmse_predictor(X, val_paths, batch_size, device)
        hist.train_rmse.append(float("nan")); hist.val_rmse.append(val_rmse)
        print(f"[nc_pgd_rankk] iter {t}/{iters} val_rmse={val_rmse:.4f}")

    return X, hist


# =========================
# Convex trainings
# =========================
def train_c_softimpute(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, iters: int,
    batch_size: int,
    lam: float,
    device: str,
    seed: int,
    oversample: int = 8,
    power: int = 0,
    init: str = "zeros",
) -> Tuple[LowRankSVD, History]:
    set_seed(seed)
    mu = compute_global_mean(train_paths)

    if init == "spectral":
        X = spectral_init_from_data(train_paths, n_users, n_items, rank, mu, batch_size, device, power=1, seed=seed)
    else:
        U = torch.zeros(n_users, rank, device=device)
        V = torch.zeros(n_items, rank, device=device)
        S = torch.zeros(rank, device=device)
        X = LowRankSVD(U=U, S=S, V=V, mu=mu)

    hist = History([], [])
    for t in range(1, iters + 1):
        # SoftImpute step: Z = X + (r_resid - pred_resid) on 次 ; X_{new}=S_lambda(Z)
        U, S, V = svd_of_Z_from_residual(
            X=X, data_paths=train_paths, batch_size=batch_size, device=device,
            rank=rank, coeff=1.0, oversample=oversample, power=power, seed=seed + 10 + t
        )
        S = torch.clamp(S - lam, min=0.0)
        X = LowRankSVD(U=U, S=S, V=V, mu=mu)

        val_rmse = rmse_predictor(X, val_paths, batch_size, device)
        hist.train_rmse.append(float("nan")); hist.val_rmse.append(val_rmse)
        print(f"[c_softimpute] iter {t}/{iters} val_rmse={val_rmse:.4f} nnzS={(S>0).sum().item()}")

    return X, hist


def train_c_ista_nuc(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, iters: int,
    batch_size: int,
    lam: float,
    eta: float,
    device: str,
    seed: int,
    oversample: int = 8,
    power: int = 0,
    init: str = "spectral",
) -> Tuple[LowRankSVD, History]:
    set_seed(seed)
    mu = compute_global_mean(train_paths)
    if init == "spectral":
        X = spectral_init_from_data(train_paths, n_users, n_items, rank, mu, batch_size, device, power=1, seed=seed)
    else:
        U = torch.randn(n_users, rank, device=device) * 0.01
        V = torch.randn(n_items, rank, device=device) * 0.01
        S = torch.ones(rank, device=device) * 0.01
        X = LowRankSVD(U=U, S=S, V=V, mu=mu)

    hist = History([], [])
    for t in range(1, iters + 1):
        # Z = X + eta*(r_resid - pred_resid) on 次 ; prox: shrink by eta*lam
        U, S, V = svd_of_Z_from_residual(
            X=X, data_paths=train_paths, batch_size=batch_size, device=device,
            rank=rank, coeff=eta, oversample=oversample, power=power, seed=seed + 20 + t
        )
        S = torch.clamp(S - eta * lam, min=0.0)
        X = LowRankSVD(U=U, S=S, V=V, mu=mu)

        val_rmse = rmse_predictor(X, val_paths, batch_size, device)
        hist.train_rmse.append(float("nan")); hist.val_rmse.append(val_rmse)
        print(f"[c_ista_nuc] iter {t}/{iters} val_rmse={val_rmse:.4f} nnzS={(S>0).sum().item()}")

    return X, hist


def train_c_fista_nuc(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    rank: int, iters: int,
    batch_size: int,
    lam: float,
    eta: float,
    device: str,
    seed: int,
    oversample: int = 8,
    power: int = 0,
) -> Tuple[LowRankSVD, History]:
    # NOTE: This is a practical (approx) FISTA in factor space; still useful for comparison.
    set_seed(seed)
    mu = compute_global_mean(train_paths)
    X = spectral_init_from_data(train_paths, n_users, n_items, rank, mu, batch_size, device, power=1, seed=seed)
    Y = X
    t_k = 1.0

    hist = History([], [])
    for it in range(1, iters + 1):
        U, S, V = svd_of_Z_from_residual(
            X=Y, data_paths=train_paths, batch_size=batch_size, device=device,
            rank=rank, coeff=eta, oversample=oversample, power=power, seed=seed + 30 + it
        )
        S = torch.clamp(S - eta * lam, min=0.0)
        X_new = LowRankSVD(U=U, S=S, V=V, mu=mu)

        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        beta = (t_k - 1.0) / t_new

        # approx extrapolation in factor space
        Y = LowRankSVD(
            U=X_new.U + beta * (X_new.U - X.U),
            S=X_new.S + beta * (X_new.S - X.S),
            V=X_new.V + beta * (X_new.V - X.V),
            mu=mu,
        )
        X = X_new
        t_k = t_new

        val_rmse = rmse_predictor(X, val_paths, batch_size, device)
        hist.train_rmse.append(float("nan")); hist.val_rmse.append(val_rmse)
        print(f"[c_fista_nuc] iter {it}/{iters} val_rmse={val_rmse:.4f} nnzS={(X.S>0).sum().item()}")

    return X, hist


@dataclass
class FWTrace(Predictor):
    n_users: int
    n_items: int
    tau: float
    mu: float
    w: List[float]
    U_atoms: List[torch.Tensor]  # (n_users,)
    V_atoms: List[torch.Tensor]  # (n_items,)

    def predict(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        if not self.w:
            return torch.full_like(u, float(self.mu), dtype=torch.float32)
        s = torch.zeros_like(u, dtype=torch.float32) + float(self.mu)
        for wj, Uj, Vj in zip(self.w, self.U_atoms, self.V_atoms):
            s = s + float(wj) * (Uj[u] * Vj[i]).to(torch.float32)
        return s


def _fw_top_singular_vecs_of_grad(
    fw: FWTrace,
    train_paths: List[str],
    batch_size: int,
    device: str,
    power_iters: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # power method on sparse gradient G, where G(u,i)=pred_resid - r_resid
    v = torch.randn(fw.n_items, device=device)
    v = v / (v.norm() + 1e-8)

    for _ in range(power_iters):
        uvec = torch.zeros(fw.n_users, device=device)
        for u, i, r in stream_batches(train_paths, batch_size, device):
            rr = r - float(fw.mu)
            pred_resid = (fw.predict(u, i) - float(fw.mu)).to(device)
            resid = pred_resid - rr
            uvec.index_add_(0, u, resid * v[i])
        uvec = uvec / (uvec.norm() + 1e-8)

        vvec = torch.zeros(fw.n_items, device=device)
        for u, i, r in stream_batches(train_paths, batch_size, device):
            rr = r - float(fw.mu)
            pred_resid = (fw.predict(u, i) - float(fw.mu)).to(device)
            resid = pred_resid - rr
            vvec.index_add_(0, i, resid * uvec[u])
        v = vvec / (vvec.norm() + 1e-8)

    return uvec, v


def train_c_fw_trace(
    train_paths: List[str], val_paths: List[str],
    n_users: int, n_items: int,
    iters: int,
    tau: float,
    batch_size: int,
    device: str,
    seed: int,
    power_iters: int = 1,
) -> Tuple[FWTrace, History]:
    set_seed(seed)
    mu = compute_global_mean(train_paths)
    fw = FWTrace(n_users=n_users, n_items=n_items, tau=tau, mu=mu, w=[], U_atoms=[], V_atoms=[])
    hist = History([], [])

    for t in range(iters):
        uvec, vvec = _fw_top_singular_vecs_of_grad(fw, train_paths, batch_size, device, power_iters=power_iters)
        gamma = 2.0 / (t + 2.0)

        # scale old weights
        fw.w = [wj * (1.0 - gamma) for wj in fw.w]

        # add new rank-1 atom for -G direction: S = -tau * u v^T
        fw.w.append(gamma * (-tau))
        fw.U_atoms.append(uvec.detach())
        fw.V_atoms.append(vvec.detach())

        val_rmse = rmse_predictor(fw, val_paths, batch_size, device)
        hist.train_rmse.append(float("nan")); hist.val_rmse.append(val_rmse)
        print(f"[c_fw_trace] iter {t+1}/{iters} val_rmse={val_rmse:.4f} atoms={len(fw.w)}")

    return fw, hist
