"""
Monte Carlo study and real-data comparison -- parallel, GPU-ready.

Experiments (results/ files):
  size   E1 size and power of five tests ............ size_power.csv
  local  E2 power against weak departures ........... local_power.csv/.pdf
  growth E3 selection of the number of units, 4 designs (p = 2, 5, 10),
            3 sample sizes, 3 vs 10 random restarts, cost of each rule
            ......................................... growth.csv, growth_tests.txt
  real   E4 four real data sets, 10-fold CV ......... real.csv, real_summary.csv, real_tests.txt
  kmax   E4b real data at several K_max (8,12,16,24) .. kmax.csv, kmax_summary.csv
  sens   E5 sensitivity to grid size/scale and the variance estimator
            ......................................... sensitivity.csv
  init   E6 one-step initialisation of the new unit . init.csv
  deep   E7 layer-wise growth of a 2-hidden-layer network
            ......................................... deep.csv, deep_tests.txt
Raw per-replication results are in *_raw.csv.

Every experiment is split into tasks (a batch of replications for one
setting). Each task runs as one batched computation on one device. Tasks are
spread over worker processes, one per device slot (--per-gpu per GPU, or
--workers CPU processes); with one slot everything runs in this process.

Examples
  python -m supgrow.experiments all --quick                 # check that everything runs
  python -m supgrow.experiments all --devices cuda --batch 100
  python -m supgrow.experiments growth real --devices cuda  # several experiments
  python -m supgrow.experiments deep --verbose 2
  python -m supgrow.experiments real --kmax 16            # E4 with K_max = 16
  python -m supgrow.experiments kmax --kmax-levels 8 12 16 24
"""
import argparse
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# Load CUDA kernels on demand: each worker process needs much less memory.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, friedmanchisquare, binomtest

from supgrow import growth_test as gt

OUT = "results"
METHODS = {
    "suplm": "Sup-LM, bootstrap (proposed)",
    "bonf": "Sup-LM, Bonferroni",
    "lwg": "Random units (White 1989; Lee et al. 1993)",
    "tlg": "Taylor LM (Terasvirta et al. 1993; Medeiros et al. 2006)",
    "corr": "Max. correlation trigger (Cascade-Corr./GradMax type)",
    "ash": "Relative RSS drop (Ash 1989 type)",
    "valgrow": "Validation-based growth",
    "aic": "AIC", "bic": "BIC", "nic": "NIC (Murata et al. 1994)",
}
TESTS = ("boot", "bonf", "lwg", "tlg", "single")
ALPHAS = (0.01, 0.05, 0.10)

# Data-generating processes (p = 2, x ~ N(0, I), noise sd SIGMA)
SIGMA, P = 0.5, 2
# Unit directions (w1, w2) differ pairwise by at least 45 degrees modulo 180, so no unit is
# close to a (sign-flipped) copy of another: tanh is odd, and a unit with
# w3 ~ -w2 would be almost redundant, making K0 = 3 indistinguishable from 2.
UNITS = [(1.5, [0.0, 1.0, 0.5]), (-1.0, [0.5, -0.5, 1.5]), (1.2, [-1.0, -1.5, 0.55])]
GRID_M, GRID_SCALE, B = 500, 3.0, 499

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _worker_init(slots, threads, fp64, verbose=1):
    global DEVICE, DTYPE
    DEVICE = torch.device(slots.get())
    DTYPE = gt.pick_dtype(DEVICE, fp64)
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)
    torch.set_num_threads(max(1, threads))
    tag = f"[{DEVICE} pid {os.getpid()}]"
    gt.set_logger(lambda m: print(f"{time.strftime('%H:%M:%S')} {tag} {m}", flush=True), verbose)
    gt.log(1, f"worker ready: {DTYPE}, {threads} CPU thread(s)"
              + (f", GPU {torch.cuda.get_device_name(DEVICE)}" if DEVICE.type == "cuda" else ""))


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


def true_theta(K0, b, scale_last=1.0):
    a = torch.tensor([u[0] for u in UNITS[:K0]], dtype=DTYPE, device=DEVICE)
    a[-1] *= scale_last
    W = torch.tensor([u[1] for u in UNITS[:K0]], dtype=DTYPE, device=DEVICE)
    th = gt.pack(torch.zeros(1, dtype=DTYPE, device=DEVICE), a[None], W[None])
    return th.expand(b, -1).clone()


def simulate(b, n, K0, gen, scale_last=1.0):
    X = torch.randn(b, n, P, generator=gen, dtype=DTYPE, device=DEVICE)
    Xt = gt.add_intercept(X)
    th = true_theta(K0, b, scale_last)
    y = gt.predict(th, Xt, K0) + SIGMA * torch.randn(b, n, generator=gen, dtype=DTYPE, device=DEVICE)
    return Xt, y, th


def all_pvalues(theta, Xt, y, K, Gamma, gen):
    g = gt.growth_stats(theta, Xt, y, K, Gamma, B=B, gen=gen)
    out = dict(boot=g["p_boot"], bonf=g["p_bonf"], single=g["p_single"],
               lwg=gt.lwg_pvalue(theta, Xt, y, K, gen),
               tlg=gt.tlg_pvalue(theta, Xt, y, K))
    return {k: v.double().cpu().numpy() for k, v in out.items()}


def _rows(reps, **cols):
    """One dict per replication; array columns are split, scalars repeated."""
    pick = lambda v, i: v[i] if isinstance(v, np.ndarray) else v
    return [dict({k: pick(v, i) for k, v in cols.items()}, rep=int(r))
            for i, r in enumerate(reps)]


# ---------------------------------------------------------------------------
# Tasks (each runs inside a worker on DEVICE)
# ---------------------------------------------------------------------------
def task_size(t):
    gen = _gen(t["seed"])
    Gamma = gt.make_grid(P + 1, GRID_M, GRID_SCALE, 1000 + t["n"], DEVICE, DTYPE)
    b = len(t["reps"])
    Xt, y, th = simulate(b, t["n"], 2, gen)
    rows = []
    gt.log(1, f"  simulated {b} data sets, n={t['n']}")
    for K_fit, case in ((2, "size"), (1, "power")):
        gt.log(1, f"  {case}: fitting K={K_fit} and testing")
        inits = [th] if K_fit == 2 else []
        theta, _ = gt.fit_best(Xt, y, K_fit, inits, 3 if K_fit == 2 else 4, gen)
        pv = all_pvalues(theta, Xt, y, K_fit, Gamma, gen)
        rows += _rows(t["reps"], n=t["n"], case=case, **pv)
    return rows


def task_local(t):
    gen = _gen(t["seed"])
    Gamma = gt.make_grid(P + 1, GRID_M, GRID_SCALE, 7, DEVICE, DTYPE)
    b, n = len(t["reps"]), 500
    Xt, y, th = simulate(b, n, 2, gen, scale_last=t["delta"])
    th1 = gt.pack(th[:, 0], th[:, 1:2], th[:, 3:6][:, None, :])     # true unit 1 only
    theta, _ = gt.fit_best(Xt, y, 1, [th1], 2, gen)
    pv = all_pvalues(theta, Xt, y, 1, Gamma, gen)
    return _rows(t["reps"], delta=t["delta"], **pv)


# ---------------------------------------------------------------------------
# DGPs for the selection experiment E3
#   p2k2, p2k3 : the two-input designs (units in UNITS)
#   p5k3, p10k4: more inputs; unit directions are orthonormal (scaled by 1.5)
# ---------------------------------------------------------------------------
DGPS = {"p2k2": (2, 2), "p2k3": (2, 3), "p5k3": (5, 3), "p10k4": (10, 4)}
A_COEF = [1.5, -1.0, 1.2, -0.9]
BIASES = [0.0, 0.5, -0.5, 0.3]


def dgp_theta(name, b):
    p, K0 = DGPS[name]
    if p == 2:
        return true_theta(K0, b)
    g = torch.Generator().manual_seed(1234 + p)
    Q, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=torch.float64))
    W = torch.cat([torch.tensor(BIASES[:K0], dtype=torch.float64)[:, None], 1.5 * Q[:, :K0].T], 1)
    a = torch.tensor(A_COEF[:K0], dtype=torch.float64)
    th = gt.pack(torch.zeros(1, dtype=torch.float64), a[None], W[None])
    return th.to(dtype=DTYPE, device=DEVICE).expand(b, -1).clone()


def simulate_dgp(name, b, n, gen):
    p, K0 = DGPS[name]
    X = torch.randn(b, n, p, generator=gen, dtype=DTYPE, device=DEVICE)
    Xt = gt.add_intercept(X)
    th = dgp_theta(name, b)
    y = gt.predict(th, Xt, K0) + SIGMA * torch.randn(b, n, generator=gen, dtype=DTYPE, device=DEVICE)
    return Xt, y


def grid_for(d, M, seed):
    """Candidate grid with pre-activation scale comparable across input
    dimensions: uniform on [-s, s]^d with s = 3 sqrt(3/d) (s = 3 for d = 3)."""
    return gt.make_grid(d, M, 3.0 * math.sqrt(3.0 / d), seed, DEVICE, DTYPE)


CRITERIA = ("aic", "bic", "nic")


def _fit_and_select(Xt, y, Xte, yte, K_max, Gamma, gen, restarts=3):
    """Common path on all data, path on an 80% split for validation growth,
    and all decision rules. Returns everything needed by _score, so that the
    selections can be scored for K_max and for any smaller maximal size."""
    b, n, d = Xt.shape
    t0 = time.time()
    gt.log(1, f"  step 1/3: common path on all {n} observations (K = 0..{K_max})")
    thetas, rss, tfit = gt.fit_path(Xt, y, K_max, Gamma, gen, n_random=restarts, return_times=True)
    perm = torch.rand(b, n, generator=gen, device=DEVICE).argsort(1)
    ntr = int(0.8 * n)
    idx_tr, idx_va = perm[:, :ntr], perm[:, ntr:]
    take = lambda A, idx: torch.gather(A, 1, idx[..., None].expand(-1, -1, A.shape[2])) \
        if A.dim() == 3 else torch.gather(A, 1, idx)
    gt.log(1, f"  step 2/3: path on the 80% training split ({ntr} obs.) for validation growth")
    th_tr, _, tfit_tr = gt.fit_path(take(Xt, idx_tr), take(y, idx_tr), K_max, Gamma, gen,
                                    n_random=restarts, return_times=True)
    gt.log(1, f"  step 3/3: applying the decision rules (bootstrap B={B})")
    sel, pv, ttest, crit = gt.select_all(thetas, rss, Xt, y, Gamma, gen, B=B,
                                         val=(take(Xt, idx_va), take(y, idx_va), th_tr),
                                         return_info=True, return_crit=True)
    mse_full = gt.path_mse(thetas, Xte, yte)
    mse_tr = gt.path_mse(th_tr, Xte, yte)
    gt.log(1, f"  test MSE by size K=0..{K_max}: " +
           " ".join(f"{v:.3f}" for v in mse_full.mean(0).tolist()) + f"  ({time.time() - t0:.0f}s)")
    return dict(sel=sel, pv=pv, crit=crit, mse_full=mse_full, mse_tr=mse_tr,
                cf=np.cumsum(tfit), cf_tr=np.cumsum(tfit_tr), ct=np.cumsum(ttest))


def _truncate(sel, crit, L):
    """Selections that the same rules would make with maximal size L <= K_max.
    Forward rules stop at the first non-rejection, so their size is min(K, L);
    criteria are minimised over the sizes 0..L."""
    out = {}
    for m, K in sel.items():
        out[m] = crit[m][:, :L + 1].argmin(1) if m in gt.CRIT_NAMES else torch.clamp(K, max=L)
    return out


def _score(sel, f, K_max):
    """Test MSE of each selected network and cost accounting for maximal size K_max."""
    mse_full, mse_tr, cf, cf_tr, ct = f["mse_full"], f["mse_tr"], f["cf"], f["cf_tr"], f["ct"]
    out = {}
    for m, K in sel.items():
        src = mse_tr if m == "valgrow" else mse_full
        Kn = K.cpu().numpy()
        out[f"K_{m}"] = Kn
        out[f"mse_{m}"] = torch.gather(src, 1, K[:, None]).squeeze(1).double().cpu().numpy()
        if "_" in m:
            continue
        if m in CRITERIA:                      # needs every size
            out[f"fits_{m}"] = np.full_like(Kn, K_max)
            out[f"time_{m}"] = np.full(Kn.shape, cf[K_max])
        elif m == "valgrow":                   # forward, on the 80% split, one size past K
            last = np.minimum(Kn + 1, K_max)
            out[f"fits_{m}"] = last
            out[f"time_{m}"] = cf_tr[last]
        else:                                  # forward: sizes 1..K fitted, tests at 0..K
            out[f"fits_{m}"] = Kn
            out[f"time_{m}"] = cf[Kn] + (ct[np.minimum(Kn, K_max - 1)] if m == "suplm" else 0.0)
    return out


def _select_and_score(Xt, y, Xte, yte, K_max, Gamma, gen, restarts=3):
    """Common path, all rules, test MSE of each selected network and cost."""
    f = _fit_and_select(Xt, y, Xte, yte, K_max, Gamma, gen, restarts=restarts)
    out = _score(f["sel"], f, K_max)
    return out, f["mse_full"].double().cpu().numpy(), f["pv"].double().cpu().numpy()


def task_growth(t):
    gen = _gen(t["seed"])
    name, n, restarts = t["dgp"], t["n"], t["restarts"]
    p, K0 = DGPS[name]
    K_max = 6 if p == 2 else K0 + 4
    M = {2: 500, 5: 1000, 10: 2000}[p]
    Gamma = grid_for(p + 1, M, 100 * K0 + n + (0 if p == 2 else 7 * p))
    b = len(t["reps"])
    Xt, y = simulate_dgp(name, b, n, gen)
    gte = torch.Generator(device=DEVICE).manual_seed(10_000 + 100 * K0 + n + (0 if p == 2 else 7 * p))
    Xte1, yte1 = simulate_dgp(name, 1, 10_000, gte)
    Xte, yte = Xte1.expand(b, -1, -1).contiguous(), yte1.expand(b, -1).contiguous()
    out, mse_all, _ = _select_and_score(Xt, y, Xte, yte, K_max, Gamma, gen, restarts=restarts)
    out["mse_oracle"] = mse_all[:, K0]
    out["mse_best"] = mse_all.min(1)
    return _rows(t["reps"], dgp=name, p=p, K0=K0, n=n, restarts=restarts, K_max=K_max, **out)


# ---------------------------------------------------------------------------
# Real data (E4)
# ---------------------------------------------------------------------------
REAL = ("diabetes", "california", "kin8nm", "cpu_act")


def load_real(name):
    """Return X (n x p) and y (n) as float64 numpy arrays. California housing
    and the OpenML data sets are downloaded once and cached by scikit-learn."""
    import sklearn.datasets as ds
    if name == "diabetes":
        X, y = ds.load_diabetes(return_X_y=True)
    elif name == "california":
        X, y = ds.fetch_california_housing(return_X_y=True)
    else:
        X, y = ds.fetch_openml(name=name, version=1, return_X_y=True, as_frame=False, parser="auto")
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _real_fold(t):
    """Standardised train/test tensors of one CV fold of a real data set."""
    from sklearn.model_selection import KFold
    X, y = load_real(t["data"])
    tr, te = list(KFold(n_splits=t["folds"], shuffle=True, random_state=2026).split(X))[t["fold"]]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    ym, ys = y[tr].mean(), y[tr].std()
    T = lambda A: torch.as_tensor(A, dtype=DTYPE, device=DEVICE)[None]
    Xtr, Xte = gt.add_intercept(T((X[tr] - mu) / sd)), gt.add_intercept(T((X[te] - mu) / sd))
    ytr, yte = T((y[tr] - ym) / ys), T((y[te] - ym) / ys)
    return X, Xtr, ytr, Xte, yte


def task_real(t):
    X, Xtr, ytr, Xte, yte = _real_fold(t)
    gen = _gen(t["seed"])
    d, K_max = Xtr.shape[2], t.get("K_max", 8)
    Gamma = grid_for(d, 2000, 2026)
    gt.log(1, f"  data {t['data']}: n_train={Xtr.shape[1]}, p={d - 1}")
    out, mse_all, pv = _select_and_score(Xtr, ytr, Xte, yte, K_max, Gamma, gen, restarts=t["restarts"])
    row = {k: v[0] for k, v in out.items()}
    row.update({f"fixed_mse_K{K}": mse_all[0, K] for K in range(K_max + 1)})
    row.update(data=t["data"], fold=t["fold"], n=X.shape[0], p=X.shape[1],
               p_values=";".join(f"{q:.3f}" for q in pv[0]))
    return [row]


def task_kmax(t):
    """One CV fold, one path up to the largest K_max level; every rule is then
    evaluated at every level (the forward rules stop at min(K, level), the
    criteria are minimised over 0..level), so no size is fitted twice."""
    X, Xtr, ytr, Xte, yte = _real_fold(t)
    gen = _gen(t["seed"])
    levels = sorted(t["levels"])
    d, Ktop = Xtr.shape[2], levels[-1]
    Gamma = grid_for(d, 2000, 2026)
    gt.log(1, f"  data {t['data']}: n_train={Xtr.shape[1]}, p={d - 1}, K_max levels {levels}")
    f = _fit_and_select(Xtr, ytr, Xte, yte, Ktop, Gamma, gen, restarts=t["restarts"])
    rows = []
    for L in levels:
        out = _score(_truncate(f["sel"], f["crit"], L), f, L)
        row = {k: v[0] for k, v in out.items()}
        row.update(data=t["data"], fold=t["fold"], K_max=L, n=X.shape[0], p=X.shape[1])
        rows.append(row)
        gt.log(1, f"  K_max={L:2d}: " + ", ".join(f"{m} K={int(out[f'K_{m}'][0])}" for m in METHODS))
    pv = f["pv"][0].double().cpu().numpy()
    rows[-1]["p_values"] = ";".join(f"{q:.3f}" for q in pv)
    return rows


# ---------------------------------------------------------------------------
# Sensitivity of the test (E5) and initialisation study (E6)
# ---------------------------------------------------------------------------
SENS = [dict(M=M, s=3.0, robust=True) for M in (50, 200, 500, 2000)] + \
       [dict(M=500, s=s, robust=True) for s in (1.0, 5.0)] + \
       [dict(M=500, s=3.0, robust=False)]


def task_sens(t):
    gen = _gen(t["seed"])
    b, n = len(t["reps"]), 500
    Xt, y, th = simulate(b, n, 2, gen)
    rows = []
    fits = {2: gt.fit_best(Xt, y, 2, [th], 3, gen)[0], 1: gt.fit_best(Xt, y, 1, [], 4, gen)[0]}
    for k, cfg in enumerate(SENS):
        Gamma = gt.make_grid(P + 1, cfg["M"], cfg["s"], 9000 + k, DEVICE, DTYPE)
        for K_fit, case in ((2, "size"), (1, "power")):
            g = gt.growth_stats(fits[K_fit], Xt, y, K_fit, Gamma, B=B, gen=gen, robust=cfg["robust"])
            rows += _rows(t["reps"], case=case, M=cfg["M"], s=cfg["s"], robust=cfg["robust"],
                          p_boot=g["p_boot"].double().cpu().numpy())
    gt.log(1, f"  sensitivity: {len(SENS)} configurations done")
    return rows


@torch.no_grad()
def task_init(t):
    """From the best 1-unit fit, fit the 2-unit network from (a) the one-step
    sup-LM unit, (b) the 1-unit fit plus a random new unit, (c) a fully random
    start; compare with the best of 10 random starts."""
    gen = _gen(t["seed"])
    b, n = len(t["reps"]), 500
    Xt, y, _ = simulate(b, n, 2, gen)
    d = Xt.shape[2]
    th1, _ = gt.fit_best(Xt, y, 1, [], 4, gen)
    Gamma = gt.make_grid(d, GRID_M, GRID_SCALE, 4242, DEVICE, DTYPE)
    g = gt.growth_stats(th1, Xt, y, 1, Gamma, B=0, gen=gen)
    c, a, W = gt.unpack(th1, 1, d)
    starts = {
        "onestep": gt.pack(c, torch.cat([a, g["beta"][:, None]], 1), torch.cat([W, g["gamma"][:, None, :]], 1)),
        "warm_random": gt.pack(c, torch.cat([a, torch.zeros_like(a)], 1),
                               torch.cat([W, torch.randn(b, 1, d, generator=gen, dtype=DTYPE, device=DEVICE)], 1)),
        "random": gt.random_init(y, 2, d, gen),
    }
    out = {}
    for k, s0 in starts.items():
        _, rss, it = gt.lm_fit(Xt, y, 2, s0, return_iters=True)
        out[f"rss_{k}"] = (rss / n).double().cpu().numpy()
        out[f"iters_{k}"] = it.cpu().numpy()
        r0 = ((gt.predict(s0, Xt, 2) - y) ** 2).mean(1)
        out[f"rss0_{k}"] = r0.double().cpu().numpy()
    _, ref = gt.fit_best(Xt, y, 2, [], 10, gen)
    out["rss_ref"] = (ref / n).double().cpu().numpy()
    gt.log(1, "  init study: mean final RSS/n " + ", ".join(
        f"{k} {out[f'rss_{k}'].mean():.4f}" for k in starts) + f", reference {out['rss_ref'].mean():.4f}")
    return _rows(t["reps"], **out)


# ---------------------------------------------------------------------------
# Deep layer-wise growth (E7)
# ---------------------------------------------------------------------------
DEEP_P, DEEP_WIDTHS, DEEP_HMAX, DEEP_SIGMA = 4, (3, 2), (8, 8), 0.3
# rule name -> (version, growth rule, starting widths)
DEEP_RULES = {
    "suplm_v1": ("v1", "suplm", (1, 1)),      # first version: partial projection, Adam
    "suplm": ("v2", "suplm", (1, 1)),         # full projection, LM with restarts
    "suplm_s22": ("v2", "suplm", (2, 2)),     # same, starting from widths (2, 2)
    "gradmax": ("v2", "gradmax", (1, 1)),
    "valgrow": ("v2", "valgrow", (1, 1)),
    "schedule": ("v2", "schedule", (1, 1)),
}
DEEP_METHODS = tuple(DEEP_RULES)


def task_deep(t):
    from supgrow import deep_growth as dg
    gen = _gen(t["seed"])
    b, n = len(t["reps"]), t["n"]
    Tch = dg.teacher(DEEP_P, DEEP_WIDTHS, 77, DTYPE, DEVICE)
    X = torch.randn(b, n, DEEP_P, generator=gen, dtype=DTYPE, device=DEVICE)
    y = dg.teacher_predict(Tch, X) + DEEP_SIGMA * torch.randn(b, n, generator=gen, dtype=DTYPE, device=DEVICE)
    gte = torch.Generator(device=DEVICE).manual_seed(31_000 + n)
    Xte = torch.randn(1, 10_000, DEEP_P, generator=gte, dtype=DTYPE, device=DEVICE)
    yte = dg.teacher_predict(Tch, Xte) + DEEP_SIGMA * torch.randn(1, 10_000, generator=gte, dtype=DTYPE, device=DEVICE)
    Xte, yte = Xte.expand(b, -1, -1).contiguous(), yte.expand(b, -1).contiguous()
    M = t.get("M", 200)
    grids = [dg.layer_grid(DEEP_P, M, 3.0 * math.sqrt(3.0 / (DEEP_P + 1)), 501, DEVICE, DTYPE),
             dg.layer_grid(DEEP_HMAX[0], M, 2.0, 502, DEVICE, DTYPE)]
    perm = torch.rand(b, n, generator=gen, device=DEVICE).argsort(1)
    ntr = int(0.8 * n)
    take = lambda A, idx: torch.gather(A, 1, idx[..., None].expand(-1, -1, A.shape[2])) \
        if A.dim() == 3 else torch.gather(A, 1, idx)
    Xa, ya = take(X, perm[:, :ntr]), take(y, perm[:, :ntr])
    Xv, yv = take(X, perm[:, ntr:]), take(y, perm[:, ntr:])
    out = {}
    # reference architectures, fitted by LM with restarts
    arch = dg.Arch(DEEP_P, DEEP_HMAX)
    for name, widths in (("oracle", DEEP_WIDTHS), ("full", DEEP_HMAX)):
        g2 = _gen(t["seed"] + 2)
        um = dg.masks_from_widths(arch, widths, b, DEVICE)
        th, _ = dg.fit_deep(arch, X, y, um, [], 5, g2)
        out[f"mse_{name}"] = ((dg.predict_v2(arch, th, Xte, um) - yte) ** 2).mean(1).double().cpu().numpy()
        gt.log(1, f"  reference {name} {widths}: test MSE {out[f'mse_{name}'].mean():.4f}")
    for meth, (ver, rule, w0) in DEEP_RULES.items():
        gt.log(1, f"  deep growth: {meth} (version {ver}, rule {rule}, start {w0})")
        g2 = _gen(t["seed"] + 1)                      # same randomness for every rule
        t0 = time.time()
        if ver == "v1":
            model, info = dg.grow(rule, X, y, DEEP_HMAX, grids, g2, B=t["B"], init_width=w0)
            with torch.no_grad():
                mse = ((model.forward(Xte) - yte) ** 2).mean(1)
        else:
            Xf, yf, val = (Xa, ya, (Xv, yv)) if rule == "valgrow" else (X, y, None)
            ar, th, um, info = dg.grow_v2(rule, Xf, yf, DEEP_HMAX, grids, g2, init_width=w0,
                                          B=t["B"], val=val)
            mse = ((dg.predict_v2(ar, th, Xte, um) - yte) ** 2).mean(1)
        w = info["widths"].cpu().numpy()
        out[f"w1_{meth}"], out[f"w2_{meth}"] = w[:, 0], w[:, 1]
        out[f"exact_{meth}"] = ((w[:, 0] == DEEP_WIDTHS[0]) & (w[:, 1] == DEEP_WIDTHS[1])).astype(float)
        out[f"params_{meth}"] = info["n_params"].cpu().numpy()
        out[f"mse_{meth}"] = mse.double().cpu().numpy()
        out[f"near_{meth}"] = (out[f"mse_{meth}"] <= 1.05 * out["mse_oracle"]).astype(float)
        out[f"time_{meth}"] = np.full(b, (time.time() - t0) / b)
        gt.log(1, f"    -> widths mean ({w[:, 0].mean():.2f}, {w[:, 1].mean():.2f}), exact "
                  f"{out[f'exact_{meth}'].mean():.2f}, test MSE {out[f'mse_{meth}'].mean():.4f}")
    return _rows(t["reps"], n=n, **out)


TASKS = dict(size=task_size, local=task_local, growth=task_growth, real=task_real,
             kmax=task_kmax, sens=task_sens, init=task_init, deep=task_deep)


def _mem():
    if DEVICE.type != "cuda":
        return ""
    return f", GPU mem peak {torch.cuda.max_memory_allocated(DEVICE) / 2**20:.0f} MB"


def run_task(t):
    t0 = time.time()
    gt.log(1, f"START {_label(t)} ({len(t.get('reps', [0]))} replication(s))")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    rows = TASKS[t["exp"]](t)
    dt = time.time() - t0
    gt.log(1, f"END   {_label(t)}: {dt:.1f}s{_mem()}")
    return t, rows, dt, str(DEVICE)


def _summary(t, rows):
    """One-line result of a finished task, printed by the main process."""
    df = pd.DataFrame(rows)
    if t["exp"] in ("size", "local"):
        parts = []
        for case, g in (df.groupby("case") if "case" in df else [("delta", df)]):
            parts.append(f"{case}: " + " ".join(f"{k}={np.mean(g[k] <= 0.05):.2f}" for k in TESTS))
        return "reject@5% " + " | ".join(parts)
    if t["exp"] == "growth":
        K0 = DGPS[t["dgp"]][1]
        return "P(K=K0) " + " ".join(f"{m}={np.mean(df[f'K_{m}'] == K0):.2f}" for m in METHODS) + \
            f" | fits suplm={df.fits_suplm.mean():.1f} bic={df.fits_bic.mean():.1f}"
    if t["exp"] == "sens":
        g = df.groupby(["case", "M", "s", "robust"]).p_boot.apply(lambda p: np.mean(p <= 0.05))
        return "reject@5% " + " ".join(f"{k[0]}/M{k[1]}/s{k[2]}/{'rob' if k[3] else 'hom'}={v:.2f}"
                                       for k, v in g.items())
    if t["exp"] == "init":
        return "mean RSS/n " + " ".join(f"{c[4:]}={df[c].mean():.4f}" for c in df if c.startswith("rss_"))
    if t["exp"] == "deep":
        return " | ".join(f"{m}: exact={df[f'exact_{m}'].mean():.2f} near={df[f'near_{m}'].mean():.2f} "
                          f"mse={df[f'mse_{m}'].mean():.4f}" for m in DEEP_METHODS) + \
            f" | oracle mse={df['mse_oracle'].mean():.4f}"
    if t["exp"] == "kmax":
        return f"{t['data']} fold {t['fold']}: " + " | ".join(
            f"K_max={int(r['K_max'])}: suplm={int(r['K_suplm'])} bic={int(r['K_bic'])} "
            f"valgrow={int(r['K_valgrow'])}" for _, r in df.iterrows())
    r = df.iloc[0]
    return f"{r['data']} fold {r['fold']}: K " + " ".join(f"{m}={int(r[f'K_{m}'])}" for m in METHODS) + \
        f" | test MSE suplm={r['mse_suplm']:.3f} bic={r['mse_bic']:.3f}"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def device_slots(args):
    use_cuda = torch.cuda.is_available() and args.devices in ("auto", "cuda")
    if args.devices == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--devices cuda requested but no GPU is visible")
    if use_cuda:
        slots = [f"cuda:{i}" for i in range(torch.cuda.device_count()) for _ in range(args.per_gpu)]
        threads = max(1, (os.cpu_count() or 1) // len(slots))
    else:
        w = args.workers or max(1, (os.cpu_count() or 1) // 2)
        slots = ["cpu"] * w
        threads = max(1, (os.cpu_count() or 1) // w)
    return slots, threads


def _progress(i, n, t, rows, dt, dev, t_start):
    el = time.time() - t_start
    eta = el / (i + 1) * (n - i - 1)
    bar = "#" * int(30 * (i + 1) / n)
    print(f"{time.strftime('%H:%M:%S')} [MAIN] [{bar:<30s}] {i+1}/{n} done | {_label(t)} "
          f"took {dt:.0f}s on {dev} | elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
    print(f"{time.strftime('%H:%M:%S')} [MAIN]     {_summary(t, rows)}", flush=True)


def _run_inprocess(tasks, args, threads, fp64, t_start, done0=0, n_total=None):
    """Run tasks one after another in this process (no extra processes)."""
    n_total = n_total or len(tasks)
    rows = []
    for i, t in enumerate(tasks):
        t, r, dt, dev = run_task(t)
        rows += r
        _progress(done0 + i, n_total, t, r, dt, dev, t_start)
    return rows


def run_all(tasks, args):
    slots, threads = device_slots(args)
    fp64 = True if args.fp64 else None
    print(f"{time.strftime('%H:%M:%S')} [MAIN] {len(tasks)} tasks on {len(slots)} worker(s): "
          f"{sorted(set(slots))}, {threads} thread(s) each, batch {args.batch}", flush=True)
    t_start = time.time()
    if len(slots) == 1:                                  # in-process, no spawning
        _worker_init(_ListQueue(slots), threads, fp64, args.verbose)
        return _run_inprocess(tasks, args, threads, fp64, t_start)

    ctx = mp.get_context("spawn")                        # required for CUDA and Windows
    q = ctx.Queue()                                      # no extra Manager process
    for s_ in slots:
        q.put(s_)
    rows, done = [], set()
    try:
        with ProcessPoolExecutor(len(slots), mp_context=ctx, initializer=_worker_init,
                                 initargs=(q, threads, fp64, args.verbose)) as ex:
            futs = {ex.submit(run_task, t): k for k, t in enumerate(tasks)}
            for i, f in enumerate(as_completed(futs)):
                t, r, dt, dev = f.result()
                rows += r
                done.add(futs[f])
                _progress(len(done) - 1, len(tasks), t, r, dt, dev, t_start)
    except (BrokenProcessPool, MemoryError, OSError) as e:
        left = [t for k, t in enumerate(tasks) if k not in done]
        print(f"{time.strftime('%H:%M:%S')} [MAIN] WARNING: worker processes failed "
              f"({type(e).__name__}: {e}). This usually means the computer ran out of RAM "
              f"while starting a second Python+CUDA process. Continuing the remaining "
              f"{len(left)} task(s) in this process on {slots[0]}. "
              f"Use --per-gpu 1 (one GPU) or fewer --workers to avoid this.", flush=True)
        _worker_init(_ListQueue([slots[0]]), max(1, os.cpu_count() or 1), fp64, args.verbose)
        rows += _run_inprocess(left, args, threads, fp64, t_start, len(done), len(tasks))
    return rows


class _ListQueue:
    def __init__(self, items):
        self.items = list(items)

    def get(self):
        return self.items.pop(0)


def _label(t):
    return " ".join(f"{k}={v}" for k, v in t.items() if k not in ("reps", "seed"))


def batches(R, size):
    return [list(range(s, min(R, s + size))) for s in range(0, R, size)]


# ---------------------------------------------------------------------------
# Experiments: build tasks, run, summarise
# ---------------------------------------------------------------------------
def exp_size(args):
    R = 20 if args.quick else 500
    tasks = [dict(exp="size", n=n, reps=r, seed=1000 * n + i)
             for n in (200, 500, 1000) for i, r in enumerate(batches(R, args.batch))]
    df = pd.DataFrame(run_all(tasks, args)).sort_values(["n", "case", "rep"])
    df.to_csv(f"{OUT}/size_power_raw.csv", index=False)
    summ = [dict(n=n, case=case, alpha=a, R=len(g), **{t: np.mean(g[t] <= a) for t in TESTS})
            for (n, case), g in df.groupby(["n", "case"]) for a in ALPHAS]
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT}/size_power.csv", index=False)
    show("E1 size and power (rejection rates)", summ[summ.alpha == 0.05])


def exp_local(args):
    R = 20 if args.quick else 300
    deltas = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    tasks = [dict(exp="local", delta=dl, reps=r, seed=7_000 + 100 * j + i)
             for j, dl in enumerate(deltas) for i, r in enumerate(batches(R, args.batch))]
    df = pd.DataFrame(run_all(tasks, args)).sort_values(["delta", "rep"])
    df.to_csv(f"{OUT}/local_power_raw.csv", index=False)
    s = pd.DataFrame([dict(delta=dl, R=len(g), **{t: np.mean(g[t] <= 0.05) for t in TESTS})
                      for dl, g in df.groupby("delta")])
    s.to_csv(f"{OUT}/local_power.csv", index=False)
    show("E2 local power at alpha=0.05", s)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    se = lambda q: 1.96 * np.sqrt(q * (1 - q) / s.R)
    for col, lab, mk in (("boot", "Sup-LM, bootstrap (proposed)", "o"),
                         ("bonf", "Sup-LM, Bonferroni", "s"),
                         ("lwg", "Random units (LWG)", "^"),
                         ("tlg", "Taylor LM (TLG)", "D"),
                         ("single", "Single random unit", "v")):
        ax.errorbar(s.delta, s[col], yerr=se(s[col]), marker=mk, ms=4, capsize=2, label=lab)
    ax.axhline(0.05, ls=":", c="grey", lw=1)
    ax.set_xlabel(r"amplitude $\delta$ of the missing unit")
    ax.set_ylabel(r"rejection rate ($\alpha=0.05$)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/local_power.pdf")


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def compare(df, K0=None, ref="suplm"):
    """Friedman across methods on MSE; each method vs the proposed one:
    Wilcoxon on MSE and exact McNemar on correct size, Holm-adjusted."""
    meths = [m for m in METHODS if f"mse_{m}" in df]
    lines = []
    try:
        st, p = friedmanchisquare(*[df[f"mse_{m}"] for m in meths])
        lines.append(f"Friedman (MSE, {len(meths)} methods): chi2={st:.2f}, p={p:.3g}")
    except ValueError as e:
        lines.append(f"Friedman not computed: {e}")
    pw, pm = {}, {}
    for m in meths:
        if m == ref:
            continue
        dlt = df[f"mse_{ref}"] - df[f"mse_{m}"]
        try:
            pw[m] = wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
        except ValueError:
            pw[m] = 1.0
        if K0 is not None:
            a, b = df[f"K_{ref}"] == K0, df[f"K_{m}"] == K0
            n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
            pm[m] = binomtest(n01, n01 + n10).pvalue if n01 + n10 > 0 else 1.0
    hw, hm = holm(pw), (holm(pm) if pm else {})
    for m in pw:
        line = (f"{ref} vs {m}: mean MSE diff {np.mean(df[f'mse_{ref}'] - df[f'mse_{m}']):.4f}, "
                f"Wilcoxon Holm p={hw[m]:.3g}")
        if m in hm:
            line += f"; McNemar (correct size) Holm p={hm[m]:.3g}"
        lines.append(line)
    return lines


def exp_growth(args):
    R = 10 if args.quick else 200
    tasks = []
    for name in DGPS:
        p, K0 = DGPS[name]
        for n in (250, 500, 1000):
            for restarts in (3, 10):
                for i, r in enumerate(batches(R, args.batch)):
                    seed = 50_000 + 1000 * K0 + n + i + (0 if p == 2 else 100_000 * p)
                    tasks.append(dict(exp="growth", dgp=name, n=n, restarts=restarts, reps=r, seed=seed))
    df = pd.DataFrame(run_all(tasks, args)).sort_values(["dgp", "n", "restarts", "rep"])
    df.to_csv(f"{OUT}/growth_raw.csv", index=False)
    summ, lines = [], []
    extra = [c[2:] for c in df.columns if c.startswith("K_") and "_" in c[2:]]
    for (name, n, rs), g in df.groupby(["dgp", "n", "restarts"]):
        K0 = DGPS[name][1]
        for m in list(METHODS) + extra:
            k = g[f"K_{m}"]
            row = dict(dgp=name, K0=K0, n=n, restarts=rs, method=m,
                       P_under=np.mean(k < K0), P_exact=np.mean(k == K0), P_over=np.mean(k > K0),
                       mse_mean=g[f"mse_{m}"].mean(), mse_sd=g[f"mse_{m}"].std(), R=len(g))
            if f"fits_{m}" in g:
                row.update(fits_mean=g[f"fits_{m}"].mean(), time_mean=g[f"time_{m}"].mean())
            summ.append(row)
        summ.append(dict(dgp=name, K0=K0, n=n, restarts=rs, method="oracle",
                         mse_mean=g.mse_oracle.mean(), mse_sd=g.mse_oracle.std(), R=len(g)))
        lines += [f"== {name} (K0={K0}), n={n}, restarts={rs} =="] + compare(g, K0=K0)
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT}/growth.csv", index=False)
    with open(f"{OUT}/growth_tests.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    main = summ[summ.method.isin(list(METHODS) + ["oracle"])]
    show("E3 selection of the size", main)
    show("E3 paired comparisons with the proposed test", "\n".join(lines))
    piv = summ[summ.method.isin(METHODS)].pivot_table(index="method", columns=["dgp", "restarts", "n"],
                                                      values="P_exact")
    show("E3 P(K = K0) by design, restarts and n", piv)


def exp_real(args):
    folds = 3 if args.quick else 10
    names = []
    for name in REAL:                                    # download / cache once, here
        try:
            X, _ = load_real(name)
            names.append(name)
            print(f"[MAIN] data set {name}: n={X.shape[0]}, p={X.shape[1]}", flush=True)
        except Exception as e:                           # no internet, renamed data set, ...
            print(f"[MAIN] WARNING: skipping data set {name}: {type(e).__name__}: {e}", flush=True)
    tasks = [dict(exp="real", data=nm, fold=f, folds=folds, restarts=10, K_max=args.kmax,
                  seed=2026 + 100 * j + f)
             for j, nm in enumerate(names) for f in range(folds)]
    df = pd.DataFrame(run_all(tasks, args)).sort_values(["data", "fold"])
    df.to_csv(f"{OUT}/real.csv", index=False)
    lines, tabs = [], []
    for nm, g in df.groupby("data"):
        lines += [f"== {nm} =="] + compare(g)
        for m in METHODS:
            tabs.append(dict(data=nm, method=m, K_median=g[f"K_{m}"].median(),
                             K_min=g[f"K_{m}"].min(), K_max=g[f"K_{m}"].max(),
                             mse_mean=g[f"mse_{m}"].mean(), mse_sd=g[f"mse_{m}"].std(),
                             fits_mean=g[f"fits_{m}"].mean(), time_mean=g[f"time_{m}"].mean()))
    tabs = pd.DataFrame(tabs)
    tabs.to_csv(f"{OUT}/real_summary.csv", index=False)
    with open(f"{OUT}/real_tests.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    show(f"E4 real data (10-fold CV, K_max={args.kmax})", tabs)
    show("E4 paired comparisons with the proposed test", "\n".join(lines))


def exp_kmax(args):
    """E4b: the real data sets at several maximal sizes K_max (default 8, 12,
    16, 24). Shows whether each rule stops on its own or at the cap, and how
    the cost of the criteria grows with K_max while forward testing does not."""
    folds = 3 if args.quick else 10
    levels = sorted(set(args.kmax_levels if not args.quick else [4, 6]))
    names = []
    for name in REAL:
        try:
            X, _ = load_real(name)
            names.append(name)
            print(f"[MAIN] data set {name}: n={X.shape[0]}, p={X.shape[1]}", flush=True)
        except Exception as e:
            print(f"[MAIN] WARNING: skipping data set {name}: {type(e).__name__}: {e}", flush=True)
    tasks = [dict(exp="kmax", data=nm, fold=f, folds=folds, restarts=10, levels=levels,
                  seed=2026 + 100 * j + f)
             for j, nm in enumerate(names) for f in range(folds)]
    df = pd.DataFrame(run_all(tasks, args)).sort_values(["data", "K_max", "fold"])
    df.to_csv(f"{OUT}/kmax.csv", index=False)
    rows = []
    for (nm, L), g in df.groupby(["data", "K_max"]):
        for m in METHODS:
            K = g[f"K_{m}"]
            rows.append(dict(data=nm, K_max=L, method=m, K_median=K.median(), K_mean=K.mean(),
                             K_min=K.min(), K_maxsel=K.max(), at_cap=np.mean(K >= L),
                             mse_mean=g[f"mse_{m}"].mean(), mse_sd=g[f"mse_{m}"].std(),
                             fits_mean=g[f"fits_{m}"].mean(), time_mean=g[f"time_{m}"].mean()))
    tabs = pd.DataFrame(rows)
    tabs.to_csv(f"{OUT}/kmax_summary.csv", index=False)
    for what in ("K_median", "at_cap", "mse_mean", "time_mean"):
        piv = tabs.pivot_table(index=["data", "method"], columns="K_max", values=what)
        show(f"E4b {what} by K_max (rows: data set, method)", piv)


def exp_sens(args):
    R = 20 if args.quick else 300
    tasks = [dict(exp="sens", reps=r, seed=88_000 + i) for i, r in enumerate(batches(R, args.batch))]
    df = pd.DataFrame(run_all(tasks, args))
    df.to_csv(f"{OUT}/sensitivity_raw.csv", index=False)
    s = df.groupby(["case", "M", "s", "robust"]).p_boot.agg(
        rate=lambda p: np.mean(p <= 0.05), R="size").reset_index()
    s.to_csv(f"{OUT}/sensitivity.csv", index=False)
    show("E5 sensitivity of the sup-LM test (n=500, alpha=0.05)", s)


def exp_init(args):
    R = 20 if args.quick else 300
    tasks = [dict(exp="init", reps=r, seed=66_000 + i) for i, r in enumerate(batches(R, args.batch))]
    df = pd.DataFrame(run_all(tasks, args))
    df.to_csv(f"{OUT}/init_raw.csv", index=False)
    rows = []
    for k in ("onestep", "warm_random", "random"):
        rows.append(dict(start=k, rss_start=df[f"rss0_{k}"].mean(), rss_final=df[f"rss_{k}"].mean(),
                         rss_final_sd=df[f"rss_{k}"].std(),
                         P_reach_best=np.mean(df[f"rss_{k}"] <= df.rss_ref * (1 + 1e-3)),
                         iters=df[f"iters_{k}"].mean()))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/init.csv", index=False)
    show("E6 initialisation of the added unit (DGP-2, n=500, K=1 -> 2)", s)


def exp_deep(args):
    R = 6 if args.quick else 100
    bsz = min(args.batch, 20)
    tasks = [dict(exp="deep", n=n, reps=r, B=99 if args.quick else 199, M=100 if args.quick else 200,
                  seed=77_000 + n + i)
             for n in (500, 2000) for i, r in enumerate(batches(R, bsz))]
    df = pd.DataFrame(run_all(tasks, args)).sort_values(["n", "rep"])
    df.to_csv(f"{OUT}/deep_raw.csv", index=False)
    rows = []
    for n, g in df.groupby("n"):
        for m in DEEP_METHODS:
            rows.append(dict(n=n, method=m, w1_mean=g[f"w1_{m}"].mean(), w2_mean=g[f"w2_{m}"].mean(),
                             P_exact=g[f"exact_{m}"].mean(), P_near_oracle=g[f"near_{m}"].mean(),
                             params=g[f"params_{m}"].mean(),
                             mse_mean=g[f"mse_{m}"].mean(), mse_sd=g[f"mse_{m}"].std(),
                             time=g[f"time_{m}"].mean(), R=len(g)))
        for m in ("oracle", "full"):
            rows.append(dict(n=n, method=m, mse_mean=g[f"mse_{m}"].mean(), mse_sd=g[f"mse_{m}"].std(), R=len(g)))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/deep.csv", index=False)
    lines = []
    for n, g in df.groupby("n"):
        lines.append(f"== n={n} ==")
        meths = list(DEEP_METHODS) + ["oracle", "full"]
        ref = "suplm"
        try:
            st, p = friedmanchisquare(*[g[f"mse_{m}"] for m in meths])
            lines.append(f"Friedman (MSE, {len(meths)} rules): chi2={st:.2f}, p={p:.3g}")
        except ValueError as e:
            lines.append(f"Friedman not computed: {e}")
        pw = {}
        for m in meths:
            if m == ref:
                continue
            dlt = g["mse_suplm"] - g[f"mse_{m}"]
            pw[m] = wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
        for m, ph in holm(pw).items():
            lines.append(f"suplm vs {m}: mean MSE diff {np.mean(g['mse_suplm'] - g[f'mse_{m}']):.4f}, "
                         f"Wilcoxon Holm p={ph:.3g}")
    with open(f"{OUT}/deep_tests.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    show("E7 deep layer-wise growth (teacher widths (3, 2), max (8, 8)); suplm = version 2", s)
    show("E7 paired comparisons", "\n".join(lines))


def show(title, obj):
    print(f"\n{'=' * 78}\n{title}\n{'-' * 78}", flush=True)
    if isinstance(obj, pd.DataFrame):
        with pd.option_context("display.width", 200, "display.max_columns", 50,
                               "display.float_format", "{:.3f}".format):
            print(obj.to_string(index=False), flush=True)
    else:
        print(obj, flush=True)
    print("=" * 78 + "\n", flush=True)


def environment(args):
    print("=" * 78)
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | "
          f"CPU cores: {os.cpu_count()}")
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {pr.name}, {pr.total_memory / 2**30:.1f} GB")
    print(f"options: {vars(args)}")
    print("=" * 78, flush=True)


EXPERIMENTS = dict(size=exp_size, local=exp_local, growth=exp_growth, real=exp_real,
                   kmax=exp_kmax, sens=exp_sens, init=exp_init, deep=exp_deep)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=["all"], choices=["all", *EXPERIMENTS],
                    help="experiments to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="few replications, to check the pipeline")
    ap.add_argument("--devices", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--per-gpu", type=int, default=1, help="worker processes per GPU")
    ap.add_argument("--workers", type=int, default=0, help="CPU worker processes (0 = half the cores)")
    ap.add_argument("--batch", type=int, default=50, help="replications per task (GPU: 50-200; CPU: 5-20)")
    ap.add_argument("--fp64", action="store_true", help="float64 on GPU too (slower on consumer GPUs)")
    ap.add_argument("--kmax", type=int, default=8,
                    help="maximal number of hidden units in E4 'real' (default 8; e.g. 16)")
    ap.add_argument("--kmax-levels", type=int, nargs="+", default=[8, 12, 16, 24],
                    help="maximal sizes compared in E4b 'kmax' (one path up to the largest)")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                    help="0: progress only; 1: every fit and test (default); 2: also LM iterations")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    environment(args)
    t0 = time.time()
    for name, fn in EXPERIMENTS.items():
        if "all" in args.which or name in args.which:
            print(f"\n########## experiment: {name} ##########", flush=True)
            te = time.time()
            fn(args)
            print(f"########## {name} finished in {(time.time() - te) / 60:.1f} min; "
                  f"files in {OUT}/ ##########", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
