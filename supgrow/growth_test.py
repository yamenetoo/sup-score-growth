"""
Sup-score (LM) growth test for single-hidden-layer tanh networks -- PyTorch,
batched over replications so that one call fits / tests many data sets at
once on a GPU (or on CPU threads).

Shapes: b = batch of independent data sets, n = sample size, d = p + 1
(inputs with an intercept column), K = hidden units, q = 1 + K + K d.

Model:  f(x; theta) = c + sum_k a_k tanh(w_k' xt),  xt = (1, x)
theta is stored as [c, a_1..a_K, vec(W)] with W of shape (K, d).
"""
import itertools
import math
import time
import torch

TINY = 1e-30

# ---------------------------------------------------------------------------
# Progress logging. Level 1: one line per fitted size / test; level 2: also
# Levenberg--Marquardt iterations. experiments.py sets the printer.
# ---------------------------------------------------------------------------
VERBOSE = 1
_PRINT = print


def set_logger(printer=print, verbose=1):
    global _PRINT, VERBOSE
    _PRINT, VERBOSE = printer, verbose


def log(level, msg):
    if VERBOSE >= level:
        _PRINT(msg)


def _q(x):
    """Short summary 'mean [min, max]' of a tensor."""
    x = x.detach().double()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return "nan"
    return f"{x.mean():.4g} [{x.min():.4g}, {x.max():.4g}]"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def pick_dtype(device, fp64=None):
    """float64 on CPU; float32 on GPU (consumer GPUs are slow in float64),
    unless fp64 is forced."""
    if fp64 is None:
        fp64 = torch.device(device).type == "cpu"
    return torch.float64 if fp64 else torch.float32


def chi2_sf(x, df):
    """Upper tail of chi2_df, elementwise, as a tensor."""
    a = torch.full_like(x, float(df) / 2)
    return torch.special.gammaincc(a, x.clamp_min(0) / 2)


def safe_solve(A, Bm):
    """Batched solve of A X = Bm that never raises: data sets whose A is
    singular (redundant hidden units, or float32 round-off) are solved with the
    Moore--Penrose pseudo-inverse instead."""
    Xs, info = torch.linalg.solve_ex(A, Bm)
    bad = (info != 0) | ~torch.isfinite(Xs).flatten(1).all(1)
    if bool(bad.any()):
        Xs = Xs.clone()
        Xs[bad] = torch.linalg.pinv(A[bad], hermitian=True) @ Bm[bad]
    return Xs


def _ridge(A, rel):
    """Add rel * mean(diag A) to the diagonal (scale-free regularisation)."""
    q = A.shape[-1]
    lam = rel * torch.diagonal(A, dim1=-2, dim2=-1).mean(-1).clamp_min(TINY)
    return A + lam[..., None, None] * torch.eye(q, dtype=A.dtype, device=A.device)


def _rel_ridge(dtype):
    return 1e-10 if dtype == torch.float64 else 1e-6


def add_intercept(X):
    return torch.cat([torch.ones_like(X[..., :1]), X], dim=-1)


def n_params(K, d):
    return 1 + K + K * d


def unpack(theta, K, d):
    b = theta.shape[0]
    return theta[:, 0], theta[:, 1:1 + K], theta[:, 1 + K:].reshape(b, K, d)


def pack(c, a, W):
    return torch.cat([c[:, None], a, W.reshape(W.shape[0], -1)], dim=1)


# ---------------------------------------------------------------------------
# Network, Jacobian, batched Levenberg--Marquardt
# ---------------------------------------------------------------------------
def predict(theta, Xt, K):
    b, n, d = Xt.shape
    c, a, W = unpack(theta, K, d)
    if K == 0:
        return c[:, None].expand(b, n)
    H = torch.tanh(torch.bmm(Xt, W.transpose(1, 2)))           # b x n x K
    return c[:, None] + torch.bmm(H, a[:, :, None]).squeeze(-1)


def jacobian(theta, Xt, K):
    """b x n x q Jacobian of f with respect to theta."""
    b, n, d = Xt.shape
    c, a, W = unpack(theta, K, d)
    ones = Xt.new_ones(b, n, 1)
    if K == 0:
        return ones
    H = torch.tanh(torch.bmm(Xt, W.transpose(1, 2)))
    D = (1.0 - H * H) * a[:, None, :]                            # b x n x K
    JW = (D[..., None] * Xt[:, :, None, :]).reshape(b, n, K * d)
    return torch.cat([ones, H, JW], dim=2)


@torch.no_grad()
def lm_fit(Xt, y, K, theta0, max_iter=400, tol=None, return_iters=False):
    """Batched Levenberg--Marquardt for least squares. Every data set in the
    batch has its own damping and stops on its own convergence.
    With return_iters=True also returns the number of iterations per data set."""
    theta = theta0.clone()
    b, n, d = Xt.shape
    q = theta.shape[1]
    tol = (1e-10 if theta.dtype == torch.float64 else 1e-7) if tol is None else tol
    eye = torch.eye(q, dtype=theta.dtype, device=theta.device)
    r = predict(theta, Xt, K) - y
    rss = (r * r).sum(1)
    lam = theta.new_full((b,), 1e-3)
    active = torch.ones(b, dtype=torch.bool, device=theta.device)
    iters = torch.zeros(b, dtype=torch.long, device=theta.device)
    t0 = time.time()
    it = 0
    for it in range(max_iter):
        iters += active.long()
        J = jacobian(theta, Xt, K)
        Jt = J.transpose(1, 2)
        A = Jt @ J
        g = (Jt @ r[:, :, None]).squeeze(-1)
        Mmat = A + lam[:, None, None] * (torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2)) + 1e-6 * eye)
        step, info = torch.linalg.solve_ex(Mmat, -g[:, :, None])
        step = step.squeeze(-1)
        cand = theta + step
        rc = predict(cand, Xt, K) - y
        rssc = (rc * rc).sum(1)
        ok = active & (info == 0) & torch.isfinite(rssc) & (rssc < rss)
        rel = (rss - rssc) / rss.clamp_min(TINY)
        theta = torch.where(ok[:, None], cand, theta)
        r = torch.where(ok[:, None], rc, r)
        rss = torch.where(ok, rssc, rss)
        lam = torch.where(ok, lam * 0.3, lam * 4.0).clamp(1e-12, 1e10)
        done = (ok & (rel < tol)) | (lam >= 1e10)
        active = active & ~done
        if VERBOSE >= 2 and it % 50 == 0:
            log(2, f"      LM K={K} iter {it:3d}: active {int(active.sum())}/{b}, "
                   f"RSS/n {_q(rss / n)}, damping {_q(lam)}")
        if not bool(active.any()):
            break
    log(2, f"      LM K={K} finished after {it + 1} iterations "
           f"({int((~active).sum())}/{b} converged, {time.time() - t0:.1f}s)")
    if return_iters:
        return theta, rss, iters
    return theta, rss


def random_init(y, K, d, gen):
    """Random starting values. Input weights have sd 1 for p <= 2 (as in the
    two-input simulations) and sd 1.5/sqrt(d) for more inputs, so that the
    hidden units are not saturated at the start."""
    b = y.shape[0]
    kw = dict(dtype=y.dtype, device=y.device, generator=gen)
    sw = 1.0 if d <= 3 else 1.5 / math.sqrt(d)
    return pack(y.mean(1), 0.5 * torch.randn(b, K, **kw), sw * torch.randn(b, K, d, **kw))


@torch.no_grad()
def fit_best(Xt, y, K, inits, n_random, gen):
    """Fit from the given starting values plus n_random random restarts, all
    in one batched call; keep the best fit for each data set."""
    b, n, d = Xt.shape
    starts = list(inits) + [random_init(y, K, d, gen) for _ in range(n_random)]
    S = len(starts)
    t0 = time.time()
    th, rss = lm_fit(Xt.repeat(S, 1, 1), y.repeat(S, 1), K, torch.cat(starts, 0))
    th = th.view(S, b, -1)
    rss = rss.view(S, b)
    best = rss.argmin(0)
    ar = torch.arange(b, device=Xt.device)
    win = torch.bincount(best, minlength=S).tolist()
    log(1, f"    fit K={K}: {b} data sets x {S} starts, RSS/n {_q(rss[best, ar] / n)}, "
           f"best start counts {win}, {time.time() - t0:.1f}s")
    return th[best, ar], rss[best, ar]


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------
def make_grid(d, M, scale, seed, device, dtype):
    """M candidate weight vectors in [-scale, scale]^d, identical across
    batches and devices for a given seed."""
    g = torch.Generator().manual_seed(seed)
    G = (torch.rand(M, d, generator=g, dtype=torch.float64) * 2 - 1) * scale
    return G.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Sup-score growth test (and the max-correlation trigger, same matrices)
# ---------------------------------------------------------------------------
@torch.no_grad()
def growth_stats(theta, Xt, y, K, Gamma, B=499, gen=None, trim=1e-2,
                 chunk=64, robust=True):
    """Batched sup-LM test of H0: the K-unit network is sufficient.

    Returns tensors of length b: T, p_boot, p_bonf, p_single, maxcorr, beta,
    and gamma (b x d), the maximising candidate."""
    b, n, d = Xt.shape
    ar = torch.arange(b, device=Xt.device)
    r = y - predict(theta, Xt, K)
    J = jacobian(theta, Xt, K)
    Q, _ = torch.linalg.qr(J)                                    # b x n x q
    H = torch.tanh(Xt @ Gamma.T)                                 # b x n x M
    Hm = H - Q @ (Q.transpose(1, 2) @ H)
    hh = (Hm * Hm).sum(1)                                        # b x M
    keep = hh / (H * H).sum(1).clamp_min(TINY) > trim
    S = torch.bmm(r[:, None, :], Hm).squeeze(1)                  # b x M
    if robust:
        V = torch.bmm((r * r)[:, None, :], Hm * Hm).squeeze(1)
    else:
        V = (r * r).mean(1, keepdim=True) * hh
    V = V.clamp_min(TINY)
    lm = torch.where(keep, S * S / V, torch.zeros_like(S))
    T, j = lm.max(1)

    cnt = torch.zeros(b, device=Xt.device, dtype=torch.long)
    for s0 in range(0, B, chunk):
        c = min(chunk, B - s0)
        Xi = torch.randn(b, n, c, generator=gen, dtype=Xt.dtype, device=Xt.device)
        Ss = torch.bmm((Xi * r[:, :, None]).transpose(1, 2), Hm)   # b x c x M
        lms = torch.where(keep[:, None, :], Ss * Ss / V[:, None, :],
                          torch.zeros_like(Ss))
        cnt += (lms.amax(2) >= T[:, None]).sum(1)
    p_boot = (1 + cnt).to(Xt.dtype) / (B + 1) if B > 0 else torch.full_like(T, float("nan"))

    M_eff = keep.sum(1).to(Xt.dtype)
    p_bonf = (M_eff * chi2_sf(T, 1)).clamp(max=1.0)
    if B > 0:
        log(1, f"    sup-LM test K={K}: T {_q(T)}, kept candidates {_q(keep.sum(1).float())}, "
               f"reject@5% {float((p_boot <= 0.05).float().mean()):.2f} of {b}")
    j0 = keep.float().argmax(1)                           # first kept unit
    p_single = chi2_sf(lm[ar, j0], 1)
    rr = (r * r).sum(1, keepdim=True)
    corr = torch.where(keep, S.abs() / torch.sqrt(rr * hh).clamp_min(TINY),
                       torch.zeros_like(S))
    return dict(T=T, p_boot=p_boot, p_bonf=p_bonf, p_single=p_single,
                maxcorr=corr.amax(1), beta=S[ar, j] / hh[ar, j].clamp_min(TINY),
                gamma=Gamma[j])


# ===========================================================================
# Competing methods from the literature (batched)
# ===========================================================================
def _aux_lm_pvalue(r, Q, Z):
    """n R^2 from regressing r on Z after projecting Z on the complement of
    span(J) (homoskedastic LM in auxiliary-regression form)."""
    n, m = r.shape[1], Z.shape[2]
    Zm = Z - Q @ (Q.transpose(1, 2) @ Z)
    A = _ridge(Zm.transpose(1, 2) @ Zm, _rel_ridge(Zm.dtype))
    coef = safe_solve(A, Zm.transpose(1, 2) @ r[:, :, None])
    fitted = (Zm @ coef).squeeze(-1)
    rc = r - r.mean(1, keepdim=True)
    stat = n * (fitted * fitted).sum(1) / (rc * rc).sum(1).clamp_min(TINY)
    return chi2_sf(stat, m)


@torch.no_grad()
def lwg_pvalue(theta, Xt, y, K, gen, q=10, n_pc=2, scale=2.0):
    """White (1989) / Lee, White & Granger (1993): q random hidden units,
    their leading principal components, n R^2 against chi2."""
    b, n, d = Xt.shape
    r = y - predict(theta, Xt, K)
    Q, _ = torch.linalg.qr(jacobian(theta, Xt, K))
    G = (torch.rand(b, q, d, generator=gen, dtype=Xt.dtype, device=Xt.device) * 2 - 1) * scale
    H = torch.tanh(torch.bmm(Xt, G.transpose(1, 2)))
    Hc = H - H.mean(1, keepdim=True)
    U, s, _ = torch.linalg.svd(Hc, full_matrices=False)
    Z = U[:, :, :n_pc] * s[:, None, :n_pc]
    return _aux_lm_pvalue(r, Q, Z)


def _poly_index(p):
    pairs = [(i, j) for i in range(p) for j in range(i, p)]
    triples = [(i, j, k) for i in range(p) for j in range(i, p) for k in range(j, p)]
    return pairs, triples


@torch.no_grad()
def tlg_pvalue(theta, Xt, y, K, max_terms=None, n_pc=3):
    """Teräsvirta, Lin & Granger (1993) Taylor-expansion LM test, as used to
    add units sequentially by Medeiros, Teräsvirta & Rech (2006): quadratic
    and cubic terms of the inputs (of their leading principal components when
    there are too many terms)."""
    b, n, d = Xt.shape
    r = y - predict(theta, Xt, K)
    Q, _ = torch.linalg.qr(jacobian(theta, Xt, K))
    X = Xt[:, :, 1:]
    p = X.shape[2]
    max_terms = max(10, n // 5) if max_terms is None else max_terms
    if p * (p + 1) // 2 + p * (p + 1) * (p + 2) // 6 > max_terms:
        Xc = X - X.mean(1, keepdim=True)
        U, s, _ = torch.linalg.svd(Xc, full_matrices=False)
        X = U[:, :, :n_pc] * s[:, None, :n_pc]
        p = n_pc
    pairs, triples = _poly_index(p)
    cols = [X[:, :, i] * X[:, :, j] for i, j in pairs]
    cols += [X[:, :, i] * X[:, :, j] * X[:, :, k] for i, j, k in triples]
    return _aux_lm_pvalue(r, Q, torch.stack(cols, 2))


@torch.no_grad()
def nic_value(theta, Xt, y, K):
    """Network information criterion (Murata, Yoshizawa & Amari, 1994), squared
    loss in the Gaussian log-likelihood scale: n log(RSS/n) + 2 tr(H^{-1} G)."""
    b, n, d = Xt.shape
    r = y - predict(theta, Xt, K)
    J = jacobian(theta, Xt, K)
    s2 = (r * r).mean(1)
    Hm = J.transpose(1, 2) @ J
    G = (J * (r * r)[:, :, None]).transpose(1, 2) @ J / s2[:, None, None]
    Hm = _ridge(Hm, _rel_ridge(Hm.dtype))
    pen = torch.diagonal(safe_solve(Hm, G), dim1=1, dim2=2).sum(-1)
    return n * torch.log(s2) + 2 * pen


# ---------------------------------------------------------------------------
# Common fitted path K = 0..K_max, shared by all methods
# ---------------------------------------------------------------------------
def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def fit_path(Xt, y, K_max, Gamma, gen, n_random=3, return_times=False):
    """Fit sizes 0..K_max for every data set in the batch. Size K+1 starts from
    the size-K fit plus the one-step sup-LM unit, and from n_random random
    restarts; the best fit is kept. Returns (list of theta_K, rss b x (K_max+1))
    and, with return_times=True, the wall time (s) of fitting each size, per
    data set (batch time / b)."""
    b, n, d = Xt.shape
    theta = y.mean(1, keepdim=True)
    thetas = [theta]
    rsss = [((y - theta) ** 2).sum(1)]
    times = [0.0]
    log(1, f"  path: fitting sizes 0..{K_max} for {b} data sets (n={n}, d={d}), "
           f"1 warm start + {n_random} random restarts per size")
    for K in range(K_max):
        _sync(Xt.device)
        t0 = time.time()
        g = growth_stats(theta, Xt, y, K, Gamma, B=0, gen=gen)
        c, a, W = unpack(theta, K, d)
        init = pack(c, torch.cat([a, g["beta"][:, None]], 1),
                    torch.cat([W, g["gamma"][:, None, :]], 1))
        theta, rss = fit_best(Xt, y, K + 1, [init], n_random, gen)
        _sync(Xt.device)
        times.append((time.time() - t0) / b)
        thetas.append(theta)
        rsss.append(rss)
    if return_times:
        return thetas, torch.stack(rsss, 1), times
    return thetas, torch.stack(rsss, 1)


def first_stop(dec):
    """dec: b x K_max boolean (add a unit at size K?). Returns the first K with
    dec False, or K_max if the rule never stops."""
    b = dec.shape[0]
    stop = torch.cat([~dec, torch.ones(b, 1, dtype=torch.bool, device=dec.device)], 1)
    return stop.float().argmax(1)


CORR_TAUS = (0.05, 0.10, 0.20)      # MaxCorr thresholds; 0.10 is the default rule "corr"
ASH_TAUS = (0.005, 0.01, 0.05)      # DNC thresholds; 0.01 is the default rule "ash"


@torch.no_grad()
def select_all(thetas, rss, Xt, y, Gamma, gen, alpha=0.05, B=499,
               val=None, ash_tau=0.01, corr_tau=0.10, return_info=False):
    """Apply all decision rules to a common fitted path.
    val: optional (Xt_val, y_val, thetas_tr) for validation-based growth.
    Returns dict method -> LongTensor (b,) of selected sizes, and the b x K_max
    tensor of sup-LM bootstrap p-values. Besides the ten main rules, the
    MaxCorr and DNC triggers are also evaluated at the thresholds in CORR_TAUS
    and ASH_TAUS (keys corr_<tau>, ash_<tau>) for the sensitivity analysis.
    With return_info=True also returns the per-data-set wall time of the
    sup-LM test at each size (list, seconds)."""
    K_max = len(thetas) - 1
    b, n, d = Xt.shape
    cols = {m: [] for m in ("suplm", "bonf", "lwg", "tlg", "corr")}
    cols.update({f"corr_{t}": [] for t in CORR_TAUS})
    pv, ttest = [], []
    for K in range(K_max):
        th = thetas[K]
        _sync(Xt.device)
        t0 = time.time()
        g = growth_stats(th, Xt, y, K, Gamma, B=B, gen=gen)
        _sync(Xt.device)
        ttest.append((time.time() - t0) / b)
        pv.append(g["p_boot"])
        cols["suplm"].append(g["p_boot"] <= alpha)
        cols["bonf"].append(g["p_bonf"] <= alpha)
        cols["corr"].append(g["maxcorr"] > corr_tau)
        for t in CORR_TAUS:
            cols[f"corr_{t}"].append(g["maxcorr"] > t)
        cols["lwg"].append(lwg_pvalue(th, Xt, y, K, gen) <= alpha)
        cols["tlg"].append(tlg_pvalue(th, Xt, y, K) <= alpha)
        log(1, f"  rules at K={K}: share adding a unit -> " + ", ".join(
            f"{m} {float(cols[m][-1].float().mean()):.2f}"
            for m in ("suplm", "bonf", "lwg", "tlg", "corr")))
    sel = {m: first_stop(torch.stack(v, 1)) for m, v in cols.items()}
    drop = (rss[:, :-1] - rss[:, 1:]) / rss[:, :-1]
    sel["ash"] = first_stop(drop > ash_tau)
    for t in ASH_TAUS:
        sel[f"ash_{t}"] = first_stop(drop > t)
    q = torch.tensor([n_params(K, d) for K in range(K_max + 1)],
                     dtype=rss.dtype, device=rss.device)
    base = n * torch.log(rss / n)
    sel["aic"] = (base + 2 * q).argmin(1)
    sel["bic"] = (base + math.log(n) * q).argmin(1)
    sel["nic"] = torch.stack([nic_value(thetas[K], Xt, y, K)
                              for K in range(K_max + 1)], 1).argmin(1)
    if val is not None:
        Xv, yv, th_tr = val
        mse = torch.stack([((predict(th_tr[K], Xv, K) - yv) ** 2).mean(1)
                           for K in range(K_max + 1)], 1)
        sel["valgrow"] = first_stop(mse[:, 1:] < mse[:, :-1])
    for m, k in sel.items():
        if "_" in m:
            continue
        counts = torch.bincount(k, minlength=K_max + 1).tolist()
        log(1, f"  selected sizes {m:8s}: counts for K=0..{K_max} {counts}")
    if return_info:
        return sel, torch.stack(pv, 1), ttest
    return sel, torch.stack(pv, 1)


@torch.no_grad()
def path_mse(thetas, Xte, yte):
    """b x (K_max+1) test MSE of every fitted size."""
    return torch.stack([((predict(th, Xte, K) - yte) ** 2).mean(1)
                        for K, th in enumerate(thetas)], 1)
