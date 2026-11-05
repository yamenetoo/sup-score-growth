"""
High-level interface for one data set.

    from supgrow import grow, sup_score_test

    res = grow(X, y, K_max=8)          # forward growth with the sup-score test
    print(res.summary())
    y_hat = res.predict(X_new)

    out = sup_score_test(X, y, K=2)    # one test: is a 2-unit network enough?

The functions wrap the batched routines of ``supgrow.growth_test`` with a
batch of size one. Inputs are NumPy arrays or tensors; by default X and y are
standardised before fitting and predictions are returned on the original
scale.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import torch

from supgrow import growth_test as gt


def _device(device):
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def default_scale(p):
    """Half-width of the candidate grid used in the paper: s = 3 sqrt(3 / (p + 1)),
    which keeps the spread of the candidate pre-activations the same across p."""
    return 3.0 * math.sqrt(3.0 / (p + 1))


class _Scaler:
    def __init__(self, X, y, standardize):
        self.on = standardize
        if standardize:
            self.mx, self.sx = X.mean(0), X.std(0).clip(min=1e-12)
            self.my, self.sy = float(y.mean()), float(y.std()) or 1.0
        else:
            self.mx, self.sx = np.zeros(X.shape[1]), np.ones(X.shape[1])
            self.my, self.sy = 0.0, 1.0

    def x(self, X):
        return (X - self.mx) / self.sx

    def y(self, y):
        return (y - self.my) / self.sy

    def y_back(self, f):
        return f * self.sy + self.my


def _as_numpy(X, y=None):
    X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if y is None:
        return X
    y = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y, dtype=float)
    return X, y.reshape(-1)


@dataclass
class GrowthResult:
    """Outcome of :func:`grow`.

    K        selected number of hidden units
    theta    parameters of the selected network, [c, a_1..a_K, vec(W)] with
             W of shape (K, p + 1) (bias first), on the standardised scale
    history  one dict per test: K, T, p_value, decision ('add' or 'stop')
    rss      residual sum of squares of each fitted size 0..K (standardised scale)
    """
    K: int
    theta: torch.Tensor
    history: list = field(default_factory=list)
    rss: list = field(default_factory=list)
    alpha: float = 0.05
    _scaler: object = None
    _dtype: object = None
    _device: object = None

    def predict(self, X):
        """Predictions of the selected network on the original scale of y."""
        X = _as_numpy(X)
        Xt = gt.add_intercept(torch.as_tensor(self._scaler.x(X), dtype=self._dtype,
                                              device=self._device))[None]
        f = gt.predict(self.theta, Xt, self.K)[0].cpu().numpy()
        return self._scaler.y_back(f)

    def summary(self):
        lines = [f"Selected K = {self.K} hidden units (alpha = {self.alpha})",
                 f"{'K':>3s} {'T_n':>10s} {'p-value':>9s}  decision"]
        for h in self.history:
            lines.append(f"{h['K']:>3d} {h['T']:>10.3f} {h['p_value']:>9.4f}  {h['decision']}")
        return "\n".join(lines)

    def __repr__(self):
        return f"GrowthResult(K={self.K}, tests={len(self.history)})"


def _prepare(X, y, standardize, device, fp64, M, scale, seed):
    X, y = _as_numpy(X, y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} values")
    dev = _device(device)
    dtype = gt.pick_dtype(dev, fp64)
    sc = _Scaler(X, y, standardize)
    Xt = gt.add_intercept(torch.as_tensor(sc.x(X), dtype=dtype, device=dev))[None]
    yt = torch.as_tensor(sc.y(y), dtype=dtype, device=dev)[None]
    p = X.shape[1]
    Gamma = gt.make_grid(p + 1, M, default_scale(p) if scale is None else scale, seed, dev, dtype)
    gen = torch.Generator(device=dev).manual_seed(seed)
    return sc, Xt, yt, Gamma, gen, dev, dtype


def sup_score_test(X, y, K, theta=None, M=500, B=499, scale=None, n_random=10, seed=0,
                   standardize=True, robust=True, device="auto", fp64=None, verbose=0):
    """Sup-score test of H0: a network with K tanh hidden units is sufficient.

    If ``theta`` is None, the K-unit network is fitted by Levenberg--Marquardt
    from ``n_random`` random starts (and the warm-started path for K > 0).
    Returns a dict with T (statistic), p_value (multiplier bootstrap),
    p_bonferroni, gamma (maximising candidate weights) and theta.
    """
    gt.set_logger(print, verbose)
    sc, Xt, yt, Gamma, gen, dev, dtype = _prepare(X, y, standardize, device, fp64, M, scale, seed)
    if theta is None:
        if K == 0:
            theta = yt.mean(1, keepdim=True)
        else:
            thetas, _ = gt.fit_path(Xt, yt, K, Gamma, gen, n_random=n_random)
            theta = thetas[K]
    g = gt.growth_stats(theta, Xt, yt, K, Gamma, B=B, gen=gen, robust=robust)
    return dict(K=K, T=float(g["T"][0]), p_value=float(g["p_boot"][0]),
                p_bonferroni=float(g["p_bonf"][0]), gamma=g["gamma"][0].cpu(), theta=theta)


def grow(X, y, K_max=8, alpha=0.05, M=500, B=499, scale=None, n_random=10, seed=0,
         standardize=True, robust=True, device="auto", fp64=None, verbose=0):
    """Forward growth of a single-hidden-layer tanh network (Algorithm 1).

    Starting from K = 0, test H0: K units are sufficient; if the bootstrap
    p-value is at most ``alpha``, add a unit (one-step initialisation from
    the maximising candidate plus ``n_random`` random restarts, best fit kept)
    and test again; stop at the first non-rejection or at ``K_max``.
    """
    gt.set_logger(print, verbose)
    sc, Xt, yt, Gamma, gen, dev, dtype = _prepare(X, y, standardize, device, fp64, M, scale, seed)
    d = Xt.shape[2]
    theta = yt.mean(1, keepdim=True)
    rss = [float(((yt - theta) ** 2).sum())]
    history, K = [], 0
    while True:
        g = gt.growth_stats(theta, Xt, yt, K, Gamma, B=B, gen=gen, robust=robust)
        T, pv = float(g["T"][0]), float(g["p_boot"][0])
        add = pv <= alpha and K < K_max
        history.append(dict(K=K, T=T, p_value=pv,
                            decision="add" if add else ("stop" if pv > alpha else "stop (K_max)")))
        if verbose:
            print(f"K={K}: T={T:.3f}, p={pv:.4f} -> {history[-1]['decision']}")
        if not add:
            break
        c, a, W = gt.unpack(theta, K, d)
        init = gt.pack(c, torch.cat([a, g["beta"][:, None]], 1),
                       torch.cat([W, g["gamma"][:, None, :]], 1))
        theta, r = gt.fit_best(Xt, yt, K + 1, [init], n_random, gen)
        rss.append(float(r[0]))
        K += 1
    return GrowthResult(K=K, theta=theta, history=history, rss=rss, alpha=alpha,
                        _scaler=sc, _dtype=dtype, _device=dev)
