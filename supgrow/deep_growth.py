"""
Layer-wise growth of deep (multi-hidden-layer) tanh networks: the vector
sup-score test, the "where to grow" rule, and three competing growth rules.

Everything is batched over b independent data sets (replications). Each data
set has its own network; networks of different widths share one tensor of the
maximal widths Hmax, and a boolean mask marks the active units.

Network with L hidden layers (input z_0 = x, with an intercept prepended):
    a_l = W_l [1, z_{l-1}],  z_l = tanh(a_l) * mask_l,  f = c + v' z_L

Adding a unit to hidden layer l with input weights gamma changes the
pre-activations of layer l+1 (or the output if l = L) by beta * tanh(gamma'[1, z_{l-1}]),
where beta has one entry per active unit of layer l+1 (one entry if l = L).
The score of beta at beta = 0 is the vector
    S(gamma) = n^{-1/2} sum_i delta_i * h~_i(gamma),
delta_i = residual times the derivative of f with respect to the next layer's
pre-activations (back-propagated error), and h~ the candidate unit after
projection on [1, z_l] (the directions the existing outgoing weights already
span). The statistic is LM(gamma) = S' V^{-1} S with the robust
V = n^{-1} sum_i h~_i^2 delta_i delta_i', and T = max over a grid of gamma;
the critical value comes from the multiplier bootstrap, as in the shallow case.
"""
import math
import time
import torch

from supgrow import growth_test as gt

log = gt.log
TINY = 1e-30


# ---------------------------------------------------------------------------
# Batched masked MLP
# ---------------------------------------------------------------------------
class MaskedMLP:
    def __init__(self, b, d, Hmax, init_width, gen, dtype, device):
        self.b, self.d, self.Hmax, self.L = b, d, list(Hmax), len(Hmax)
        kw = dict(dtype=dtype, device=device)
        dims = [d] + list(Hmax)
        self.W = []
        for l in range(self.L):
            w = torch.randn(b, Hmax[l], dims[l] + 1, generator=gen, **kw) / math.sqrt(dims[l] + 1)
            self.W.append(w)
        self.v = 0.1 * torch.randn(b, Hmax[-1], generator=gen, **kw)
        self.c = torch.zeros(b, **kw)
        self.mask = []
        for l in range(self.L):
            m = torch.zeros(b, Hmax[l], dtype=torch.bool, device=device)
            m[:, :init_width[l]] = True
            self.mask.append(m)
        self._zero_inactive()

    def params(self):
        return self.W + [self.v, self.c]

    def widths(self):
        return torch.stack([m.sum(1) for m in self.mask], 1)          # b x L

    def n_params(self):
        w = self.widths().double()
        ins = torch.cat([torch.full_like(w[:, :1], self.d), w[:, :-1]], 1)
        return (w * (ins + 1)).sum(1) + w[:, -1] + 1

    @torch.no_grad()
    def _zero_inactive(self):
        for l in range(self.L):
            self.W[l][~self.mask[l]] = 0.0
        self.v[~self.mask[-1]] = 0.0

    def forward(self, X, keep_pre=False):
        z, pre, zs = X, [], [X]
        for l in range(self.L):
            a = torch.bmm(gt.add_intercept(z), self.W[l].transpose(1, 2))
            pre.append(a)
            z = torch.tanh(a) * self.mask[l][:, None, :]
            zs.append(z)
        f = torch.bmm(z, self.v[:, :, None]).squeeze(-1) + self.c[:, None]
        return (f, pre, zs) if keep_pre else f

    def snapshot(self):
        return ([w.detach().clone() for w in self.W], self.v.detach().clone(),
                self.c.detach().clone(), [m.clone() for m in self.mask])

    @torch.no_grad()
    def restore(self, snap, which):
        """Restore the snapshot for the data sets where which is True."""
        W, v, c, mask = snap
        for l in range(self.L):
            self.W[l][which] = W[l][which]
            self.mask[l][which] = mask[l][which]
        self.v[which] = v[which]
        self.c[which] = c[which]

    @torch.no_grad()
    def add_unit(self, layer, gamma, beta, which):
        """Activate one new unit in `layer` for data sets in `which` (bool b).
        gamma: b x (in+1) input weights; beta: b output weight used only when
        the unit feeds the output (layer = L-1); for lower layers the outgoing
        weights start at zero, so the network function is unchanged."""
        m = self.mask[layer]
        free = (~m).float().argmax(1)                                  # first free slot
        ok = which & (~m).any(1)
        idx = torch.nonzero(ok).squeeze(1)
        j = free[idx]
        m[idx, j] = True
        self.W[layer][idx, j] = gamma[idx]
        if layer == self.L - 1:
            self.v[idx, j] = beta[idx]
        else:
            self.W[layer + 1][idx, :, j + 1] = 0.0
        return ok


def train(model, X, y, steps, lr, train_mask):
    """Full-batch Adam on the sum over data sets of the mean squared error.
    Data sets with train_mask False are not updated."""
    ps = model.params()
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ps, lr=lr)
    w = train_mask.to(X.dtype)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (((model.forward(X) - y) ** 2).mean(1) * w).sum()
        loss.backward()
        with torch.no_grad():
            for l in range(model.L):
                model.W[l].grad *= model.mask[l][:, :, None]
            model.v.grad *= model.mask[-1]
        opt.step()
    for p in ps:
        p.requires_grad_(False)
    model._zero_inactive()
    with torch.no_grad():
        return ((model.forward(X) - y) ** 2).mean(1)


# ---------------------------------------------------------------------------
# Vector sup-score test for adding a unit to one layer
# ---------------------------------------------------------------------------
def layer_grid(in_dim, M, scale, seed, device, dtype):
    return gt.make_grid(in_dim + 1, M, scale, seed, device, dtype)


def layer_test(model, X, y, layer, Gamma, B=499, gen=None, trim=1e-2,
               chunk=8, ridge=1e-6):
    """Sup-score test for adding one unit to `layer` (0-based), batched.
    Returns dict with p (bootstrap p-value), T, gamma (b x in+1), beta (b),
    gradnorm (largest normalised gradient norm, the GradMax-type trigger)."""
    L = model.L
    for p in model.params():
        p.requires_grad_(True)
    f, pre, zs = model.forward(X, keep_pre=True)
    r = (y - f).detach()
    if layer == L - 1:
        delta = r[:, :, None]                                          # b x n x 1
        nmask = torch.ones(model.b, 1, dtype=torch.bool, device=X.device)
    else:
        nxt = pre[layer + 1]
        g = torch.autograd.grad((r * f).sum(), nxt)[0]                 # r_i df_i/da_ij
        nmask = model.mask[layer + 1]
        delta = g * nmask[:, None, :]
    for p in model.params():
        p.requires_grad_(False)
    with torch.no_grad():
        zin = gt.add_intercept(zs[layer])                              # input of the layer
        H = torch.tanh(zin @ Gamma.T)                                  # b x n x M
        Bm = gt.add_intercept(zs[layer + 1])                           # [1, z_layer]
        G = Bm.transpose(1, 2) @ Bm
        Pinv = torch.linalg.pinv(G, hermitian=True)
        Hm = H - Bm @ (Pinv @ (Bm.transpose(1, 2) @ H))
        hh = (Hm * Hm).sum(1)
        keep = hh / (H * H).sum(1).clamp_min(TINY) > trim               # b x M
        S = torch.einsum("bnm,bnk->bkm", delta, Hm)                     # b x M x m
        V = torch.einsum("bnk,bnm,bnp->bkmp", Hm * Hm, delta, delta)    # b x M x m x m
        m = delta.shape[2]
        eye = torch.eye(m, dtype=X.dtype, device=X.device)
        scale = torch.diagonal(V, dim1=2, dim2=3).mean(-1).clamp_min(TINY)
        V = V + (~nmask).to(X.dtype)[:, None, :, None] * eye * scale[..., None, None] \
              + ridge * scale[..., None, None] * eye
        Lc, info = torch.linalg.cholesky_ex(V)
        Linv = torch.linalg.solve_triangular(
            Lc, eye.expand_as(Lc), upper=False)                         # b x M x m x m
        Sw = torch.einsum("bkmp,bkp->bkm", Linv, S)
        lm = torch.where(keep & (info == 0), (Sw * Sw).sum(-1), torch.zeros_like(hh))
        T, j = lm.max(1)
        cnt = torch.zeros(model.b, dtype=torch.long, device=X.device)
        n = X.shape[1]
        for s0 in range(0, B, chunk):
            c = min(chunk, B - s0)
            Xi = torch.randn(model.b, n, c, generator=gen, dtype=X.dtype, device=X.device)
            Ss = torch.einsum("bnc,bnm,bnk->bckm", Xi, delta, Hm)
            Sws = torch.einsum("bkmp,bckp->bckm", Linv, Ss)
            lms = torch.where((keep & (info == 0))[:, None, :], (Sws * Sws).sum(-1),
                              torch.zeros_like(Sws[..., 0]))
            cnt += (lms.amax(2) >= T[:, None]).sum(1)
        pval = (1 + cnt).to(X.dtype) / (B + 1)
        ar = torch.arange(model.b, device=X.device)
        dn = torch.sqrt((delta * delta).sum((1, 2)))[:, None]           # b x 1
        gn = torch.where(keep, torch.linalg.vector_norm(S, dim=-1) /
                         (dn * torch.sqrt(hh)).clamp_min(TINY), torch.zeros_like(hh))
        gnorm, jg = gn.max(1)
        onestep = lambda k: (r * Hm[ar, :, k]).sum(1) / hh[ar, k].clamp_min(TINY)
        beta, beta_g = onestep(j), onestep(jg)
    return dict(p=pval, T=T, gamma=Gamma[j], beta=beta, gradnorm=gnorm,
                gamma_g=Gamma[jg], beta_g=beta_g)


# ---------------------------------------------------------------------------
# Growth procedures
# ---------------------------------------------------------------------------
def grow(method, X, y, Hmax, grids, gen, alpha=0.05, B=499, tau=0.10,
         steps0=1500, steps1=500, lr=0.01, max_add=None, val=None,
         init_width=(1, 1)):
    """Grow networks for every data set in the batch with one of the rules
      'suplm'    : add to the layer with the smallest sup-score p-value if it is
                   <= alpha / (number of eligible layers); stop otherwise
      'gradmax'  : add to the layer with the largest normalised gradient norm
                   if it exceeds tau (GradMax-type uncalibrated trigger)
      'valgrow'  : candidate as in 'gradmax' (no threshold); keep it only if
                   the validation MSE decreases after training; stop otherwise
      'schedule' : add one unit after every training round, alternating
                   layers, until every layer is full (Net2Net/Firefly-type
                   schedule; ends at the maximal widths)
    Returns the model and a dict of per-data-set statistics."""
    b, n, d = X.shape
    L = len(Hmax)
    model = MaskedMLP(b, d, Hmax, init_width, gen, X.dtype, X.device)
    active = torch.ones(b, dtype=torch.bool, device=X.device)
    train(model, X, y, steps0, lr, active)
    max_add = max_add or sum(Hmax) - sum(init_width)
    t0 = time.time()
    tests = 0
    if val is not None:
        Xv, yv = val
        with torch.no_grad():
            best_val = ((model.forward(Xv) - yv) ** 2).mean(1)
    for step in range(max_add):
        if not bool(active.any()):
            break
        widths = model.widths()
        eligible = widths < torch.tensor(Hmax, device=X.device)[None, :]      # b x L
        if method == "schedule":
            layer_pref = torch.full((b,), step % L, device=X.device, dtype=torch.long)
            lay = torch.where(eligible[torch.arange(b, device=X.device), layer_pref], layer_pref,
                              eligible.float().argmax(1))
            add = active & eligible.any(1)
            for l in range(L):
                which = add & (lay == l)
                if bool(which.any()):
                    in_dim = d if l == 0 else Hmax[l - 1]
                    g0 = torch.randn(b, in_dim + 1, generator=gen, dtype=X.dtype,
                                     device=X.device) / math.sqrt(in_dim + 1)
                    model.add_unit(l, g0, torch.zeros(b, dtype=X.dtype, device=X.device), which)
        else:
            res = []
            for l in range(L):
                res.append(layer_test(model, X, y, l, grids[l],
                                      B=B if method == "suplm" else 0, gen=gen))
                tests += 1
            if method == "suplm":
                P = torch.stack([r_["p"] for r_ in res], 1)
                P = torch.where(eligible, P, torch.full_like(P, 2.0))
                pmin, lay = P.min(1)
                n_el = eligible.sum(1).clamp_min(1).to(X.dtype)
                add = active & (pmin <= alpha / n_el)
                key_g, key_b = "gamma", "beta"
            else:
                Gn = torch.stack([r_["gradnorm"] for r_ in res], 1)
                Gn = torch.where(eligible, Gn, torch.full_like(Gn, -1.0))
                gmax, lay = Gn.max(1)
                add = active & eligible.any(1) & ((gmax > tau) if method == "gradmax" else True)
                key_g, key_b = "gamma_g", "beta_g"
            snap = model.snapshot() if method == "valgrow" else None
            for l in range(L):
                which = add & (lay == l)
                if bool(which.any()):
                    model.add_unit(l, res[l][key_g], res[l][key_b], which)
        stopped = active & ~add
        active = active & add
        log(1, f"  [{method}] step {step}: added a unit in {int(add.sum())}/{b} data sets; "
               f"widths mean {model.widths().double().mean(0).tolist()}")
        if not bool(active.any()):
            break
        train(model, X, y, steps1, lr, active)
        if method == "valgrow":
            with torch.no_grad():
                vm = ((model.forward(Xv) - yv) ** 2).mean(1)
            worse = active & (vm >= best_val)
            model.restore(snap, worse)
            best_val = torch.where(active & ~worse, vm, best_val)
            active = active & ~worse
    return model, dict(widths=model.widths(), n_params=model.n_params(),
                       grow_time=(time.time() - t0) / b, tests=tests)


# ---------------------------------------------------------------------------
# Teacher network for the simulation
# ---------------------------------------------------------------------------
def teacher(p, widths, seed, dtype, device):
    """Two-hidden-layer tanh teacher with well-separated units: first-layer
    directions are orthonormal (scaled), second-layer weights are a fixed
    well-conditioned matrix."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=torch.float64))
    h1, h2 = widths
    W1 = torch.cat([torch.tensor([0.0, 0.5, -0.5, 0.3, -0.3][:h1], dtype=torch.float64)[:, None],
                    1.5 * Q[:, :h1].T], 1)
    W2 = torch.cat([torch.tensor([0.2, -0.2, 0.1][:h2], dtype=torch.float64)[:, None],
                    torch.tensor([[1.5, -1.0, 0.8], [-0.7, 1.4, 1.2], [1.0, 0.6, -1.3]],
                                 dtype=torch.float64)[:h2, :h1]], 1)
    v = torch.tensor([1.5, -1.2, 1.0][:h2], dtype=torch.float64)
    to = lambda t: t.to(dtype=dtype, device=device)
    return to(W1), to(W2), to(v)


def teacher_predict(T, X):
    W1, W2, v = T
    z1 = torch.tanh(gt.add_intercept(X) @ W1.T)
    z2 = torch.tanh(gt.add_intercept(z1) @ W2.T)
    return z2 @ v


# ===========================================================================
# Version 2: full projection, Levenberg--Marquardt fits with restarts
# ===========================================================================
# Parameters of every network are stored as one flat vector theta (b x q):
# for each hidden layer l the matrix W_l (Hmax_l x (in_l + 1), bias first),
# then the output weights v (Hmax_L) and the intercept c. Inactive units have
# zero activation (unit masks um[l], bool b x Hmax_l), so their parameters
# have zero Jacobian columns and do not move during fitting.

class Arch:
    def __init__(self, d, Hmax):
        self.d, self.Hmax, self.L = d, list(Hmax), len(Hmax)
        dims = [d] + list(Hmax)
        self.shapes = [(Hmax[l], dims[l] + 1) for l in range(self.L)]
        self.offsets, o = [], 0
        for h, w in self.shapes:
            self.offsets.append(o)
            o += h * w
        self.off_v = o
        self.q = o + Hmax[-1] + 1

    def W(self, th, l):
        h, w = self.shapes[l]
        o = self.offsets[l]
        return th[:, o:o + h * w].reshape(th.shape[0], h, w)

    def v(self, th):
        return th[:, self.off_v:self.off_v + self.Hmax[-1]]

    def c(self, th):
        return th[:, -1]


def forward_jac(arch, th, X, um, need_jac=True):
    """Batched forward pass and analytic Jacobian of f with respect to theta.
    Returns f (b x n), J (b x n x q) or None, zs (activations, zs[0] = X) and
    D (list over hidden layers of df/da_l, b x n x Hmax_l)."""
    b, n, _ = X.shape
    z, zs, zts = X, [X], []
    for l in range(arch.L):
        zt = gt.add_intercept(z)
        zts.append(zt)
        z = torch.tanh(torch.bmm(zt, arch.W(th, l).transpose(1, 2))) * um[l][:, None, :].to(X.dtype)
        zs.append(z)
    v = arch.v(th)
    f = torch.bmm(z, v[:, :, None]).squeeze(-1) + arch.c(th)[:, None]
    D = [None] * arch.L
    d_next = v[:, None, :].expand(b, n, -1)                    # df/dz_L
    for l in reversed(range(arch.L)):
        zl = zs[l + 1]
        D[l] = d_next * (1 - zl * zl) * um[l][:, None, :].to(X.dtype)   # df/da_l
        if l > 0:
            d_next = torch.bmm(D[l], arch.W(th, l)[:, :, 1:])
    if not need_jac:
        return f, None, zs, D
    cols = []
    for l in range(arch.L):
        cols.append((D[l][:, :, :, None] * zts[l][:, :, None, :]).reshape(b, n, -1))
    cols += [zs[-1], X.new_ones(b, n, 1)]
    return f, torch.cat(cols, 2), zs, D


def masks_from_widths(arch, widths, b, device):
    um = []
    for l in range(arch.L):
        m = torch.zeros(b, arch.Hmax[l], dtype=torch.bool, device=device)
        m[:, :widths[l]] = True
        um.append(m)
    return um


def random_theta(arch, y, gen):
    b = y.shape[0]
    kw = dict(dtype=y.dtype, device=y.device, generator=gen)
    parts = []
    for h, w in arch.shapes:
        parts.append((1.5 / math.sqrt(w)) * torch.randn(b, h * w, **kw))
    parts.append(0.5 * torch.randn(b, arch.Hmax[-1], **kw))
    parts.append(y.mean(1, keepdim=True))
    return torch.cat(parts, 1)


@torch.no_grad()
def lm_deep(arch, th0, X, y, um, max_iter=300, tol=None):
    """Batched Levenberg--Marquardt for the masked deep network."""
    th = th0.clone()
    b, n, _ = X.shape
    q = th.shape[1]
    tol = (1e-10 if th.dtype == torch.float64 else 1e-7) if tol is None else tol
    eye = torch.eye(q, dtype=th.dtype, device=th.device)
    f, J, _, _ = forward_jac(arch, th, X, um)
    r = f - y
    rss = (r * r).sum(1)
    lam = th.new_full((b,), 1e-3)
    active = torch.ones(b, dtype=torch.bool, device=th.device)
    for it in range(max_iter):
        A = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ r[:, :, None]).squeeze(-1)
        Mm = A + lam[:, None, None] * (torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2)) + 1e-6 * eye)
        step, info = torch.linalg.solve_ex(Mm, -g[:, :, None])
        cand = th + step.squeeze(-1)
        fc, Jc, _, _ = forward_jac(arch, cand, X, um)
        rc = fc - y
        rssc = (rc * rc).sum(1)
        ok = active & (info == 0) & torch.isfinite(rssc) & (rssc < rss)
        rel = (rss - rssc) / rss.clamp_min(TINY)
        th = torch.where(ok[:, None], cand, th)
        r = torch.where(ok[:, None], rc, r)
        J = torch.where(ok[:, None, None], Jc, J)
        rss = torch.where(ok, rssc, rss)
        lam = torch.where(ok, lam * 0.3, lam * 4.0).clamp(1e-12, 1e10)
        active = active & ~((ok & (rel < tol)) | (lam >= 1e10))
        if VERBOSE_DEEP() >= 2 and it % 50 == 0:
            log(2, f"      LM-deep iter {it:3d}: active {int(active.sum())}/{b}, RSS/n {gt._q(rss / n)}")
        if not bool(active.any()):
            break
    return th, rss


def VERBOSE_DEEP():
    return gt.VERBOSE


def _rep(t, S):
    return t.repeat(S, *([1] * (t.dim() - 1)))


@torch.no_grad()
def fit_deep(arch, X, y, um, inits, n_random, gen):
    """Fit from the given starts plus n_random random starts (same widths);
    keep the best fit per data set."""
    b = X.shape[0]
    starts = list(inits) + [random_theta(arch, y, gen) for _ in range(n_random)]
    S = len(starts)
    th, rss = lm_deep(arch, torch.cat(starts, 0), _rep(X, S), _rep(y, S), [_rep(m, S) for m in um])
    th, rss = th.view(S, b, -1), rss.view(S, b)
    best = rss.argmin(0)
    ar = torch.arange(b, device=X.device)
    return th[best, ar], rss[best, ar]


@torch.no_grad()
def score_test_full(arch, th, X, y, um, layer, Gamma, B=199, gen=None, trim=1e-2,
                    chunk_g=50, chunk_b=16):
    """Vector sup-score test for adding a unit to `layer`, with the candidate
    projected on the span of the FULL Jacobian (Theorem 4)."""
    b, n, _ = X.shape
    f, J, zs, D = forward_jac(arch, th, X, um)
    r = y - f
    L = arch.L
    if layer == L - 1:
        e = X.new_ones(b, n, 1)
        emask = torch.ones(b, 1, dtype=torch.bool, device=X.device)
    else:
        e = D[layer + 1]
        emask = um[layer + 1]
    m = e.shape[2]
    zin = gt.add_intercept(zs[layer])
    Hc = torch.tanh(zin @ Gamma.T)                                   # b x n x M
    M = Hc.shape[2]
    Jt = J.transpose(1, 2)
    Pinv = torch.linalg.pinv(Jt @ J, hermitian=True)
    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    WW = torch.empty(b, n, M, m, dtype=X.dtype, device=X.device)    # whitened r_i * phi~_i
    lm = torch.zeros(b, M, dtype=X.dtype, device=X.device)
    keep = torch.zeros(b, M, dtype=torch.bool, device=X.device)
    num = torch.zeros(b, M, dtype=X.dtype, device=X.device)
    beta = torch.zeros(b, M, dtype=X.dtype, device=X.device)
    rnorm = torch.sqrt((r * r).sum(1)).clamp_min(TINY)
    for k0 in range(0, M, chunk_g):
        k1 = min(M, k0 + chunk_g)
        phi = Hc[:, :, k0:k1, None] * e[:, :, None, :]               # b x n x mc x m
        mc = k1 - k0
        flat = phi.reshape(b, n, mc * m)
        pt = flat - J @ (Pinv @ (Jt @ flat))
        pt = pt.reshape(b, n, mc, m)
        ss_t = (pt * pt).sum((1, 3))
        ss = (phi * phi).sum((1, 3)).clamp_min(TINY)
        keep[:, k0:k1] = ss_t / ss > trim
        w = r[:, :, None, None] * pt
        S = w.sum(1)                                                  # b x mc x m
        V = torch.einsum("bnkm,bnkp->bkmp", w, w)
        sc = torch.diagonal(V, dim1=2, dim2=3).mean(-1).clamp_min(TINY)
        V = V + (~emask).to(X.dtype)[:, None, :, None] * eye * sc[..., None, None] \
              + 1e-6 * sc[..., None, None] * eye
        Lc, info = torch.linalg.cholesky_ex(V)
        Linv = torch.linalg.solve_triangular(Lc, eye.expand_as(Lc), upper=False)
        keep[:, k0:k1] &= info == 0
        WW[:, :, k0:k1] = torch.einsum("bkmp,bnkp->bnkm", Linv, w)
        Sw = torch.einsum("bkmp,bkp->bkm", Linv, S)
        lm[:, k0:k1] = (Sw * Sw).sum(-1)
        num[:, k0:k1] = torch.linalg.vector_norm(S, dim=-1) / (rnorm[:, None] * torch.sqrt(ss_t).clamp_min(TINY))
        if m == 1:
            beta[:, k0:k1] = S[..., 0] / ss_t.clamp_min(TINY)
    lm = torch.where(keep, lm, torch.zeros_like(lm))
    T, j = lm.max(1)
    gn = torch.where(keep, num, torch.zeros_like(num))
    gnorm, jg = gn.max(1)
    cnt = torch.zeros(b, dtype=torch.long, device=X.device)
    for s0 in range(0, B, chunk_b):
        c = min(chunk_b, B - s0)
        xi = torch.randn(b, n, c, generator=gen, dtype=X.dtype, device=X.device)
        Ss = torch.einsum("bnc,bnkm->bckm", xi, WW)
        lms = torch.where(keep[:, None, :], (Ss * Ss).sum(-1), torch.zeros_like(Ss[..., 0]))
        cnt += (lms.amax(2) >= T[:, None]).sum(1)
    p = (1 + cnt).to(X.dtype) / (B + 1) if B > 0 else torch.ones_like(T)
    ar = torch.arange(b, device=X.device)
    return dict(p=p, T=T, gamma=Gamma[j], beta=beta[ar, j], gradnorm=gnorm,
                gamma_g=Gamma[jg], beta_g=beta[ar, jg])


def _insert_unit(arch, th, um, layer, gamma, beta, which):
    """Activate the first free unit of `layer` for data sets in `which`; input
    weights gamma; outgoing weights zero (hidden) or beta (last hidden layer)."""
    th, um = th.clone(), [m.clone() for m in um]
    mk = um[layer]
    free = (~mk).float().argmax(1)
    ok = which & (~mk).any(1)
    idx = torch.nonzero(ok).squeeze(1)
    if idx.numel() == 0:
        return th, um, ok
    j = free[idx]
    mk[idx, j] = True
    h, w = arch.shapes[layer]
    o = arch.offsets[layer]
    for t, (i_, j_) in enumerate(zip(idx.tolist(), j.tolist())):
        th[i_, o + j_ * w:o + (j_ + 1) * w] = gamma[i_]
        if layer == arch.L - 1:
            th[i_, arch.off_v + j_] = beta[i_]
        else:
            h2, w2 = arch.shapes[layer + 1]
            o2 = arch.offsets[layer + 1]
            th[i_, o2 + (j_ + 1):o2 + h2 * w2:w2] = 0.0             # column j_+1 of W_{l+1}
    return th, um, ok


def n_params_active(arch, um):
    w = torch.stack([m.sum(1) for m in um], 1).double()
    ins = torch.cat([torch.full_like(w[:, :1], arch.d), w[:, :-1]], 1)
    return (w * (ins + 1)).sum(1) + w[:, -1] + 1


@torch.no_grad()
def grow_v2(method, X, y, Hmax, grids, gen, init_width=(1, 1), alpha=0.05, B=199,
            tau=0.10, n_random=4, val=None):
    """Layer-wise growth with LM fits (warm start + n_random random starts at
    every size) and the full-projection test. Methods: 'suplm', 'gradmax',
    'valgrow', 'schedule' (as in grow())."""
    b, n, d = X.shape
    arch = Arch(d, Hmax)
    L = arch.L
    um = masks_from_widths(arch, init_width, b, X.device)
    th, rss = fit_deep(arch, X, y, um, [], n_random + 1, gen)
    active = torch.ones(b, dtype=torch.bool, device=X.device)
    t0 = time.time()
    if val is not None:
        Xv, yv = val
        best_val = ((forward_jac(arch, th, Xv, um, need_jac=False)[0] - yv) ** 2).mean(1)
    Hm = torch.tensor(Hmax, device=X.device)
    for step in range(sum(Hmax) - sum(init_width)):
        if not bool(active.any()):
            break
        widths = torch.stack([m.sum(1) for m in um], 1)
        eligible = widths < Hm[None, :]
        if method == "schedule":
            pref = torch.full((b,), step % L, device=X.device, dtype=torch.long)
            lay = torch.where(eligible[torch.arange(b, device=X.device), pref], pref,
                              eligible.float().argmax(1))
            add = active & eligible.any(1)
            res = None
        else:
            res = [score_test_full(arch, th, X, y, um, l, grids[l],
                                   B=B if method == "suplm" else 0, gen=gen) for l in range(L)]
            if method == "suplm":
                P = torch.stack([r_["p"] for r_ in res], 1)
                P = torch.where(eligible, P, torch.full_like(P, 2.0))
                pmin, lay = P.min(1)
                add = active & (pmin <= alpha / eligible.sum(1).clamp_min(1).to(X.dtype))
            else:
                G = torch.stack([r_["gradnorm"] for r_ in res], 1)
                G = torch.where(eligible, G, torch.full_like(G, -1.0))
                gmax, lay = G.max(1)
                add = active & eligible.any(1)
                if method == "gradmax":
                    add = add & (gmax > tau)
        th_new, um_new = th, um
        for l in range(L):
            which = add & (lay == l)
            if not bool(which.any()):
                continue
            if res is None:
                in_dim = arch.shapes[l][1]
                g0 = torch.randn(b, in_dim, generator=gen, dtype=X.dtype, device=X.device) / math.sqrt(in_dim)
                bz = torch.zeros(b, dtype=X.dtype, device=X.device)
                th_new, um_new, _ = _insert_unit(arch, th_new, um_new, l, g0, bz, which)
            else:
                kg, kb = ("gamma", "beta") if method == "suplm" else ("gamma_g", "beta_g")
                th_new, um_new, _ = _insert_unit(arch, th_new, um_new, l, res[l][kg], res[l][kb], which)
        idx = torch.nonzero(add).squeeze(1)
        if idx.numel() > 0:
            sub_um = [m[idx] for m in um_new]
            th_fit, _ = fit_deep(arch, X[idx], y[idx], sub_um, [th_new[idx]], n_random, gen)
            th_new = th_new.clone()
            th_new[idx] = th_fit
        if method == "valgrow" and idx.numel() > 0:
            vm = ((forward_jac(arch, th_new, Xv, um_new, need_jac=False)[0] - yv) ** 2).mean(1)
            better = add & (vm < best_val)
            keep_new = better
            best_val = torch.where(better, vm, best_val)
            add = better
        else:
            keep_new = add
        th = torch.where(keep_new[:, None], th_new, th)
        um = [torch.where(keep_new[:, None], mn, mo) for mn, mo in zip(um_new, um)]
        active = active & add
        log(1, f"  [{method} v2] step {step}: added a unit in {int(add.sum())}/{b}; widths mean "
               f"{torch.stack([m.sum(1) for m in um], 1).double().mean(0).tolist()}")
    widths = torch.stack([m.sum(1) for m in um], 1)
    return arch, th, um, dict(widths=widths, n_params=n_params_active(arch, um),
                              grow_time=(time.time() - t0) / b)


@torch.no_grad()
def predict_v2(arch, th, X, um):
    return forward_jac(arch, th, X, um, need_jac=False)[0]
