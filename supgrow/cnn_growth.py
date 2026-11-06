"""
Exploratory experiment: statistical growth of a convolutional network.

A small tanh CNN classifies images (MNIST, Fashion-MNIST, CIFAR-10):
    x -> conv1 (C1 filters, 3x3) -> tanh -> 2x2 average pool
      -> conv2 (C2 filters, 3x3) -> tanh -> average pool to 3x3 -> linear (10 classes)
(The 3x3 pooled grid keeps coarse spatial information; a global average pool
loses it and leaves the network far too weak, so that every rule grows to the
maximal size.)
conv1 can grow from 2 to Cmax1 filters (default 48) and conv2 from 2 to Cmax2
(default 32). Adding a filter to a
layer is tested with the vector sup-score statistic of the paper, adapted to
the multinomial (cross-entropy) likelihood:

* candidate filter in conv2 with kernel gamma: its feature map, pooled to
  3x3, g_i(gamma) in R^9, feeds the classifier through new weights beta in
  R^{K x 9}; per-image score w_i = (y_i - p_i)[:K-1] (x) g~_i(gamma)  (dim 9(K-1))
* candidate filter in conv1: its pooled feature map h_i(gamma) feeds conv2
  through a new kernel slice beta in R^{C2 x 3 x 3}; per-image score
  w_i = d loss_i / d beta at beta = 0  (error at conv2 times the map)  (dim C2*9)

g~ / h~ are the candidate after projection on the existing channels the new
weights act alongside (constant + pooled conv2 channels, or constant + pooled
conv1 maps): the partial projection of the paper, since the full Jacobian of a
CNN is too large. The statistic is max_gamma S' V^{-1} S with the robust V,
calibrated by the multiplier bootstrap, computed on a subsample of the
training images. Training uses Adam, as usual for CNNs.

Rules compared (all start at (2, 2) filters and can reach (Cmax1, Cmax2)):
  suplm        Bonferroni over eligible layers; add a filter to the layer with the
               largest T / critical value if it exceeds 1, otherwise stop
  gradmax      add to the layer with the largest normalised score norm if > tau
  schedule     add a filter after every training round, alternating layers
               (random new filter)
  gradmax_sched  GradMax-style growth: the same fixed schedule up to the
               maximal widths, but each new filter is the candidate with the
               largest gradient norm of its (zero) outgoing weights
  suplm_refit  the widths chosen by suplm, trained from scratch for the same
               number of epochs as the suplm run (separates the effect of the
               architecture from the effect of growing / longer training)
  fixed        no growth, widths (2,2), (8,8), (16,16), (32,32), (Cmax1, Cmax2),
               trained for the same number of epochs as the suplm run of the
               same seed (--fixed-epochs match, default) or for a fixed number

Usage:
  python -m supgrow.cnn_growth --quick               # small check
  python -m supgrow.cnn_growth --data mnist fashion --seeds 5
  python -m supgrow.cnn_growth --preset cifar        # CIFAR-10 (E9)
Parallel: every (data set, seed, rule) and every fixed-width fit is a job; jobs run
in worker processes, --per-gpu per GPU (default 2) on all visible GPUs, or in one
process when there is no GPU (--workers k for k CPU processes).
Results: results/<name>_raw.csv, results/<name>.csv, results/<name>_log.txt with
<name> = cnn (default) or cifar (--preset cifar), set by --out-name.
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "results"
LOGF = None
TINY = 1e-30


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOGF is not None:
        LOGF.write(line + "\n")
        LOGF.flush()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(name, n_train, device, seed=0):
    """Return standardised tensors Xtr (n,C,H,W), ytr, Xte, yte.
    MNIST / Fashion-MNIST: C=1, 28x28. CIFAR-10: C=3, 32x32 (per-channel scaling)."""
    try:
        import torchvision
        ds = {"mnist": torchvision.datasets.MNIST,
              "fashion": torchvision.datasets.FashionMNIST,
              "cifar10": torchvision.datasets.CIFAR10}[name]
        tr = ds(root="data", train=True, download=True)
        te = ds(root="data", train=False, download=True)
        if name == "cifar10":
            Xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).float() / 255.0
            Xte = torch.tensor(te.data).permute(0, 3, 1, 2).float() / 255.0
            ytr, yte = torch.tensor(tr.targets).long(), torch.tensor(te.targets).long()
        else:
            Xtr, ytr = tr.data.float()[:, None] / 255.0, tr.targets.long()
            Xte, yte = te.data.float()[:, None] / 255.0, te.targets.long()
    except Exception as e:                                   # no torchvision: OpenML
        log(f"torchvision not usable ({type(e).__name__}); loading {name} from OpenML")
        from sklearn.datasets import fetch_openml
        oml = {"mnist": "mnist_784", "fashion": "Fashion-MNIST", "cifar10": "CIFAR_10"}[name]
        X, y = fetch_openml(oml, version=1, return_X_y=True, as_frame=False, parser="auto")
        shape = (-1, 3, 32, 32) if name == "cifar10" else (-1, 1, 28, 28)
        X = torch.tensor(np.asarray(X, dtype=np.float32)).reshape(shape) / 255.0
        y = torch.tensor(np.asarray(y).astype(int), dtype=torch.long)
        Xtr, ytr, Xte, yte = X[:-10000], y[:-10000], X[-10000:], y[-10000:]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Xtr.shape[0], generator=g)[:n_train]
    Xtr, ytr = Xtr[idx], ytr[idx]
    mu = Xtr.mean((0, 2, 3), keepdim=True)
    sd = Xtr.std((0, 2, 3), keepdim=True)
    f = lambda A: ((A - mu) / sd).to(device)
    return f(Xtr), ytr.to(device), f(Xte), yte.to(device)


# ---------------------------------------------------------------------------
# Growable CNN
# ---------------------------------------------------------------------------
POOL = 3          # conv2 maps are average-pooled to POOL x POOL before the classifier
NPOS = POOL * POOL


class GrowCNN(nn.Module):
    def __init__(self, Cmax=(16, 16), init=(2, 2), n_classes=10, in_ch=1):
        super().__init__()
        self.Cmax, self.in_ch = Cmax, in_ch
        self.conv1 = nn.Conv2d(in_ch, Cmax[0], 3, padding=1)
        self.conv2 = nn.Conv2d(Cmax[0], Cmax[1], 3, padding=1)
        self.fc = nn.Linear(Cmax[1] * NPOS, n_classes)
        m1 = torch.zeros(Cmax[0]); m1[:init[0]] = 1
        m2 = torch.zeros(Cmax[1]); m2[:init[1]] = 1
        self.register_buffer("m1", m1)
        self.register_buffer("m2", m2)
        self.zero_inactive()

    @torch.no_grad()
    def zero_inactive(self):
        a1, a2 = self.m1.bool(), self.m2.bool()
        self.conv1.weight[~a1] = 0; self.conv1.bias[~a1] = 0
        self.conv2.weight[~a2] = 0; self.conv2.bias[~a2] = 0
        self.conv2.weight[:, ~a1] = 0
        self.fc.weight[:, ~a2.repeat_interleave(NPOS)] = 0

    def mask_grads(self):
        m1, m2 = self.m1, self.m2
        self.conv1.weight.grad *= m1[:, None, None, None]
        self.conv1.bias.grad *= m1
        self.conv2.weight.grad *= m2[:, None, None, None] * m1[None, :, None, None]
        self.conv2.bias.grad *= m2
        self.fc.weight.grad *= m2.repeat_interleave(NPOS)[None, :]

    def forward(self, x, keep=False):
        a1 = self.conv1(x)
        z1 = torch.tanh(a1) * self.m1[None, :, None, None]
        p1 = F.avg_pool2d(z1, 2)
        a2 = self.conv2(p1)
        z2 = torch.tanh(a2) * self.m2[None, :, None, None]
        g = F.adaptive_avg_pool2d(z2, POOL).flatten(1)               # n x (C2 * 9)
        logits = self.fc(g)
        if keep:
            return logits, dict(p1=p1, a2=a2, z2=z2, g=g)
        return logits

    def widths(self):
        return int(self.m1.sum()), int(self.m2.sum())

    def n_params(self):
        c1, c2 = self.widths()
        return c1 * (9 * self.in_ch + 1) + c2 * (9 * c1 + 1) + self.fc.out_features * (c2 * NPOS + 1)


def train(model, X, y, epochs, lr=2e-3, bs=128, gen=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen, device="cpu").to(X.device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            model.mask_grads()
            opt.step()
    model.zero_inactive()


@torch.no_grad()
def evaluate(model, X, y, bs=2000):
    model.eval()
    nll, acc = 0.0, 0.0
    for s in range(0, X.shape[0], bs):
        out = model(X[s:s + bs])
        nll += F.cross_entropy(out, y[s:s + bs], reduction="sum").item()
        acc += (out.argmax(1) == y[s:s + bs]).float().sum().item()
    return nll / X.shape[0], acc / X.shape[0]


# ---------------------------------------------------------------------------
# Generic vector sup-score test from per-observation scores
# ---------------------------------------------------------------------------
@torch.no_grad()
def vector_sup_test(Wobs, keep, B, gen, ridge=1e-6, chunk=16):
    """Wobs: n x M x m per-observation score vectors (one per candidate);
    keep: M bool. Returns p-value, T, argmax index, normalised score norms."""
    n, M, m = Wobs.shape
    S = Wobs.sum(0)                                               # M x m
    V = torch.einsum("nkm,nkp->kmp", Wobs, Wobs)                  # M x m x m
    sc = torch.diagonal(V, dim1=1, dim2=2).mean(-1).clamp_min(TINY)
    eye = torch.eye(m, dtype=Wobs.dtype, device=Wobs.device)
    dead = torch.diagonal(V, dim1=1, dim2=2) <= 1e-12 * sc[:, None]   # unused dims
    V = V + dead.to(Wobs.dtype)[:, :, None] * eye * sc[:, None, None] + ridge * sc[:, None, None] * eye
    Lc, info = torch.linalg.cholesky_ex(V)
    ok = keep & (info == 0)
    Linv = torch.linalg.solve_triangular(Lc, eye.expand_as(Lc), upper=False)
    Sw = torch.einsum("kmp,kp->km", Linv, S)
    lm = torch.where(ok, (Sw * Sw).sum(-1), torch.zeros(M, device=Wobs.device, dtype=Wobs.dtype))
    T, j = lm.max(0)
    Ww = torch.einsum("kmp,nkp->nkm", Linv, Wobs)
    tstar = []
    for s0 in range(0, B, chunk):
        c = min(chunk, B - s0)
        xi = torch.randn(n, c, generator=gen, dtype=Wobs.dtype).to(Wobs.device)
        Ss = torch.einsum("nc,nkm->ckm", xi, Ww)
        lms = torch.where(ok[None], (Ss * Ss).sum(-1), torch.zeros_like(Ss[..., 0]))
        tstar.append(lms.amax(1))
    tstar = torch.cat(tstar).cpu().numpy() if B > 0 else np.zeros(0)
    p = (1 + int((tstar >= float(T)).sum())) / (B + 1)
    rms = torch.sqrt((Wobs * Wobs).sum((0, 2)) * n).clamp_min(TINY)
    gnorm = torch.where(ok, torch.linalg.vector_norm(S, dim=-1) / rms, torch.zeros_like(rms))
    return p, float(T), int(j), gnorm, tstar


def _residualise(Hc, Bm):
    """Least-squares residual of the columns of Hc (N x M) on Bm (N x r)."""
    G = Bm.T @ Bm
    coef = torch.linalg.pinv(G, hermitian=True) @ (Bm.T @ Hc)
    return Hc - Bm @ coef


@torch.no_grad()
def candidates(layer, M, Cin, gen, device):
    """Random candidate kernels (M x Cin x 3 x 3) and biases, scaled so that
    pre-activations have a spread comparable to the existing filters."""
    sc = 1.5 / math.sqrt(9 * Cin)
    K = (sc * torch.randn(M, Cin, 3, 3, generator=gen)).to(device)    # CPU generator, then move
    b = (0.5 * torch.randn(M, generator=gen)).to(device)
    return K, b


def layer_test(model, X, y, layer, M, B, gen, trim=1e-2, chunk_m=16):
    """Sup-score test for adding a filter to conv1 (layer 0) or conv2 (layer 1)."""
    model.eval()
    n = X.shape[0]
    K = model.fc.out_features
    a1m, a2m = model.m1.bool(), model.m2.bool()
    with torch.enable_grad():
        Xg = X
        logits, h = model(Xg, keep=True)
        lp = F.log_softmax(logits, 1)
        loss = F.nll_loss(lp, y, reduction="sum")
        a2 = h["a2"]
        d2 = torch.autograd.grad(loss, a2)[0] if layer == 0 else None   # d loss_i / d a2
    p1, z2 = h["p1"].detach(), h["z2"].detach()
    P = lp.exp().detach()
    R = (P - F.one_hot(y, K).to(P.dtype))[:, :K - 1]                  # n x (K-1)
    if layer == 1:
        Kc, bc = candidates(1, M, model.Cmax[0], gen, X.device)
        Kc = Kc * a1m[None, :, None, None]                            # only active inputs
        hmap = torch.tanh(F.conv2d(p1, Kc, bc, padding=1))            # n x M x 14 x 14
        g = F.adaptive_avg_pool2d(hmap, POOL).flatten(2)              # n x M x 9
        zp = F.adaptive_avg_pool2d(z2, POOL)[:, a2m].flatten(1)       # existing classifier inputs
        Bm = torch.cat([torch.ones(n, 1, device=X.device), zp], 1)
        gflat = g.reshape(n, M * NPOS)
        gt_ = _residualise(gflat, Bm).reshape(n, M, NPOS)
        keep = (gt_ * gt_).sum((0, 2)) / (g * g).sum((0, 2)).clamp_min(TINY) > trim
        Wobs = (R[:, None, :, None] * gt_[:, :, None, :]).reshape(n, M, (K - 1) * NPOS)
        p, T, j, gn, ts = vector_sup_test(Wobs, keep, B, gen)
        return dict(p=p, T=T, tstar=ts, kernel=Kc[j], bias=bc[j], gnorm=float(gn.max()),
                    kernel_g=Kc[int(gn.argmax())], bias_g=bc[int(gn.argmax())])
    # layer 0: candidate filter in conv1, its pooled map feeds conv2
    Kc, bc = candidates(0, M, model.in_ch, gen, X.device)
    C2 = model.Cmax[1]
    H2 = p1.shape[-1]                                                 # 14 (28x28 input) or 16 (32x32)
    HW = H2 * H2
    d2f = (d2 * a2m[None, :, None, None]).detach().flatten(2)        # n x C2 x HW
    pm = p1[:, a1m]                                                   # n x c1 x H2 x H2
    Bm = torch.cat([torch.ones(n * HW, 1, device=X.device),
                    pm.permute(0, 2, 3, 1).reshape(n * HW, -1)], 1)
    Wobs = torch.empty(n, M, C2 * 9, device=X.device)
    keep = torch.zeros(M, dtype=torch.bool, device=X.device)
    for k0 in range(0, M, chunk_m):
        k1 = min(M, k0 + chunk_m)
        hp = F.avg_pool2d(torch.tanh(F.conv2d(X, Kc[k0:k1], bc[k0:k1], padding=1)), 2)   # n x mc x H2 x H2
        mc = k1 - k0
        Hflat = hp.permute(0, 2, 3, 1).reshape(n * HW, mc)
        Ht = _residualise(Hflat, Bm)
        keep[k0:k1] = (Ht * Ht).sum(0) / (Hflat * Hflat).sum(0).clamp_min(TINY) > trim
        ht = Ht.reshape(n, H2, H2, mc).permute(0, 3, 1, 2)            # n x mc x H2 x H2
        U = F.unfold(ht.reshape(n * mc, 1, H2, H2), 3, padding=1)      # (n mc) x 9 x HW
        U = U.reshape(n, mc, 9, HW)
        Wobs[:, k0:k1] = torch.einsum("njp,nkup->nkju", d2f, U).reshape(n, mc, C2 * 9)
    p, T, j, gn, ts = vector_sup_test(Wobs, keep, B, gen)
    jg = int(gn.argmax())
    return dict(p=p, T=T, tstar=ts, kernel=Kc[j], bias=bc[j], gnorm=float(gn.max()),
                kernel_g=Kc[jg], bias_g=bc[jg])


@torch.no_grad()
def add_filter(model, layer, kernel, bias):
    """Activate the first free filter of the layer; outgoing weights start at
    zero, so the network function is unchanged."""
    if layer == 0:
        j = int((model.m1 == 0).nonzero()[0])
        model.m1[j] = 1
        model.conv1.weight[j] = kernel
        model.conv1.bias[j] = bias
        model.conv2.weight[:, j] = 0
    else:
        j = int((model.m2 == 0).nonzero()[0])
        model.m2[j] = 1
        model.conv2.weight[j] = kernel
        model.conv2.bias[j] = bias
        model.fc.weight[:, j * NPOS:(j + 1) * NPOS] = 0


def grow(rule, Xtr, ytr, args, gen, tag=""):
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    CM = (args.cmax1, args.cmax2)
    model = GrowCNN(Cmax=CM, init=(2, 2), in_ch=Xtr.shape[1]).to(Xtr.device)
    train(model, Xtr, ytr, args.epochs0, gen=gen)
    epochs = args.epochs0
    t_test = 0.0
    n_s = min(args.n_test, Xtr.shape[0])
    t0 = time.time()
    for step in range(CM[0] + CM[1] - 4):
        w = model.widths()
        elig = [w[0] < CM[0], w[1] < CM[1]]
        if not any(elig):
            break
        if rule == "schedule":
            layer = step % 2 if elig[step % 2] else elig.index(True)
            Kc, bc = candidates(layer, 1, model.in_ch if layer == 0 else CM[0], gen, Xtr.device)
            add_filter(model, layer, Kc[0], bc[0])
            msg = f"layer {layer + 1}"
        elif rule == "gradmax_sched":
            # GradMax-style: fixed schedule to the maximal widths; the new filter is the
            # candidate whose zero outgoing weights receive the largest gradient
            layer = step % 2 if elig[step % 2] else elig.index(True)
            sub = torch.randperm(Xtr.shape[0], generator=gen, device="cpu")[:n_s].to(Xtr.device)
            t1 = time.time()
            r = layer_test(model, Xtr[sub], ytr[sub], layer, args.M, 0, gen)
            t_test += time.time() - t1
            add_filter(model, layer, r["kernel_g"], r["bias_g"])
            msg = f"layer {layer + 1}; |S|/rms={r['gnorm']:.3f}"
        else:
            sub = torch.randperm(Xtr.shape[0], generator=gen, device="cpu")[:n_s].to(Xtr.device)
            t1 = time.time()
            res = [layer_test(model, Xtr[sub], ytr[sub], l, args.M, args.B if rule == "suplm" else 0, gen)
                   if elig[l] else None for l in range(2)]
            t_test += time.time() - t1
            info = " | ".join(f"conv{l + 1}: p={r['p']:.3f} T={r['T']:.1f} |S|/rms={r['gnorm']:.3f}"
                              for l, r in enumerate(res) if r is not None)
            if rule == "suplm":
                # Bonferroni across eligible layers: layer l rejects if T_l exceeds the
                # (1 - alpha/L_e) bootstrap quantile c_l. Among rejecting layers the unit
                # goes where T_l / c_l is largest. (The p-values cannot be compared when
                # both sit at the bootstrap floor 1/(B+1), and T itself cannot be
                # compared across layers because the score dimensions differ.)
                Le = sum(r is not None for r in res)
                ratio = {}
                for l, r in enumerate(res):
                    if r is not None:
                        c = float(np.quantile(r["tstar"], 1 - args.alpha / Le))
                        ratio[l] = r["T"] / max(c, TINY)
                info += " | T/crit: " + ", ".join(f"conv{l + 1} {v:.2f}" for l, v in ratio.items())
                layer = max(ratio, key=ratio.get)
                if ratio[layer] <= 1.0:
                    log(f"  {tag} [{rule}] step {step}: widths {w}; {info} -> stop")
                    break
                add_filter(model, layer, res[layer]["kernel"], res[layer]["bias"])
            else:
                cands = [(r["gnorm"], l) for l, r in enumerate(res) if r is not None]
                gmax, layer = max(cands)
                if gmax <= args.tau:
                    log(f"  {tag} [{rule}] step {step}: widths {w}; {info} -> stop")
                    break
                add_filter(model, layer, res[layer]["kernel_g"], res[layer]["bias_g"])
            msg = f"layer {layer + 1}; {info}"
        train(model, Xtr, ytr, args.epochs1, gen=gen)
        epochs += args.epochs1
        log(f"  {tag} [{rule}] step {step}: added a filter to {msg} -> widths {model.widths()}")
    w = model.widths()
    if w[0] >= CM[0] or w[1] >= CM[1]:
        log(f"  {tag} [{rule}] note: widths {w} reached the cap {CM} in "
            + " and ".join(f"conv{l + 1}" for l in range(2) if w[l] >= CM[l]))
    return model, time.time() - t0, epochs, t_test


def fit_fixed(widths, epochs, Xtr, ytr, Xte, yte, args, seed, dev):
    """Train a network of fixed widths from scratch for a given number of epochs."""
    gen = torch.Generator(device="cpu").manual_seed(1000 * seed + 11)
    torch.manual_seed(1000 * seed + 11)
    model = GrowCNN(Cmax=(args.cmax1, args.cmax2), init=widths, in_ch=Xtr.shape[1]).to(dev)
    t0 = time.time()
    train(model, Xtr, ytr, epochs, gen=gen)
    nll, acc = evaluate(model, Xte, yte)
    return model, nll, acc, time.time() - t0


# ---------------------------------------------------------------------------
# Jobs. Stage 1: one growth rule on one (data set, seed). Stage 2: SupLM refit
# and fixed widths, which need the epochs of the SupLM run of the same seed.
# Every job is seeded by (seed, rule), so results do not depend on how the
# jobs are spread over processes and GPUs.
# ---------------------------------------------------------------------------
DEV = None          # device of this process (set by _worker_init or run)
_DATA = {}          # one cached (data set, seed) per process


def _get_data(name, args, seed):
    key = (name, seed, args.n_train)
    if key not in _DATA:
        _DATA.clear()                                            # keep one data set in memory
        if DEV is not None and DEV.type == "cuda":
            torch.cuda.empty_cache()
        _DATA[key] = load_data(name, args.n_train, DEV, seed)
    return _DATA[key]


def job_grow(name, seed, rule, args):
    Xtr, ytr, Xte, yte = _get_data(name, args, seed)
    gen = torch.Generator(device="cpu").manual_seed(1000 * seed + 7)
    tag = f"{name}/s{seed}"
    log(f"  {tag} START {rule} on {DEV}")
    model, tg, ep, tt = grow(rule, Xtr, ytr, args, gen, tag)
    nll, acc = evaluate(model, Xte, yte)
    c1, c2 = model.widths()
    log(f"  {tag} RESULT {rule}: widths {model.widths()}, params {model.n_params()}, "
        f"test acc {acc:.4f}, test NLL {nll:.4f}, {ep} epochs, growth time {tg:.0f}s "
        f"(tests {tt:.0f}s = {100 * tt / max(tg, TINY):.0f}%)")
    return dict(data=name, seed=seed, rule=rule, c1=c1, c2=c2, params=model.n_params(),
                epochs=ep, cap1=int(c1 >= args.cmax1), cap2=int(c2 >= args.cmax2),
                test_nll=nll, test_acc=acc, grow_time=tg, test_time=tt,
                test_share=tt / max(tg, TINY))


def job_fixed(name, seed, label, widths, epochs, args):
    Xtr, ytr, Xte, yte = _get_data(name, args, seed)
    model, nll, acc, tf = fit_fixed(tuple(widths), epochs, Xtr, ytr, Xte, yte, args, seed, DEV)
    log(f"  {name}/s{seed} RESULT {label} {tuple(widths)}: params {model.n_params()}, "
        f"test acc {acc:.4f}, test NLL {nll:.4f} ({epochs} epochs from scratch, {tf:.0f}s, {DEV})")
    capped = label == "suplm_refit"
    return dict(data=name, seed=seed, rule=label, c1=widths[0], c2=widths[1], params=model.n_params(),
                epochs=epochs, cap1=int(capped and widths[0] >= args.cmax1),
                cap2=int(capped and widths[1] >= args.cmax2), test_nll=nll, test_acc=acc,
                grow_time=np.nan, test_time=np.nan, test_share=np.nan)


def _run_job(job):
    kind, kw = job
    t0 = time.time()
    row = job_grow(**kw) if kind == "grow" else job_fixed(**kw)
    return job, row, time.time() - t0


def _stage2_jobs(name, seed, suplm_row, args):
    """Refit of the SupLM widths and the fixed widths, with the SupLM epochs."""
    w_s, ep_s = (int(suplm_row["c1"]), int(suplm_row["c2"])), int(suplm_row["epochs"])
    ep_f = ep_s if args.fixed_epochs == "match" else int(args.fixed_epochs)
    jobs = [("fixed", dict(name=name, seed=seed, label="suplm_refit", widths=w_s, epochs=ep_s, args=args))]
    cand = ((2, 2), (8, 8), (16, 16), (32, 32), (64, 64), (args.cmax1, args.cmax2))
    for w in sorted({w for w in cand if w[0] <= args.cmax1 and w[1] <= args.cmax2}):
        jobs.append(("fixed", dict(name=name, seed=seed, label=f"fixed{w[0]}x{w[1]}", widths=w,
                                   epochs=ep_f, args=args)))
    return jobs


def _worker_init(devices, counter, log_path, threads):
    """Runs once in every worker process: pick a device, open the shared log."""
    global DEV, LOGF
    with counter.get_lock():
        k = counter.value
        counter.value += 1
    DEV = torch.device(devices[k % len(devices)])
    if DEV.type == "cuda":
        torch.cuda.set_device(DEV)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(threads)
    LOGF = open(log_path, "a")


def _devices(args):
    """List of device names, one entry per worker slot."""
    if args.device != "cpu" and torch.cuda.is_available():
        gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return [g for g in gpus for _ in range(max(1, args.per_gpu))]
    return ["cpu"] * max(1, args.workers or 1)


def run(args):
    global LOGF, DEV
    os.makedirs(OUT, exist_ok=True)
    name_out = args.out_name
    log_path = f"{OUT}/{name_out}_log.txt"
    LOGF = open(log_path, "a")
    slots = _devices(args)
    log("=" * 70)
    gpu = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) \
        if torch.cuda.is_available() else "none"
    log(f"torch {torch.__version__}; GPUs: {gpu}; worker slots: {len(slots)} {slots}; options {vars(args)}")
    for name in args.data:                                       # download once, in the main process
        try:
            load_data(name, 10, torch.device("cpu"), 0)
        except Exception as e:
            log(f"WARNING: could not load {name}: {type(e).__name__}: {e}")
    jobs = [("grow", dict(name=nm, seed=sd, rule=r, args=args))
            for nm in args.data for sd in range(args.seeds) for r in args.rules]
    n_total = len(jobs) + sum(len(_stage2_jobs(nm, sd, dict(c1=2, c2=2, epochs=1), args))
                              for nm in args.data for sd in range(args.seeds))
    rows, done, t_all = [], 0, time.time()

    def finished(job, row, dt):
        nonlocal done
        rows.append(row)
        done += 1
        el = time.time() - t_all
        log(f"[{'#' * int(30 * done / n_total):<30s}] {done}/{n_total} jobs done "
            f"({job[1]['name']}/s{job[1]['seed']} {row['rule']}: acc {row['test_acc']:.4f}, {dt:.0f}s), "
            f"elapsed {el / 60:.1f} min, ETA {el / done * (n_total - done) / 60:.1f} min")
        pd.DataFrame(rows).to_csv(f"{OUT}/{name_out}_raw.csv", index=False)
        if job[0] == "grow" and row["rule"] == "suplm":
            return _stage2_jobs(row["data"], row["seed"], row, args)
        return []

    def run_inline(pending):
        global DEV
        DEV = torch.device(slots[0])
        if DEV.type == "cuda":
            torch.backends.cudnn.benchmark = True
        while pending:
            job = pending.pop(0)
            pending += finished(*_run_job(job))

    if len(slots) == 1:
        run_inline(jobs)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        from concurrent.futures.process import BrokenProcessPool
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        ctx = mp.get_context("spawn")                            # required for CUDA and on Windows
        counter = ctx.Value("i", 0)
        threads = max(1, (os.cpu_count() or 2) // len(slots))
        pending, running, failed = list(jobs), {}, []
        try:
            with ProcessPoolExecutor(len(slots), mp_context=ctx, initializer=_worker_init,
                                     initargs=(slots, counter, log_path, threads)) as ex:
                while pending or running:
                    while pending and len(running) < 2 * len(slots):
                        job = pending.pop(0)
                        running[ex.submit(_run_job, job)] = job
                    fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                    for fu in fin:
                        job = running.pop(fu)
                        try:
                            pending += finished(*fu.result())
                        except BrokenProcessPool:
                            running[fu] = job                    # handled below: run it again inline
                            raise
                        except Exception as e:                   # e.g. CUDA out of memory in one job
                            log(f"WARNING: job {job[0]} {job[1]['name']}/s{job[1]['seed']} "
                                f"{job[1].get('rule', job[1].get('label'))} failed "
                                f"({type(e).__name__}: {e}); it will be run again in the main process")
                            failed.append(job)
        except (BrokenProcessPool, MemoryError, OSError) as e:
            log(f"WARNING: worker processes failed ({type(e).__name__}: {e}); "
                f"finishing the remaining jobs in the main process (use --per-gpu 1 to avoid this)")
            left = pending + list(running.values())
            run_inline(left)
        if failed:
            log(f"running {len(failed)} failed job(s) in the main process")
            run_inline(failed)
    df = pd.DataFrame(rows).sort_values(["data", "seed", "rule"])
    df.to_csv(f"{OUT}/{name_out}_raw.csv", index=False)
    summ = df.groupby(["data", "rule"]).agg(
        c1=("c1", "mean"), c2=("c2", "mean"), params=("params", "mean"),
        epochs=("epochs", "mean"), cap1=("cap1", "mean"), cap2=("cap2", "mean"),
        acc_mean=("test_acc", "mean"), acc_sd=("test_acc", "std"),
        nll_mean=("test_nll", "mean"), time=("grow_time", "mean"),
        test_share=("test_share", "mean"), R=("seed", "size")).reset_index()
    summ.to_csv(f"{OUT}/{name_out}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\n" + summ.to_string(index=False))
    log(f"total time {(time.time() - t_all) / 60:.1f} min")
    LOGF.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["mnist", "fashion"], choices=["mnist", "fashion", "cifar10"])
    ap.add_argument("--rules", nargs="+", default=["suplm", "gradmax", "schedule", "gradmax_sched"],
                    choices=["suplm", "gradmax", "schedule", "gradmax_sched"],
                    help="growth rules (suplm is always needed: refit and fixed widths use its epochs)")
    ap.add_argument("--out-name", default=None, help="prefix of the output files (default cnn, or cifar)")
    ap.add_argument("--preset", choices=["cifar"], default=None,
                    help="cifar: CIFAR-10, all 50000 training images, up to (64, 64) filters, 3 seeds")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=10000, help="training images used")
    ap.add_argument("--n-test", type=int, default=2000, help="training images used for each score test")
    ap.add_argument("--cmax1", type=int, default=48, help="maximal filters in conv1")
    ap.add_argument("--cmax2", type=int, default=32, help="maximal filters in conv2")
    ap.add_argument("--fixed-epochs", default="match",
                    help="'match': fixed widths get the epochs of the suplm run of the same seed; "
                         "or an integer number of epochs")
    ap.add_argument("--M", type=int, default=48, help="candidate filters per test")
    ap.add_argument("--B", type=int, default=499, help="bootstrap replications")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.10, help="GradMax-type threshold")
    ap.add_argument("--epochs0", type=int, default=5, help="epochs before growth starts")
    ap.add_argument("--epochs1", type=int, default=2, help="epochs after each added filter")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--per-gpu", type=int, default=2,
                    help="worker processes per GPU (the networks are small, so 2-3 keep a GPU busy; "
                         "use 1 if Windows reports MemoryError)")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes without a GPU (0 = 1, i.e. no parallelism)")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args(argv)
    if args.preset == "cifar":
        args.data, args.n_train, args.cmax1, args.cmax2 = ["cifar10"], 50000, 64, 64
        if args.seeds == 5:
            args.seeds = 3
        args.out_name = args.out_name or "cifar"
    if args.out_name is None:
        args.out_name = "cifar" if args.data == ["cifar10"] else "cnn"
    if "suplm" not in args.rules:
        args.rules = ["suplm"] + args.rules
    if args.quick:
        args.seeds, args.n_train, args.n_test, args.cmax1, args.cmax2 = 1, 2000, 300, 8, 6
        args.M, args.B, args.epochs0, args.epochs1 = 16, 49, 1, 1
        args.data = ["cifar10"] if args.preset == "cifar" else ["mnist"]
    run(args)


if __name__ == "__main__":
    main()
