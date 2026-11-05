"""
Exploratory experiment: statistical growth of a convolutional network.

A small tanh CNN classifies images (MNIST, Fashion-MNIST):
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
  suplm_refit  the widths chosen by suplm, trained from scratch for the same
               number of epochs as the suplm run (separates the effect of the
               architecture from the effect of growing / longer training)
  fixed        no growth, widths (2,2), (8,8), (16,16), (32,32), (Cmax1, Cmax2),
               trained for the same number of epochs as the suplm run of the
               same seed (--fixed-epochs match, default) or for a fixed number

Usage:
  python -m supgrow.cnn_growth --quick               # small check
  python -m supgrow.cnn_growth --data mnist fashion --seeds 5
Results: results/cnn_raw.csv, results/cnn.csv, results/cnn_log.txt
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
    """Return standardised tensors Xtr (n,1,28,28), ytr, Xte, yte."""
    X = y = None
    try:
        import torchvision
        ds = {"mnist": torchvision.datasets.MNIST,
              "fashion": torchvision.datasets.FashionMNIST}[name]
        tr = ds(root="data", train=True, download=True)
        te = ds(root="data", train=False, download=True)
        Xtr, ytr = tr.data.float() / 255.0, tr.targets.long()
        Xte, yte = te.data.float() / 255.0, te.targets.long()
    except Exception as e:                                   # no torchvision: OpenML
        log(f"torchvision not usable ({type(e).__name__}); loading {name} from OpenML")
        from sklearn.datasets import fetch_openml
        oml = {"mnist": "mnist_784", "fashion": "Fashion-MNIST"}[name]
        X, y = fetch_openml(oml, version=1, return_X_y=True, as_frame=False, parser="auto")
        X = torch.tensor(X, dtype=torch.float32).reshape(-1, 28, 28) / 255.0
        y = torch.tensor(y.astype(int), dtype=torch.long)
        Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Xtr.shape[0], generator=g)[:n_train]
    Xtr, ytr = Xtr[idx], ytr[idx]
    mu, sd = Xtr.mean(), Xtr.std()
    f = lambda A: ((A - mu) / sd)[:, None].to(device)
    return f(Xtr), ytr.to(device), f(Xte), yte.to(device)


# ---------------------------------------------------------------------------
# Growable CNN
# ---------------------------------------------------------------------------
POOL = 3          # conv2 maps are average-pooled to POOL x POOL before the classifier
NPOS = POOL * POOL


class GrowCNN(nn.Module):
    def __init__(self, Cmax=(16, 16), init=(2, 2), n_classes=10):
        super().__init__()
        self.Cmax = Cmax
        self.conv1 = nn.Conv2d(1, Cmax[0], 3, padding=1)
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
        return c1 * (9 + 1) + c2 * (9 * c1 + 1) + self.fc.out_features * (c2 * NPOS + 1)


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
    Kc, bc = candidates(0, M, 1, gen, X.device)
    C2 = model.Cmax[1]
    d2f = (d2 * a2m[None, :, None, None]).detach().flatten(2)        # n x C2 x 196
    pm = p1[:, a1m]                                                   # n x c1 x 14 x 14
    Bm = torch.cat([torch.ones(n * 196, 1, device=X.device),
                    pm.permute(0, 2, 3, 1).reshape(n * 196, -1)], 1)
    Wobs = torch.empty(n, M, C2 * 9, device=X.device)
    keep = torch.zeros(M, dtype=torch.bool, device=X.device)
    for k0 in range(0, M, chunk_m):
        k1 = min(M, k0 + chunk_m)
        hp = F.avg_pool2d(torch.tanh(F.conv2d(X, Kc[k0:k1], bc[k0:k1], padding=1)), 2)   # n x mc x 14 x 14
        mc = k1 - k0
        Hflat = hp.permute(0, 2, 3, 1).reshape(n * 196, mc)
        Ht = _residualise(Hflat, Bm)
        keep[k0:k1] = (Ht * Ht).sum(0) / (Hflat * Hflat).sum(0).clamp_min(TINY) > trim
        ht = Ht.reshape(n, 14, 14, mc).permute(0, 3, 1, 2)            # n x mc x 14 x 14
        U = F.unfold(ht.reshape(n * mc, 1, 14, 14), 3, padding=1)      # (n mc) x 9 x 196
        U = U.reshape(n, mc, 9, 196)
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
    model = GrowCNN(Cmax=CM, init=(2, 2)).to(Xtr.device)
    train(model, Xtr, ytr, args.epochs0, gen=gen)
    epochs = args.epochs0
    n_s = min(args.n_test, Xtr.shape[0])
    t0 = time.time()
    for step in range(CM[0] + CM[1] - 4):
        w = model.widths()
        elig = [w[0] < CM[0], w[1] < CM[1]]
        if not any(elig):
            break
        if rule == "schedule":
            layer = step % 2 if elig[step % 2] else elig.index(True)
            Kc, bc = candidates(layer, 1, 1 if layer == 0 else CM[0], gen, Xtr.device)
            add_filter(model, layer, Kc[0], bc[0])
            msg = f"layer {layer + 1}"
        else:
            sub = torch.randperm(Xtr.shape[0], generator=gen, device="cpu")[:n_s].to(Xtr.device)
            res = [layer_test(model, Xtr[sub], ytr[sub], l, args.M, args.B if rule == "suplm" else 0, gen)
                   if elig[l] else None for l in range(2)]
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
    return model, time.time() - t0, epochs


def fit_fixed(widths, epochs, Xtr, ytr, Xte, yte, args, seed, dev):
    """Train a network of fixed widths from scratch for a given number of epochs."""
    gen = torch.Generator(device="cpu").manual_seed(1000 * seed + 11)
    torch.manual_seed(1000 * seed + 11)
    model = GrowCNN(Cmax=(args.cmax1, args.cmax2), init=widths).to(dev)
    t0 = time.time()
    train(model, Xtr, ytr, epochs, gen=gen)
    nll, acc = evaluate(model, Xte, yte)
    return model, nll, acc, time.time() - t0


def run(args):
    global LOGF
    os.makedirs(OUT, exist_ok=True)
    LOGF = open(f"{OUT}/cnn_log.txt", "a")
    dev = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    log("=" * 70)
    log(f"torch {torch.__version__} on {dev}"
        + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else "") + f"; options {vars(args)}")
    rows = []
    total = len(args.data) * args.seeds
    done, t_all = 0, time.time()
    for name in args.data:
        for seed in range(args.seeds):
            Xtr, ytr, Xte, yte = load_data(name, args.n_train, dev, seed)
            log(f"data {name}, seed {seed}: train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}")
            ep_rule = {}
            for rule in ("suplm", "gradmax", "schedule"):
                gen = torch.Generator(device="cpu").manual_seed(1000 * seed + 7)
                tag = f"{name}/s{seed}"
                model, tg, ep = grow(rule, Xtr, ytr, args, gen, tag)
                ep_rule[rule] = (model.widths(), ep)
                nll, acc = evaluate(model, Xte, yte)
                c1, c2 = model.widths()
                rows.append(dict(data=name, seed=seed, rule=rule, c1=c1, c2=c2, params=model.n_params(),
                                 epochs=ep, cap1=int(c1 >= args.cmax1), cap2=int(c2 >= args.cmax2),
                                 test_nll=nll, test_acc=acc, grow_time=tg))
                log(f"  {tag} RESULT {rule}: widths {model.widths()}, params {model.n_params()}, "
                    f"test acc {acc:.4f}, test NLL {nll:.4f}, {ep} epochs, growth time {tg:.0f}s")
            # (a) the architecture chosen by suplm, trained from scratch with the same epochs
            w_s, ep_s = ep_rule["suplm"]
            model, nll, acc, tf = fit_fixed(w_s, ep_s, Xtr, ytr, Xte, yte, args, seed, dev)
            rows.append(dict(data=name, seed=seed, rule="suplm_refit", c1=w_s[0], c2=w_s[1],
                             params=model.n_params(), epochs=ep_s, cap1=int(w_s[0] >= args.cmax1),
                             cap2=int(w_s[1] >= args.cmax2), test_nll=nll, test_acc=acc, grow_time=np.nan))
            log(f"  {name}/s{seed} RESULT suplm_refit {w_s}: params {model.n_params()}, test acc {acc:.4f}, "
                f"test NLL {nll:.4f} ({ep_s} epochs from scratch, {tf:.0f}s)")
            # (b) fixed widths with the same epoch budget as suplm (or a fixed number)
            ep_f = ep_s if args.fixed_epochs == "match" else int(args.fixed_epochs)
            cand = ((2, 2), (8, 8), (16, 16), (32, 32), (args.cmax1, args.cmax2))
            fixed = sorted({w for w in cand if w[0] <= args.cmax1 and w[1] <= args.cmax2})
            for w in fixed:
                model, nll, acc, tf = fit_fixed(w, ep_f, Xtr, ytr, Xte, yte, args, seed, dev)
                rows.append(dict(data=name, seed=seed, rule=f"fixed{w[0]}x{w[1]}", c1=w[0], c2=w[1],
                                 params=model.n_params(), epochs=ep_f, cap1=0, cap2=0,
                                 test_nll=nll, test_acc=acc, grow_time=np.nan))
                log(f"  {name}/s{seed} RESULT fixed {w}: params {model.n_params()}, test acc {acc:.4f}, "
                    f"test NLL {nll:.4f} ({ep_f} epochs, {tf:.0f}s)")
            done += 1
            el = time.time() - t_all
            log(f"[{'#' * int(30 * done / total):<30s}] {done}/{total} data-seed pairs done, "
                f"elapsed {el / 60:.1f} min, ETA {el / done * (total - done) / 60:.1f} min")
            pd.DataFrame(rows).to_csv(f"{OUT}/cnn_raw.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby(["data", "rule"]).agg(
        c1=("c1", "mean"), c2=("c2", "mean"), params=("params", "mean"),
        epochs=("epochs", "mean"), cap1=("cap1", "mean"), cap2=("cap2", "mean"),
        acc_mean=("test_acc", "mean"), acc_sd=("test_acc", "std"),
        nll_mean=("test_nll", "mean"), time=("grow_time", "mean"), R=("seed", "size")).reset_index()
    summ.to_csv(f"{OUT}/cnn.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\n" + summ.to_string(index=False))
    LOGF.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["mnist", "fashion"], choices=["mnist", "fashion"])
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
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args(argv)
    if args.quick:
        args.seeds, args.n_train, args.n_test, args.cmax1, args.cmax2 = 1, 2000, 300, 8, 6
        args.M, args.B, args.epochs0, args.epochs1, args.data = 16, 49, 1, 1, ["mnist"]
    run(args)


if __name__ == "__main__":
    main()
