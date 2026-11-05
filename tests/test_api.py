"""Small checks of the public interface (run with: pytest -q)."""
import numpy as np
import torch

from supgrow import grow, sup_score_test


def _data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    xt = np.c_[np.ones(n), X]
    f = 1.5 * np.tanh(xt @ [0, 1, 0.5]) - 1.0 * np.tanh(xt @ [0.5, -0.5, 1.5])
    return X, f + 0.5 * rng.standard_normal(n)


def test_test_returns_valid_pvalue():
    X, y = _data()
    out = sup_score_test(X, y, K=0, M=100, B=99, device="cpu")
    assert 0 < out["p_value"] <= 1
    assert out["T"] >= 0


def test_linear_signal_rejects_k0():
    X, y = _data()
    out = sup_score_test(X, y, K=0, M=100, B=99, device="cpu")
    assert out["p_value"] <= 0.05


def test_grow_and_predict():
    X, y = _data()
    res = grow(X, y, K_max=4, M=100, B=99, n_random=3, device="cpu")
    assert 0 <= res.K <= 4
    yhat = res.predict(X)
    assert yhat.shape == (X.shape[0],)
    assert np.isfinite(yhat).all()
    assert res.history[-1]["decision"].startswith("stop")


def test_pure_noise_usually_stops_at_zero():
    rng = np.random.default_rng(3)
    X, y = rng.standard_normal((300, 2)), rng.standard_normal(300)
    torch.manual_seed(0)
    res = grow(X, y, K_max=3, M=100, B=99, device="cpu")
    assert res.K <= 1
