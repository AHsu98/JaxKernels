"""fit_kernel objectives against direct formulas: LOO against brute-force refits, the marginal likelihood against the
Gaussian log-density, and the noise floor that every builder shares."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.nn import softplus
from scipy.stats import multivariate_normal

from jaxkernels import GaussianRBFKernel, MaternKernel, softplus_inverse
from jaxkernels.fit_kernel import (SIGMA2_FLOOR, build_every_other_obj, build_loocv, build_neg_marglike,
                                   build_neg_marglike_partialobs, build_random_split_obj, fit_kernel, noise_variance)
from jaxkernels.kerneltools import vectorize_kfunc


def _data(n=25, d=2, m=None, seed=0):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.uniform(0, 1, (n, d)))
    y = jnp.sin(3 * X[:, 0]) + X[:, 1] ** 2
    if m is not None:
        y = jnp.stack([y, jnp.cos(2 * X[:, 1])], axis=1)
    return X, y + 0.01 * jnp.asarray(rng.standard_normal(y.shape))


def _params(kernel, sigma2):
    return {"kernel": kernel, "transformed_sigma2": softplus_inverse(jnp.asarray(sigma2 - SIGMA2_FLOOR))}


def brute_force_loo(kernel, X, y, s2):
    K = np.asarray(vectorize_kfunc(kernel)(X, X))
    y = np.asarray(y)
    errs = []
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        c = np.linalg.solve(K[np.ix_(keep, keep)] + s2 * np.eye(keep.sum()), y[keep])
        errs.append(y[i] - K[i, keep] @ c)
    return np.array(errs)


@pytest.mark.parametrize("m", [None, 2])
def test_loocv_matches_brute_force(m):
    X, y = _data(m=m)
    k = MaternKernel(2, jnp.array([0.3, 0.5]))
    s2 = 1e-3
    val = build_loocv(X, y)(_params(k, s2))
    e = brute_force_loo(k, X, y, s2)
    assert float(val) == pytest.approx(np.mean(e**2), rel=1e-9)


@pytest.mark.parametrize("m", [None, 2])
def test_neg_marglike_matches_direct_formula(m):
    X, y = _data(m=m)
    k = GaussianRBFKernel(jnp.array([0.3, 0.5]), 1.7)
    s2 = 1e-2
    val = float(build_neg_marglike(X, y)(_params(k, s2)))
    C = np.asarray(vectorize_kfunc(k)(X, X)) + s2 * np.eye(len(X))
    Y = np.asarray(y).reshape(len(X), -1)
    logpdf = sum(multivariate_normal(np.zeros(len(X)), C).logpdf(Y[:, j]) for j in range(Y.shape[1]))
    # the objective is 2 * (-log N(y | 0, C)) - n m log(2 pi)
    assert val == pytest.approx(-2 * logpdf - Y.size * np.log(2 * np.pi), rel=1e-11)


def test_partialobs_marglike():
    rng = np.random.default_rng(1)
    t = jnp.linspace(0, 1, 20)
    v = jnp.asarray(rng.uniform(0.5, 1.5, (20, 1)))
    y = jnp.asarray(rng.standard_normal(20))
    k = GaussianRBFKernel(0.3)
    s2 = 1e-2
    val = float(build_neg_marglike_partialobs(t, y, v)(_params(k, s2)))
    C = np.asarray(vectorize_kfunc(k)(t, t)) * np.asarray(v @ v.T) + s2 * np.eye(20)
    logpdf = multivariate_normal(np.zeros(20), C).logpdf(np.asarray(y))
    assert val == pytest.approx(-2 * logpdf - 20 * np.log(2 * np.pi), rel=1e-11)


def test_all_builders_use_the_same_noise_variance():
    """Every objective sees softplus(raw) + SIGMA2_FLOOR, which is what fit_kernel returns."""
    X, y = _data()
    k = GaussianRBFKernel(0.3)
    s2 = 2e-6                                           # close to the floor, where the old mismatch mattered
    params = _params(k, s2)
    assert float(noise_variance(params)) == pytest.approx(s2, rel=1e-12)
    e = brute_force_loo(k, X, y, s2)
    assert float(build_loocv(X, y)(params)) == pytest.approx(np.mean(e**2), rel=1e-6)
    K = np.asarray(vectorize_kfunc(k)(X[::2], X[::2]))
    pred = np.asarray(vectorize_kfunc(k)(X[1::2], X[::2])) @ np.linalg.solve(K + s2 * np.eye(len(K)),
                                                                            np.asarray(y[::2]))
    assert float(build_every_other_obj(X, y)(params)) == pytest.approx(np.mean((pred - np.asarray(y[1::2]))**2),
                                                                       rel=1e-6)
    assert np.isfinite(float(build_random_split_obj(X, y)(params)))


def test_fit_functions_initialize_the_requested_total_variance(monkeypatch):
    fitmod = importlib.import_module("jaxkernels.fit_kernel")
    monkeypatch.setattr(fitmod, "run_gradient_descent", lambda loss, params, **kwargs: (params, []))
    monkeypatch.setattr(fitmod, "run_jaxopt_solver", lambda solver, params, **kwargs: (params, [], None))
    requested = 7 * SIGMA2_FLOOR
    X, y = _data(n=4)
    _, ordinary, _ = fitmod.fit_kernel(GaussianRBFKernel(0.5), requested, X, y, show_progress=False)
    _, partial, _ = fitmod.fit_kernel_partialobs(
        GaussianRBFKernel(0.5), requested, X[:, 0], y, jnp.ones((len(X), 1)), show_progress=False
    )
    np.testing.assert_allclose([ordinary, partial], requested, rtol=2e-15, atol=0.0)


def test_fit_kernel_runs_and_returns_the_optimized_variance():
    X, y = _data(n=20)
    k, s2, hist = fit_kernel(GaussianRBFKernel(0.5), 1e-2, X, y, loss_builder=build_loocv, max_gd_iter=20,
                             max_lbfgs_iter=20, show_progress=False)
    assert np.isfinite(float(s2)) and float(s2) >= SIGMA2_FLOOR
    assert float(hist[1]["values"][-1]) <= float(hist[0]["values"][0]) + 1e-12


def test_gradients_are_finite_and_match_fd():
    X, y = _data()
    loss = build_neg_marglike(X, y)
    params = _params(MaternKernel(2, jnp.array([0.3, 0.5])), 1e-2)
    g = jax.grad(loss)(params)
    h = 1e-6
    for idx in range(2):
        def at(dl):
            p = dict(params)
            p["kernel"] = jax.tree_util.tree_map(lambda a: a, params["kernel"])
            raw = params["kernel"].raw_lengthscale.at[idx].add(dl)
            import equinox as eqx
            p["kernel"] = eqx.tree_at(lambda kk: kk.raw_lengthscale, params["kernel"], raw)
            return float(loss(p))
        fd = (at(h) - at(-h)) / (2 * h)
        assert float(g["kernel"].raw_lengthscale[idx]) == pytest.approx(fd, rel=1e-6)
    assert float(softplus(params["transformed_sigma2"])) > 0
