"""Data-only objectives on observations of linear functionals, against independent references: the Gram matrix from
explicit per-entry derivatives, the Gaussian log-density (scipy), leave-one-out and K-fold residuals from brute-force
refits, the posterior from the textbook formula; gradients against finite differences."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from jaxkernels import GaussianRBFKernel, MaternKernel, ScalarMaternKernel
from jaxkernels.kerneltools import eval_k, partial_op, derivative_op, laplacian
from jaxkernels.objectives import (Observations, covariance, neg_log_marginal_likelihood, loo_residuals, loo_mse,
                                   loo_nlpd, kfold_indices, kfold_residuals, kfold_mse, kfold_nlpd, posterior)
from _helpers import count_traces

J = eqx.filter_jit          # the objectives are meant to be jitted: eagerly, nested derivative functionals are slow


def _problem(seed=0):
    """u(x) = sin(2 x0) cos(x1) observed as values (8 points), d/dx0 (5 points), Laplacian (4 points)."""
    rng = np.random.default_rng(seed)
    X0, X1, X2 = (jnp.asarray(rng.uniform(0, 1.5, (n, 2))) for n in (8, 5, 4))
    obs = Observations([(eval_k, X0), (partial_op(0), X1), (laplacian, X2)])
    u = lambda x: jnp.sin(2 * x[0]) * jnp.cos(x[1])
    y = jnp.concatenate([jax.vmap(u)(X0), jax.vmap(lambda x: jax.grad(u)(x)[0])(X1),
                         jax.vmap(lambda x: jnp.trace(jax.hessian(u)(x)))(X2)])
    y = y + 0.01 * jnp.asarray(rng.standard_normal(y.shape))
    return obs, y


def reference_gram(k, obs):
    """Closed-form derivatives of the ARD RBF kernel k~(tau) = var exp(-sum tau_i^2 / (2 l_i^2)), tau = x - y, in NumPy
    (no autodiff): L_x M_y k = L_tau (-1)^ord(M) M_tau k~. With A = sum_i (tau_i^2 / l_i^4 - 1 / l_i^2) = Lap k~ / k~:
    d0 k~ = -tau_0/l_0^2 k~, d0^2 k~ = (tau_0^2/l_0^4 - 1/l_0^2) k~, d0 Lap k~ = (2 tau_0/l_0^4 - A tau_0/l_0^2) k~,
    Lap^2 k~ = (sum_i 2/l_i^4 - 4 sum_i tau_i^2/l_i^6 + A^2) k~."""
    ls, var = np.asarray(k.lengthscale), float(k.variance)
    order = {eval_k: 0, partial_op(0): 1, laplacian: 2}

    def entry(a, b, x, y):
        t = np.asarray(x) - np.asarray(y)
        kt = var * np.exp(-0.5 * np.sum(t**2 / ls**2))
        A = np.sum(t**2 / ls**4 - 1 / ls**2)
        d0 = -t[0] / ls[0] ** 2
        d00 = t[0] ** 2 / ls[0] ** 4 - 1 / ls[0] ** 2
        d0lap = 2 * t[0] / ls[0] ** 4 - A * t[0] / ls[0] ** 2
        lap2 = np.sum(2 / ls**4) - 4 * np.sum(t**2 / ls**6) + A**2
        table = {(0, 0): 1.0, (0, 1): -d0, (0, 2): A, (1, 0): d0, (1, 1): -d00, (1, 2): d0lap,
                 (2, 0): A, (2, 1): -d0lap, (2, 2): lap2}
        return table[(a, b)] * kt
    rows = [(order[op], x) for op, X in zip(obs.ops, obs.points) for x in np.asarray(X)]
    return np.array([[entry(a, b, x, y) for b, y in rows] for a, x in rows])


KERNEL = GaussianRBFKernel(jnp.array([0.5, 0.8]), 1.3)
OBS, Y = _problem()


def test_gram_against_reference():
    obs = OBS
    np.testing.assert_allclose(np.asarray(J(obs.gram)(KERNEL)), reference_gram(KERNEL, obs), rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(J(obs.gram_diagonal)(KERNEL), np.diag(reference_gram(KERNEL, obs)), rtol=1e-12)


def test_loo_with_matern_derivative_observations():
    """Values and first derivatives with the ARD Matérn-5/2 (twice differentiable per argument)."""
    rng = np.random.default_rng(3)
    X0, X1 = jnp.asarray(rng.uniform(0, 1, (10, 2))), jnp.asarray(rng.uniform(0, 1, (6, 2)))
    obs = Observations([(eval_k, X0), (partial_op(1), X1)])
    y = jnp.concatenate([jnp.sin(3 * X0[:, 1]) + X0[:, 0], 3 * jnp.cos(3 * X1[:, 1])])
    k = MaternKernel(2, jnp.array([0.4, 0.6]))
    C = np.asarray(J(covariance)(k, 1e-4, obs))
    e, var = J(loo_residuals)(k, 1e-4, obs, y)
    for i in range(len(y)):
        ei, vi = _brute_force_heldout(C, np.asarray(y), np.array([i]))
        assert float(e[i]) == pytest.approx(float(ei[0]), rel=1e-7, abs=1e-10)
        assert float(var[i]) == pytest.approx(float(vi[0, 0]), rel=1e-7)


@pytest.mark.parametrize("noise", [1e-3, (1e-4, 1e-3, 1e-2)])
def test_marginal_likelihood(noise):
    obs, y = OBS, Y
    C = reference_gram(KERNEL, obs)
    C = C + np.diag(np.asarray(obs.noise_diagonal(noise)) + 1e-10 * np.diag(C))
    ref = -multivariate_normal(np.zeros(len(y)), C).logpdf(np.asarray(y))
    assert float(J(neg_log_marginal_likelihood)(KERNEL, noise, obs, y)) == pytest.approx(ref, rel=1e-10)
    np.testing.assert_allclose(J(covariance)(KERNEL, noise, obs), C, rtol=1e-12, atol=1e-11)


def _brute_force_heldout(C, y, F):
    """y_F - E[y_F | y_-F] and Cov[y_F | y_-F] by conditioning directly."""
    rest = np.setdiff1d(np.arange(len(y)), F)
    S = np.linalg.solve(C[np.ix_(rest, rest)], C[np.ix_(rest, F)])
    mean = S.T @ y[rest]
    cov = C[np.ix_(F, F)] - C[np.ix_(F, rest)] @ S
    return y[F] - mean, cov


def test_loo_against_brute_force():
    obs, y = _problem()
    noise = 1e-3
    C = np.asarray(J(covariance)(KERNEL, noise, obs))
    e, var = J(loo_residuals)(KERNEL, noise, obs, y)
    yn = np.asarray(y)
    for i in range(len(y)):
        ei, vi = _brute_force_heldout(C, yn, np.array([i]))
        assert float(e[i]) == pytest.approx(float(ei[0]), rel=1e-7, abs=1e-10)
        assert float(var[i]) == pytest.approx(float(vi[0, 0]), rel=1e-7)
    ref_nlpd = np.mean([0.5 * np.log(2 * np.pi * v) + 0.5 * ee**2 / v for ee, v in zip(np.asarray(e), np.asarray(var))])
    assert float(J(loo_nlpd)(KERNEL, noise, obs, y)) == pytest.approx(ref_nlpd, rel=1e-12)
    assert float(J(loo_mse)(KERNEL, noise, obs, y)) == pytest.approx(float(np.mean(np.asarray(e) ** 2)), rel=1e-12)


def test_kfold_against_brute_force():
    obs, y = _problem()
    noise = (1e-4, 1e-3, 1e-2)
    C = np.asarray(J(covariance)(KERNEL, noise, obs))
    folds = kfold_indices(len(y), 4, seed=1)
    yn = np.asarray(y)
    total, count, sq = 0.0, 0, 0.0
    res = J(lambda k, o, yy: [(e, A) for _, e, A in kfold_residuals(k, noise, o, yy, folds)])(KERNEL, obs, y)
    for (e, A), F, Fref in zip(res, folds, folds):
        eref, cov = _brute_force_heldout(C, yn, Fref)
        np.testing.assert_allclose(np.asarray(e), eref, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(np.linalg.inv(np.asarray(A)), cov, rtol=1e-7, atol=1e-12)
        total -= multivariate_normal(np.zeros(len(F)), cov).logpdf(eref)
        count += len(F)
        sq += np.sum(eref**2)
    assert float(J(lambda k, o, yy: kfold_nlpd(k, noise, o, yy, folds))(KERNEL, obs, y)) == pytest.approx(
        total / count, rel=1e-8)
    assert float(J(lambda k, o, yy: kfold_mse(k, noise, o, yy, folds))(KERNEL, obs, y)) == pytest.approx(
        sq / count, rel=1e-8)
    # leave-one-out is K-fold with singleton folds
    e_loo, _ = J(loo_residuals)(KERNEL, noise, obs, y)
    singles = [np.array([i]) for i in range(len(y))]
    e_single = J(lambda k, o, yy: [e for _, e, _ in kfold_residuals(k, noise, o, yy, singles)])(KERNEL, obs, y)
    np.testing.assert_allclose([float(e[0]) for e in e_single], np.asarray(e_loo), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("fold", [np.array([len(Y)]), np.array([-1]), np.array([1, 1]), np.array([1.0])])
def test_kfold_rejects_invalid_indices(fold):
    with pytest.raises(ValueError, match="invalid fold indices"):
        kfold_residuals(KERNEL, 1e-3, OBS, Y, [fold])


def test_posterior():
    obs, y = _problem()
    noise = 1e-3
    targets = Observations([(eval_k, jnp.array([[0.2, 0.3], [1.0, 0.5]])), (partial_op(1), jnp.array([[0.4, 0.4]]))])
    mean, var = J(posterior)(KERNEL, noise, obs, y, targets)
    C = np.asarray(J(covariance)(KERNEL, noise, obs))
    Ks = np.asarray(J(lambda k, t, o: t.gram(k, o))(KERNEL, targets, obs))
    Kss = np.asarray(J(lambda k, t: t.gram(k))(KERNEL, targets))
    np.testing.assert_allclose(mean, Ks @ np.linalg.solve(C, np.asarray(y)), rtol=1e-8)
    np.testing.assert_allclose(var, np.diag(Kss - Ks @ np.linalg.solve(C, Ks.T)), rtol=1e-6, atol=1e-12)
    truth = np.array([np.sin(0.4) * np.cos(0.3), np.sin(2.0) * np.cos(0.5), -np.sin(0.8) * np.sin(0.4)])
    assert np.all(np.abs(np.asarray(mean) - truth) < 3 * np.sqrt(np.asarray(var)))   # sanity: truth within 3 std


def test_multi_output():
    obs, y = _problem()
    Y = jnp.stack([y, 2 * y + 0.1], axis=1)
    noise = 1e-3
    nlml = J(neg_log_marginal_likelihood)
    assert float(nlml(KERNEL, noise, obs, Y)) == pytest.approx(
        float(nlml(KERNEL, noise, obs, Y[:, 0])) + float(nlml(KERNEL, noise, obs, Y[:, 1])), rel=1e-12)
    loo = J(loo_residuals)
    e, _ = loo(KERNEL, noise, obs, Y)
    np.testing.assert_allclose(e[:, 1], loo(KERNEL, noise, obs, Y[:, 1])[0], rtol=1e-12, atol=1e-14)
    folds = kfold_indices(len(y), 3)
    mse = J(lambda k, o, yy: kfold_mse(k, noise, o, yy, folds))
    assert float(mse(KERNEL, obs, Y)) == pytest.approx(
        0.5 * (float(mse(KERNEL, obs, Y[:, 0])) + float(mse(KERNEL, obs, Y[:, 1]))), rel=1e-12)


@pytest.mark.parametrize("objective", ["nlml", "loo_mse", "loo_nlpd", "kfold_nlpd"])
def test_gradients_against_fd(objective):
    obs, y = _problem()
    folds = kfold_indices(len(y), 4, seed=2)
    fns = {"nlml": neg_log_marginal_likelihood, "loo_mse": loo_mse, "loo_nlpd": loo_nlpd,
           "kfold_nlpd": lambda k, s, o, yy: kfold_nlpd(k, s, o, yy, folds)}
    f = fns[objective]

    def loss(params):
        return f(params["kernel"], jnp.exp(params["log_noise"]), obs, y)
    params = {"kernel": KERNEL, "log_noise": jnp.log(1e-3)}
    g = jax.jit(jax.grad(loss))(params)
    loss_jit = jax.jit(loss)
    h = 1e-5
    checks = [(lambda p, d: {"kernel": eqx.tree_at(lambda k: k.raw_lengthscale, p["kernel"],
                                                   p["kernel"].raw_lengthscale.at[0].add(d)),
                             "log_noise": p["log_noise"]}, g["kernel"].raw_lengthscale[0]),
              (lambda p, d: {"kernel": eqx.tree_at(lambda k: k.raw_variance, p["kernel"], p["kernel"].raw_variance + d),
                             "log_noise": p["log_noise"]}, g["kernel"].raw_variance),
              (lambda p, d: {"kernel": p["kernel"], "log_noise": p["log_noise"] + d}, g["log_noise"])]
    for shift, ad in checks:
        fd = (-loss_jit(shift(params, 2 * h)) + 8 * loss_jit(shift(params, h)) - 8 * loss_jit(shift(params, -h))
              + loss_jit(shift(params, -2 * h))) / (12 * h)
        assert float(ad) == pytest.approx(float(fd), rel=2e-6, abs=1e-8)


def test_no_retrace_across_hyperparameters():
    obs, y = _problem()
    traces = []

    @eqx.filter_jit
    def f(k, s):
        traces.append(1)
        return neg_log_marginal_likelihood(k, s, obs, y)
    for ls in ([0.5, 0.8], [0.3, 0.4], [1.0, 2.0]):
        f(MaternKernel(3, jnp.array(ls)), 1e-3)
    assert len(traces) == 1


def test_jitter_makes_duplicates_factorable():
    X = jnp.array([[0.1, 0.2], [0.1, 0.2], [0.5, 0.5]])
    y = jnp.array([1.0, 1.0, 0.3])
    k = GaussianRBFKernel(0.4)
    assert not np.isfinite(float(neg_log_marginal_likelihood(k, 0.0, X, y, jitter=0.0)))
    assert np.isfinite(float(neg_log_marginal_likelihood(k, 0.0, X, y, jitter=1e-8)))


def test_scalar_inputs_and_plain_arrays():
    """obs may be a plain array of points (values only); scalar-input kernels use derivative_op."""
    x = jnp.linspace(0, 1, 15)
    y = jnp.sin(3 * x)
    k = ScalarMaternKernel(2, 0.3)
    v1 = neg_log_marginal_likelihood(k, 1e-4, x, y)
    v2 = neg_log_marginal_likelihood(k, 1e-4, Observations([(eval_k, x)]), y)
    assert float(v1) == float(v2)
    obs = Observations([(eval_k, x), (derivative_op(1), x[::3])])
    yy = jnp.concatenate([y, 3 * jnp.cos(3 * x[::3])])
    e, var = loo_residuals(k, 1e-4, obs, yy)
    assert np.all(np.isfinite(np.asarray(e))) and np.all(np.asarray(var) > 0)
