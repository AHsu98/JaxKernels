"""Matérn kernels: values against scipy's Bessel K, exact Taylor coefficients at r = 0 (computed here independently,
in rational arithmetic, from the Rasmussen & Williams polynomial), all mixed partials at coincident points up to
order 2p (diagonally anisotropic radial and tensor-product forms, forward and reverse mode), and the Laplacian
functional."""
import itertools
import math
from fractions import Fraction
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma, kv

from jaxkernels import MaternKernel, ScalarMaternKernel, TensorProductKernel
from jaxkernels.kerneltools import eval_k, get_kernel_block_ops
from jaxkernels.matern import matern_derivative_at_zero, matern_phi


def scipy_matern(p, r):
    nu = p + 0.5
    z = np.sqrt(2 * nu) * np.abs(np.asarray(r, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = 2 ** (1 - nu) / gamma(nu) * z**nu * kv(nu, z)
    return np.where(z == 0, 1.0, v)


@lru_cache(maxsize=None)
def taylor_in_z(p, n_terms):
    """Coefficients of f(z) = exp(-z) p!/(2p)! sum_i (p+i)!/(i!(p-i)!) (2z)^(p-i) as a power series in z."""
    poly = [Fraction(0)] * (p + 1)                       # poly[j] = coefficient of z^j
    for i in range(p + 1):
        poly[p - i] += (Fraction(math.factorial(p), math.factorial(2 * p)) * math.factorial(p + i)
                        / (math.factorial(i) * math.factorial(p - i)) * 2 ** (p - i))
    return [sum(poly[j] * Fraction((-1) ** (n - j), math.factorial(n - j)) for j in range(min(n, p) + 1))
            for n in range(n_terms)]


@lru_cache(maxsize=None)
def phi_taylor(p, m):
    """phi_p^(m)(0) = m! (2 nu)^m [z^(2m)] f(z) for m <= p (phi_p(s) = f(sqrt(2 nu s)))."""
    c = taylor_in_z(p, 2 * m + 1)[2 * m]
    return float(math.factorial(m) * Fraction(2 * p + 1) ** m * c)


@pytest.mark.parametrize("p", range(6))
def test_taylor_series_smoothness(p):
    """The odd powers z^1, z^3, ..., z^(2p-1) vanish (k is C^2p at 0) and z^(2p+1) does not (exactly C^2p)."""
    c = taylor_in_z(p, 2 * p + 2)
    assert all(c[j] == 0 for j in range(1, 2 * p, 2))
    assert c[2 * p + 1] != 0
    for m in range(p + 1):
        assert matern_derivative_at_zero(p, m) == pytest.approx(phi_taylor(p, m), rel=1e-14)


@pytest.mark.parametrize("p", range(6))
def test_values_against_scipy(p):
    r = np.linspace(-4, 4, 81)
    phi = np.asarray(jax.vmap(lambda t: matern_phi(p, 0, t * t))(jnp.asarray(r)))
    np.testing.assert_allclose(phi, scipy_matern(p, r), rtol=0, atol=1e-15)
    # kernels: scalar, isotropic in R^3, diagonally anisotropic in R^2
    x = jnp.asarray(np.random.default_rng(p).uniform(-1, 1, (10, 3)))
    y = jnp.asarray(np.random.default_rng(p + 10).uniform(-1, 1, (10, 3)))
    k = ScalarMaternKernel(p, 0.7, 1.3)
    np.testing.assert_allclose(jax.vmap(k)(x[:, 0], y[:, 0]), 1.3 * scipy_matern(p, (x[:, 0] - y[:, 0]) / 0.7),
                               rtol=1e-14)
    k = MaternKernel(p, 0.6, 0.9)
    np.testing.assert_allclose(jax.vmap(k)(x, y), 0.9 * scipy_matern(p, np.linalg.norm(x - y, axis=1) / 0.6),
                               rtol=1e-14)
    ls = np.array([0.3, 0.8])
    k = MaternKernel(p, jnp.asarray(ls))
    r = np.sqrt(np.sum(((x[:, :2] - y[:, :2]) / ls) ** 2, axis=1))
    np.testing.assert_allclose(jax.vmap(k)(x[:, :2], y[:, :2]), scipy_matern(p, r), rtol=1e-14)


def _coincident_value(p, alpha, beta, ls, var):
    """d^alpha_x d^beta_y k(x, y) at x = y for k = var * phi_p(sum_i (x_i - y_i)^2 / ls_i^2)."""
    delta = [a + b for a, b in zip(alpha, beta)]
    if any(d % 2 for d in delta):
        return 0.0
    gam = [d // 2 for d in delta]
    m = sum(gam)
    val = var * (-1) ** sum(beta) * phi_taylor(p, m)
    for d, g, l in zip(delta, gam, ls):
        val *= math.factorial(d) / math.factorial(g) / l**d
    return val


def _derivative_tensors(F, u0, order, mode):
    f = F
    out = []
    for _ in range(order):
        f = (jax.jacfwd if mode == "fwd" else jax.jacrev)(f)
        out.append(np.asarray(jax.jit(f)(u0)))
    return out


def _check_tensors(tensors, d, value_fn, rtol):
    for order, T in enumerate(tensors, start=1):
        worst = 0.0
        scale = np.max(np.abs(T)) + 1e-300
        for idx in itertools.product(range(2 * d), repeat=order):
            alpha = [sum(1 for i in idx if i == c) for c in range(d)]
            beta = [sum(1 for i in idx if i == d + c) for c in range(d)]
            exact = value_fn(alpha, beta)
            worst = max(worst, abs(T[idx] - exact) / (abs(exact) + 1e-12 * scale))
        assert worst < rtol, f"order {order}: worst relative error {worst:.2e}"


@pytest.mark.parametrize("p", [1, 2, 3, pytest.param(4, marks=pytest.mark.slow)])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_anisotropic_mixed_partials_at_coincident_points(p, mode):
    """All partial derivatives d^alpha_x d^beta_y of the diagonally anisotropic Matérn at x = y with
    |alpha + beta| <= 2p (d = 2), against the Taylor values; forward mode (jacfwd^n) and reverse mode (jacrev^n)."""
    if mode == "rev" and p > 3:
        pytest.skip("reverse mode checked to order 6")
    d, ls, var = 2, (0.4, 0.9), 1.3
    k = MaternKernel(p, jnp.asarray(ls), var)
    x0 = jnp.array([0.3, -0.2])
    F = lambda u: k(u[:d], u[d:])
    order = 2 * p if mode == "fwd" or p <= 2 else 6
    tensors = _derivative_tensors(F, jnp.concatenate([x0, x0]), order, mode)
    _check_tensors(tensors, d, lambda a, b: _coincident_value(p, a, b, ls, var), rtol=1e-11)


@pytest.mark.parametrize("ps", [(1, 2), (2, 2), (2, 3)])
def test_tensor_product_matern_mixed_partials(ps):
    """TensorProductKernel of ScalarMaternKernels: d^alpha_x d^beta_y at x = y is the product of 1-D values; each
    coordinate allows p_i derivatives per argument (the mixed partial d_x1 d_x2 d_y1 d_y2 needs only p_i >= 1)."""
    d, ls = 2, (0.5, 0.8)
    k = TensorProductKernel([ScalarMaternKernel(ps[0], ls[0]), ScalarMaternKernel(ps[1], ls[1])])
    x0 = jnp.array([0.1, 0.7])
    F = lambda u: k(u[:d], u[d:])
    order = 2 * min(ps)                                     # every multi-index up to this order is covered
    tensors = _derivative_tensors(F, jnp.concatenate([x0, x0]), order, "fwd")

    def value(alpha, beta):
        out = 1.0
        for c in range(d):
            out *= _coincident_value(ps[c], [alpha[c]], [beta[c]], [ls[c]], 1.0)
        return out
    _check_tensors(tensors, d, value, rtol=1e-11)


def laplacian(k, index):
    """Trace of the Hessian in one argument."""
    def lapk(*x):
        return jnp.trace(jax.hessian(k, argnums=index)(*x))
    return lapk


@pytest.mark.parametrize("p", [2, 3])
def test_laplacian_blocks(p):
    """[eval, laplacian] kernel blocks (what InducingPointRKHS assembles for a Poisson-type problem): the diagonal
    of the laplacian-laplacian block at x = y matches the Taylor value sum_ij d^2_xi d^2_yj k, the cross block
    eval-laplacian matches sum_i d^2_yi k, and the whole Gram matrix is PSD (needs p >= 2)."""
    d, ls, var = 2, (0.5, 0.7), 1.1
    k = MaternKernel(p, jnp.asarray(ls), var)
    X = jnp.asarray(np.random.default_rng(0).uniform(0, 1, (12, d)))
    K = np.asarray(get_kernel_block_ops(k, [eval_k, laplacian], [eval_k, laplacian])(X, X))
    n = X.shape[0]
    lap_lap = sum(_coincident_value(p, [2 * (c == i) for c in range(d)], [2 * (c == j) for c in range(d)], ls, var)
                  for i in range(d) for j in range(d))
    ev_lap = sum(_coincident_value(p, [0] * d, [2 * (c == j) for c in range(d)], ls, var) for j in range(d))
    np.testing.assert_allclose(np.diag(K)[n:], lap_lap, rtol=1e-12)
    np.testing.assert_allclose(np.diag(K[:n, n:]), ev_lap, rtol=1e-12)
    ev = np.linalg.eigvalsh((K + K.T) / 2)
    assert ev[0] > -1e-10 * ev[-1]


def test_higher_orders_than_four():
    """p is not limited to 0..4 (closed form for any p)."""
    k = ScalarMaternKernel(6, 0.5)
    r = np.linspace(-2, 2, 21)
    np.testing.assert_allclose(jax.vmap(lambda t: k(0.0, t))(jnp.asarray(r)), scipy_matern(6, r / 0.5), atol=1e-15)
    g = lambda x, y: k(x, y)
    for n in range(1, 7):
        g = jax.grad(g, argnums=n % 2)
    exact = _coincident_value(6, [3], [3], [0.5], 1.0)
    assert float(jax.jit(g)(0.2, 0.2)) == pytest.approx(exact, rel=1e-11)


def test_nu_and_errors():
    assert MaternKernel(2).nu == 2.5
    with pytest.raises(ValueError):
        MaternKernel(1.5)
    with pytest.raises(ValueError):
        MaternKernel(-1)
    with pytest.raises(ValueError):
        MaternKernel(2, jnp.array([0.5, 0.5]))(jnp.zeros(3), jnp.zeros(3))   # lengthscale shape mismatch
    with pytest.raises(ValueError):
        ScalarMaternKernel(2)(jnp.zeros(2), jnp.zeros(2))
