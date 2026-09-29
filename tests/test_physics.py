"""Physics-informed and non-stationary kernels: the heat kernel against a direct numerical convolution and the
variance of its PDE residual, divergence-freeness of the stream-function kernel, the monotone warps, the weighted
sum over families."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxkernels import (GaussianRBFKernel, MaternKernel, HeatKernel, heat_residual_op, DivergenceFreeKernel,
                        IndexedMatrixKernel, TanhWarp, WarpedKernel, WeightedSumKernel, RationalQuadraticKernel)
from jaxkernels.kerneltools import eval_k, partial_op, get_kernel_block_ops
from _helpers import gram

J = eqx.filter_jit


def test_heat_kernel_initial_slice_is_rbf():
    k = HeatKernel(jnp.array([0.3, 0.5]), jnp.array([0.1, 0.2]), 1.4)
    r = GaussianRBFKernel(jnp.array([0.3, 0.5]), 1.4)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, y = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
        assert float(k(jnp.r_[0.0, x], jnp.r_[0.0, y])) == pytest.approx(float(r(jnp.asarray(x), jnp.asarray(y))),
                                                                          rel=1e-14)


def test_heat_kernel_against_numerical_convolution():
    """Cov(u(t, x), u(s, x')) = int int G_t(x - a) G_s(x' - b) k0(a - b) da db, by quadrature (1-D)."""
    ell, kappa, var = 0.3, 0.2, 1.3
    k = HeatKernel(ell, kappa, var)
    g = np.linspace(-4, 4, 2001)
    h = g[1] - g[0]
    G = lambda t, z: np.exp(-z**2 / (4 * kappa * t)) / np.sqrt(4 * np.pi * kappa * t)
    k0 = var * np.exp(-(g[:, None] - g[None, :]) ** 2 / (2 * ell**2))
    for (t, x), (s, xp) in [((0.1, 0.2), (0.3, -0.1)), ((0.5, 0.0), (0.05, 0.4)), ((0.2, 0.3), (0.2, 0.3))]:
        quad = h * h * G(t, x - g) @ k0 @ G(s, xp - g)
        assert float(k(jnp.array([t, x]), jnp.array([s, xp]))) == pytest.approx(quad, rel=1e-8)


@pytest.mark.parametrize("dim,ls,kappa", [(1, 0.3, 0.1), (2, (0.3, 0.5), 0.05), (2, (0.3, 0.5), (0.1, 0.02))])
def test_heat_residual_has_zero_variance(dim, ls, kappa):
    """L = d_t - sum_i kappa_i d_ii annihilates the kernel in each argument: the Gram block of L (both arguments)
    and the cross block L-eval vanish to round-off, relative to the blocks of d_t."""
    k = HeatKernel(jnp.asarray(ls), jnp.asarray(kappa))
    rng = np.random.default_rng(1)
    Z = jnp.asarray(np.column_stack([rng.uniform(0, 1, 12), rng.uniform(-1, 1, (12, dim))]))
    L = heat_residual_op(kappa if not isinstance(kappa, tuple) else tuple(kappa), dim)
    dt = partial_op(0)
    blocks = J(lambda k, Z: get_kernel_block_ops(k, [L, dt, eval_k], [L, dt, eval_k])(Z, Z))(k, Z)
    n = Z.shape[0]
    LL, LE = blocks[:n, :n], blocks[:n, 2 * n:]
    TT = blocks[n:2 * n, n:2 * n]
    scale = float(jnp.max(jnp.abs(TT)))
    assert float(jnp.max(jnp.abs(LL))) < 1e-12 * scale
    assert float(jnp.max(jnp.abs(LE))) < 1e-12 * float(jnp.max(jnp.abs(blocks[n:2 * n, 2 * n:])))
    ev = np.linalg.eigvalsh(np.asarray(blocks[n:, n:]))
    assert ev[0] > -1e-10 * ev[-1]
    # the functional is cached: same arguments, same object (structure)
    assert heat_residual_op(kappa if not isinstance(kappa, tuple) else tuple(kappa), dim) is L


def test_heat_kernel_hyperparameters_learnable():
    """d/d(kappa) of the kernel matches finite differences (kappa is a leaf, the PDE a property of the kernel)."""
    z1, z2 = jnp.array([0.3, 0.1, 0.2]), jnp.array([0.6, -0.2, 0.4])
    f = lambda kap: HeatKernel(jnp.array([0.3, 0.4]), kap)(z1, z2)
    g = jax.grad(f)(0.1)
    assert float(g) == pytest.approx(float((f(0.1 + 1e-6) - f(0.1 - 1e-6)) / 2e-6), rel=1e-7)


def test_heat_kernel_invalid():
    with pytest.raises(ValueError):
        HeatKernel(0.3, -0.1)
    with pytest.raises(ValueError):
        HeatKernel(jnp.array([0.3, 0.4]), 0.1)(jnp.zeros(2), jnp.zeros(2))     # 1 space dim, 2 lengthscales


@pytest.mark.parametrize("stream", [GaussianRBFKernel(0.4), MaternKernel(2, jnp.array([0.3, 0.5]))])
def test_divergence_free(stream):
    K = DivergenceFreeKernel(stream)
    rng = np.random.default_rng(2)
    X, Y = jnp.asarray(rng.uniform(0, 1, (6, 2))), jnp.asarray(rng.uniform(0, 1, (6, 2)))
    div = J(jax.vmap(lambda x, y: jnp.trace(jax.jacfwd(K, 0)(x, y), axis1=0, axis2=2)))(X, Y)   # sum_i d_xi K_ij
    dK = J(jax.vmap(lambda x, y: jax.jacfwd(K, 0)(x, y)))(X, Y)
    assert float(jnp.max(jnp.abs(div))) < 1e-13 * float(jnp.max(jnp.abs(dK)))
    # symmetry K(x, y) = K(y, x)^T and PSD block Gram
    np.testing.assert_allclose(jax.vmap(K)(X, Y), jnp.swapaxes(jax.vmap(K)(Y, X), 1, 2), rtol=1e-12, atol=1e-14)
    G = np.asarray(J(K.gram)(jnp.concatenate([X, Y])))
    ev = np.linalg.eigvalsh((G + G.T) / 2)
    assert ev[0] > -1e-10 * ev[-1]


def test_divergence_free_rbf_closed_form():
    ell, var = 0.4, 1.2
    K = DivergenceFreeKernel(GaussianRBFKernel(ell, var))
    x, y = jnp.array([0.1, 0.7]), jnp.array([0.5, 0.2])
    t = np.asarray(x - y)
    k = var * np.exp(-t @ t / (2 * ell**2))
    ref = k / ell**2 * np.array([[1 - t[1] ** 2 / ell**2, t[0] * t[1] / ell**2],
                                 [t[0] * t[1] / ell**2, 1 - t[0] ** 2 / ell**2]])
    np.testing.assert_allclose(K(x, y), ref, rtol=1e-13, atol=1e-14)


def test_indexed_matrix_kernel():
    K = DivergenceFreeKernel(GaussianRBFKernel(0.4))
    k = IndexedMatrixKernel(K)
    x, y = jnp.array([0.1, 0.7]), jnp.array([0.5, 0.2])
    for i in range(2):
        for j in range(2):
            assert float(k(jnp.r_[x, float(i)], jnp.r_[y, float(j)])) == pytest.approx(float(K(x, y)[i, j]), rel=1e-14)
    # the vector field x -> (k((x,0),(y,c)), k((x,1),(y,c))) is divergence-free for each (y, c)
    for c in (0.0, 1.0):
        zy = jnp.r_[y, c]
        div = jax.grad(lambda x: k(jnp.r_[x, 0.0], zy))(x)[0] + jax.grad(lambda x: k(jnp.r_[x, 1.0], zy))(x)[1]
        assert abs(float(div)) < 1e-13
    # the Gram matrix over (point, component) inputs equals the block Gram of K (component-major)
    X = jnp.asarray(np.random.default_rng(3).uniform(0, 1, (5, 2)))
    Z = jnp.concatenate([jnp.column_stack([X, jnp.zeros(5)]), jnp.column_stack([X, jnp.ones(5)])])
    np.testing.assert_allclose(gram(k, Z), K.gram(X), rtol=1e-13, atol=1e-15)


def test_tanh_warp_monotone_and_stretching():
    w = TanhWarp(jnp.array([0.3, 0.7]), jnp.array([0.05, 0.02]), jnp.array([0.02, 0.01]))
    x = jnp.linspace(-1, 2, 3001)
    wx = jax.vmap(w)(x)
    dw = jax.vmap(jax.grad(w))(x)
    assert float(jnp.min(dw)) >= 1.0 - 1e-15 and bool(jnp.all(jnp.diff(wx) > 0))
    assert float(jax.grad(w)(0.3)) == pytest.approx(1 + 0.05 / 0.02 + 0.02 / 0.01 * float(1 / jnp.cosh(0.4 / 0.01) ** 2),
                                                    rel=1e-12)
    # far from the fronts the warp is a shift: local lengthscale unchanged
    assert float(jax.grad(w)(1.9)) == pytest.approx(1.0, abs=1e-12)


def test_moving_front():
    w = TanhWarp(0.2, 0.05, 0.02, axis=1, velocities=0.5, time_axis=0)
    # at time t the front is at 0.2 + 0.5 t: the stretch is maximal there
    for t in (0.0, 0.4, 1.0):
        xs = jnp.linspace(0, 1, 2001)
        slope = jax.vmap(lambda xx: jax.grad(lambda v: w(jnp.array([t, v]))[1])(xx))(xs)
        assert float(xs[jnp.argmax(slope)]) == pytest.approx(0.2 + 0.5 * t, abs=1e-3)
    with pytest.raises(ValueError):
        TanhWarp(0.2, 0.05, 0.02, axis=1, velocities=0.5)


def test_warped_kernel_is_base_on_warped_points():
    w = TanhWarp(0.5, 0.1, 0.05, axis=1)
    base = GaussianRBFKernel(jnp.array([0.3, 0.2]))
    k = WarpedKernel(base, w)
    x, y = jnp.array([0.1, 0.45]), jnp.array([0.3, 0.55])
    assert float(k(x, y)) == pytest.approx(float(base(w(x), w(y))), rel=1e-15)
    # non-stationary: the correlation across the front is lower than the same separation away from it
    far = float(k(jnp.array([0.1, 0.1]), jnp.array([0.1, 0.2])))
    across = float(k(jnp.array([0.1, 0.45]), jnp.array([0.1, 0.55])))
    assert across < far


def test_weighted_sum():
    ks = [GaussianRBFKernel(0.3), RationalQuadraticKernel(0.5, 2.0), MaternKernel(2, 0.4)]
    w = jnp.array([0.5, 0.3, 0.2])
    k = WeightedSumKernel(ks, w)
    x, y = jnp.array([0.1, 0.4]), jnp.array([0.3, 0.2])
    assert float(k(x, y)) == pytest.approx(sum(float(wi * ki(x, y)) for wi, ki in zip(w, ks)), rel=1e-14)
    kn = WeightedSumKernel(ks, 2 * w, normalize=True)
    assert float(kn(x, y)) == pytest.approx(float(k(x, y)), rel=1e-14)
    assert float(kn(x, x)) == pytest.approx(1.0, rel=1e-14)
    np.testing.assert_allclose(kn.weights, w, rtol=1e-14)
    with pytest.raises(ValueError):
        WeightedSumKernel(ks, jnp.array([1.0, 1.0]))
    with pytest.raises(TypeError):
        WeightedSumKernel([])
