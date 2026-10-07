"""Periodic kernels: periodicity, closed forms (GPML exp-sine-squared, RBF of the circle embedding), Taylor values
of the periodic Matérn at coincident points and at the periodic images."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxkernels import (GaussianRBFKernel, PeriodicKernel, PeriodicMaternKernel, TensorProductKernel,
                        TransformedKernel, MaternKernel)
from _helpers import gram


def periodic_transform(x, period=1.0):
    return jnp.hstack([jnp.sin(x * (2 * jnp.pi / period)), jnp.cos(x * (2 * jnp.pi / period))]) / (period * 2 * jnp.pi)


@pytest.mark.parametrize("make", [lambda: PeriodicKernel(0.7, 0.3, 1.2), lambda: PeriodicMaternKernel(2, 0.7, 0.3)])
def test_periodicity(make):
    k = make()
    x = jnp.linspace(-1, 1, 9)
    y = 0.37
    base = jax.vmap(lambda a: k(a, y))(x)
    for n in (-2, 1, 3):
        np.testing.assert_allclose(jax.vmap(lambda a: k(a + n * 0.7, y))(x), base, rtol=1e-12, atol=1e-14)


def test_gpml_convention_and_rd_transform():
    P, lg = 0.8, 0.9
    ell = P * lg / (2 * np.pi)
    k = PeriodicKernel(P, ell, 1.0, min_lengthscale=0.0)
    tau = np.linspace(-1, 1, 11)
    np.testing.assert_allclose(jax.vmap(lambda t: k(0.0, t))(jnp.asarray(tau)),
                               np.exp(-2 * np.sin(np.pi * tau / P) ** 2 / lg**2), rtol=1e-13)
    k1 = PeriodicKernel(1.0, 0.3)
    k2 = TransformedKernel(GaussianRBFKernel(0.3), periodic_transform)
    X = jnp.linspace(0, 1, 7)
    np.testing.assert_allclose(gram(k1, X), jax.vmap(jax.vmap(k2, (None, 0)), (0, None))(X[:, None], X[:, None]),
                               rtol=1e-13)


def test_small_distance_lengthscale():
    """For |x - y| << P the kernels behave like their non-periodic counterparts with the same lengthscale."""
    tau = 1e-3
    kp, kr = PeriodicKernel(5.0, 0.2), GaussianRBFKernel(0.2)
    assert float(kp(0.0, tau)) == pytest.approx(float(kr(jnp.array([0.0]), jnp.array([tau]))), rel=1e-9)
    kp, km = PeriodicMaternKernel(2, 5.0, 0.2), MaternKernel(2, 0.2)
    assert float(kp(0.0, tau)) == pytest.approx(float(km(0.0, tau)), rel=1e-8)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_periodic_matern_derivatives_at_coincidence_and_images(p):
    """d^a_x d^b_y k at y = x and at y = x + P: s(tau) = (P/(pi l))^2 sin^2(pi tau / P) = tau^2/l^2 (1 + O(tau^2)),
    so the Taylor values differ from the Euclidean Matérn's from order 4 on; compare with the Euclidean kernel
    of the same s instead: phi_p(s(tau)) differentiated through the exact series of s."""
    P, l = 0.9, 0.35
    k = PeriodicMaternKernel(p, P, l)
    # reference: phi_p(s(tau)) with s written as a jnp expression (different code path: no chordal_sqdist)
    from jaxkernels.matern import matern_phi
    ref = lambda x, y: matern_phi(p, 0, ((P / (jnp.pi * l)) * jnp.sin(jnp.pi * (x - y) / P)) ** 2)
    for shift in (0.0, P, -2 * P):
        f, g = k, ref
        for n in range(2 * p):
            arg = n % 2
            f, g = jax.grad(f, arg), jax.grad(g, arg)
            a, b = float(f(0.3, 0.3 + shift)), float(g(0.3, 0.3))
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9 * (3 / l) ** (n + 1)), (shift, n + 1)
    # and the second derivative in closed form: d^2/dtau^2 phi(s(tau)) at 0 = phi'(0) s''(0) = phi'(0) * 2 / l^2
    from jaxkernels.matern import matern_derivative_at_zero
    d2 = jax.grad(jax.grad(k, 0), 0)(0.3, 0.3)
    assert float(d2) == pytest.approx(matern_derivative_at_zero(p, 1) * 2 / l**2, rel=1e-12)


def test_space_time_tensor_product_gram_psd():
    k = TensorProductKernel([MaternKernel(2, 0.4), PeriodicKernel(2.0, 0.3)])
    X = jnp.asarray(np.random.default_rng(0).uniform(0, 2, (30, 2)))
    ev = np.linalg.eigvalsh(np.asarray(gram(k, X)))
    assert ev[0] > -1e-12 * ev[-1]


def test_invalid():
    with pytest.raises(ValueError):
        PeriodicKernel(-1.0, 0.3)
    with pytest.raises(ValueError):
        PeriodicMaternKernel(2, 1.0, 0.001)
    with pytest.raises(ValueError):
        PeriodicKernel(jnp.array([1.0, 2.0]), 0.3)(jnp.zeros(3), jnp.zeros(3))
