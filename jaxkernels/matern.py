"""Half-integer Matérn kernels as functions of the squared scaled distance.

    k(x, y) = variance * phi_p(s),   s = sum_i ((x_i - y_i) / lengthscale_i)^2,   nu = p + 1/2.

s is a polynomial in x, y and 1 / lengthscale, so every derivative of k is the chain rule applied to phi_p and a
polynomial. `matern_phi` supplies the closed-form derivatives of phi_p as a recursive custom JVP, so nested autodiff
to any depth is exact, including at x = y, where autodiff through r = sqrt(s) gives NaN.

With z = sqrt(2 nu s) and g_mu(z) = z^mu K_mu(z) (K the modified Bessel function of the second kind),

    phi_p(s) = c_nu g_nu(z),   c_nu = 2^(1 - nu) / Gamma(nu),

and d/dw g_mu = -g_(mu - 1) / 2 for w = z^2 = 2 nu s, so

    phi_p^(m)(s) = c_nu (-nu)^m g_(nu - m)(z).

For mu = n + 1/2, g_mu is exp(-z) times a sum of powers of z with positive coefficients, so phi_p^(m) is evaluated
without cancellation. For m > p the order nu - m is negative and g is singular at z = 0, like z^(2(p - m) + 1).

Near x = y: by Faa di Bruno, a derivative of total order k of phi(s(x, y)) is a sum of terms phi^(m)(s) times
products of derivatives of s. Since grad s = 0 at x = y, every term with m > k/2 has an exactly zero factor there,
and near x = y the terms with m > p are O(r^(2p + 1 - k)). So phi^(m)(0) := 0 for m > p, and for 0 < z < Z_MIN the
singular branch is evaluated at Z_MIN (the affected terms are below rounding error), which avoids overflow for
distinct points that are extremely close.

The kernel is therefore 2p times differentiable at x = y. Higher derivatives do not exist there and come out finite
but wrong (a NaN placeholder would also turn the valid orders into NaN), so a functional of order m applied to both
arguments needs p >= m.
"""
import math
from fractions import Fraction
from functools import lru_cache, partial

import jax
import jax.numpy as jnp

Z_MIN = 1e-15


@lru_cache(maxsize=None)
def _phi_coefficients(p, m):
    """(scale, leading power, coefficients) for phi_p^(m)(s) as an exponential times a polynomial in z.

    Exact rational arithmetic, then floats. Uses K_(n+1/2)(z) = sqrt(pi/(2z)) e^-z sum_j a_nj (2z)^-j,
    a_nj = (n+j)! / (j! (n-j)!), and c_nu sqrt(pi/2) = 2^p p! / (2p)! for nu = p + 1/2.
    """
    nu = Fraction(2 * p + 1, 2)
    scale = Fraction(2**p * math.factorial(p), math.factorial(2 * p)) * (-nu) ** m
    k = p - m                                # order nu - m = k + 1/2
    n = k if k >= 0 else -k - 1              # K_(n + 1/2)
    coefs = []
    for j in range(n + 1):
        a = Fraction(math.factorial(n + j), math.factorial(j) * math.factorial(n - j))
        coefs.append(a / 2**j)
    return float(scale), k, tuple(float(c) for c in coefs)


def _phi_value(p, m, s):
    scale, leading_power, coefs = _phi_coefficients(p, m)
    nu = p + 0.5
    z = jnp.sqrt(2.0 * nu * s)
    if m <= p:                               # polynomial in z, exponents k, k-1, ..., 0 (k = p - m)
        poly = coefs[0] * jnp.ones_like(z)
        for c in coefs[1:]:                  # Horner, highest power first
            poly = poly * z + c
        return scale * jnp.exp(-z) * poly
    zs = jnp.maximum(z, Z_MIN)               # singular branch (m > p): exponents -(n+1), ..., -(2n+1)
    inv = 1.0 / zs
    poly = coefs[-1] * jnp.ones_like(zs)
    for c in coefs[-2::-1]:                  # Horner in 1/z
        poly = poly * inv + c
    val = scale * jnp.exp(-zs) * poly * inv ** (-leading_power)
    return jnp.where(s > 0, val, 0.0)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def matern_phi(p, m, s):
    """m-th derivative of the Matérn-(p + 1/2) profile phi_p(s) in the squared scaled distance s >= 0.

    phi_p(0) = 1; phi_p(s) = exp(-sqrt(2 nu s)) * polynomial (e.g. p = 1: (1 + sqrt(3 s)) exp(-sqrt(3 s))).
    Differentiable to any order through its custom JVP (see the module docstring for m > p at s = 0).
    """
    return _phi_value(p, m, s)


@matern_phi.defjvp
def _matern_phi_jvp(p, m, primals, tangents):
    (s,), (ds,) = primals, tangents
    return matern_phi(p, m, s), matern_phi(p, m + 1, s) * ds


def matern_derivative_at_zero(p, m):
    """phi_p^(m)(0) for m <= p (Taylor coefficients: phi_p(s) = sum_m phi_p^(m)(0) s^m / m! + O(s^(p + 1/2)))."""
    if m > p:
        raise ValueError(f"phi_p^(m)(0) is infinite for m > p (p={p}, m={m})")
    scale, _, coefs = _phi_coefficients(p, m)
    return scale * coefs[-1]                 # the z^0 coefficient (exponent k - n = 0 for m <= p)
