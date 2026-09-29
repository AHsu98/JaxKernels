"""Half-integer Matérn kernels, as functions of the squared scaled distance.

    k(x, y) = variance * phi_p(s),   s = sum_i ((x_i - y_i) / lengthscale_i)^2,   nu = p + 1/2.

Why s and not r = sqrt(s): s is a polynomial in x, y and the lengthscales, so every derivative of k (in x, y, or
the hyperparameters) is the chain rule applied to phi_p and a polynomial. phi_p has closed-form derivatives of
every order (below); `matern_phi` gives them to JAX as a custom JVP rule, recursively, so jax.grad / jax.hessian /
jacfwd nested to any depth give exact derivatives, including at x = y. (Autodiff through r = sqrt(s) at s = 0
gives NaN or one-sided values instead.)

With z = sqrt(2 nu s) and g_mu(z) = z^mu K_mu(z) (K the modified Bessel function of the second kind),

    phi_p(s) = c_nu g_nu(z),   c_nu = 2^(1 - nu) / Gamma(nu),

and d/dw g_mu = -g_(mu - 1) / 2 for w = z^2 = 2 nu s, so

    phi_p^(m)(s) = c_nu (-nu)^m g_(nu - m)(z).

For mu = n + 1/2 (n >= 0), g_mu is exp(-z) times a polynomial of degree n in z: finite at z = 0. For m > p the
order nu - m is negative and g is singular at z = 0 (like z^(2(p - m) + 1)). The kernel is 2p times
differentiable at x = y: by Faa di Bruno, a derivative of total order k of phi(s(x, y)) is a sum of terms
phi^(m)(s) * (products of derivatives of s); s is quadratic and grad s = 0 at x = y, so at x = y only m = k/2 <= p
survives, and near x = y the terms with m > p are O(r^(2p + 1 - k)). So phi^(m)(0) := 0 for m > p (it is always
multiplied by an exact zero), and for 0 < z < Z_MIN the singular branch is evaluated at Z_MIN (the affected
terms are below Z_MIN^(2p + 1 - k) <= 1e-15 relative), which avoids overflow for extremely close, distinct points.

Derivatives of order > 2p at x = y do not exist (the true values are infinite); they come out finite and wrong
(e.g. for p = 1, d^2/dx^2 d^2/dy^2 k(x, x) = 0 instead of +infinity; the sympy version gave a negative value,
-27 in the func-keql audit a10) and must not be
used: a basis or residual functional of order m needs p >= m. There is no way to make only the invalid orders fail
here: the placeholder phi^(m)(0) is multiplied by an exact zero in every valid derivative (a NaN placeholder would
turn those into NaN), and by a nonzero factor only in invalid ones.
"""
import math
from fractions import Fraction
from functools import lru_cache, partial

import jax
import jax.numpy as jnp

Z_MIN = 1e-15


@lru_cache(maxsize=None)
def _phi_coefficients(p, m):
    """(scale, exponents, coefficients): phi_p^(m)(s) = scale * exp(-z) * sum_j coef_j * z**exp_j.

    Exact rational arithmetic, then floats. Uses K_(n+1/2)(z) = sqrt(pi/(2z)) e^-z sum_j a_nj (2z)^-j,
    a_nj = (n+j)! / (j! (n-j)!), and c_nu sqrt(pi/2) = 2^p p! / (2p)! for nu = p + 1/2.
    """
    nu = Fraction(2 * p + 1, 2)
    scale = Fraction(2**p * math.factorial(p), math.factorial(2 * p)) * (-nu) ** m
    k = p - m                                # order nu - m = k + 1/2
    n = k if k >= 0 else -k - 1              # K_(n + 1/2)
    coefs, exps = [], []
    for j in range(n + 1):
        a = Fraction(math.factorial(n + j), math.factorial(j) * math.factorial(n - j))
        coefs.append(a / 2**j)
        exps.append(k - j)                   # z^(mu - 1/2 - j), mu - 1/2 = k
    return float(scale), tuple(exps), tuple(float(c) for c in coefs)


def _phi_value(p, m, s):
    scale, exps, coefs = _phi_coefficients(p, m)
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
    val = scale * jnp.exp(-zs) * poly * inv ** (-exps[0])
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
    scale, exps, coefs = _phi_coefficients(p, m)
    return scale * coefs[-1]                 # the z^0 coefficient (exponent k - n = 0 for m <= p)


# --------------------------------------------------------------------------------------------------------------
# Legacy sympy construction: the implementation of ScalarMaternKernel up to JaxKernels 2b015b7. The package no
# longer uses it; kept for comparison (sympy is imported only when it is called). It rebuilt the core on every
# call (0.03-0.25 s) and the result was stored as a static field, so every kernel instance was new structure;
# for p = 0 its JVP rule indexed an empty list (not differentiable at all).
# --------------------------------------------------------------------------------------------------------------
def make_custom_jvp_function(f, fprime):
    """Return a function with custom JVP defined by (fprime)."""
    @jax.custom_jvp
    def f_wrapped(x):
        return f(x)

    @f_wrapped.defjvp
    def f_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        return f(x), fprime(x) * x_dot
    return f_wrapped


def make_sympy_callable(expr):
    import sympy2jax

    def inner(d):
        return sympy2jax.SymbolicModule(expr)(d=d)
    return inner


def get_sympy_matern(p):
    import sympy as sym
    from sympy import factorial
    d2 = sym.symbols('d2', positive=True, real=True)
    exp_multiplier = -sym.sqrt(2 * p + 1)
    coefficients = [
        (factorial(p) / factorial(2 * p)) * (factorial(p + i) / (factorial(i) * factorial(p - i)))
        * (sym.sqrt(8 * p + 4))**(p - i)
        for i in range(p + 1)]
    powers = list(range(p, -1, -1))
    matern = (
        sum([
            c * sym.sqrt((d2**power)) for c, power in zip(coefficients, powers)
        ]
        )
        * sym.exp(exp_multiplier * sym.sqrt(d2))
    )
    return d2, matern


def build_matern_core(p):
    """Legacy sympy-built Matérn core of the scaled difference d (see the note above)."""
    import sympy as sym
    import sympy2jax
    d2, matern = get_sympy_matern(p)
    d = sym.var('d', pos=True, real=True)

    maternd = sym.powdenest(matern.subs(d2, d**2))
    subrule = {
        d * sym.DiracDelta(d): 0,
        sym.Abs(d) * sym.DiracDelta(d): 0,
        sym.Abs(d) * sym.sign(d): d,
        d * sym.sign(d): sym.Abs(d)
    }

    def compute_next_derivative(expr):
        return sym.powdenest(sym.expand(expr.diff(d).subs(subrule))).subs(subrule)

    derivatives = [compute_next_derivative(maternd)]
    for k in range(2 * p - 1):
        derivatives.append(compute_next_derivative(derivatives[-1]))

    jax_derivatives = [make_sympy_callable(f) for f in derivatives]

    wrapped_derivatives = [
        make_custom_jvp_function(f, fprime)
        for f, fprime in zip(jax_derivatives[:-1], jax_derivatives[1:])
    ]

    matern_func_raw = sympy2jax.SymbolicModule(maternd)
    core_matern = jax.custom_jvp(lambda d: matern_func_raw(d=d))

    @core_matern.defjvp
    def core_matern_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        ans = core_matern(x)
        ans_dot = wrapped_derivatives[0](x) * x_dot
        return ans, ans_dot

    return core_matern
