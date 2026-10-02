"""Periodic kernels: stationary kernels of the chordal distance on circles.

Each coordinate x_i is mapped to the circle of circumference P_i (the period) in R^2, e_i(x) = (P_i / 2 pi) *
(cos(2 pi x_i / P_i), sin(2 pi x_i / P_i)); a kernel that is positive definite on R^(2d) restricted to the
embedded torus is positive definite and P-periodic in every coordinate. With lengthscales l_i,

    s(x, y) = sum_i |e_i(x) - e_i(y)|^2 / l_i^2 = sum_i ((P_i / (pi l_i)) sin(pi (x_i - y_i) / P_i))^2,

and s ~ sum_i ((x_i - y_i) / l_i)^2 for |x_i - y_i| << P_i, so l is a lengthscale in units of x, as for the
non-periodic kernels. s is analytic in (x, y), and s = 0 with grad s = 0 wherever x_i - y_i is a multiple of P_i,
so derivatives of Matérn profiles of s behave exactly as in the Euclidean case (matern.py).

    PeriodicKernel(period, lengthscale)        k = variance * exp(-s / 2)          (exp-sine-squared, MacKay)
    PeriodicMaternKernel(p, period, lengthscale)  k = variance * phi_p(s)          (Matérn of the chordal distance)

GPML's exp-sine-squared, exp(-2 sin^2(pi tau / P) / l_G^2), is PeriodicKernel with l = P l_G / (2 pi). For P = 1 the
PeriodicKernel equals func-keql's `TransformedKernel(GaussianRBFKernel(l), periodic_transform)` (examples/rd_common.py).

Smoothness: PeriodicKernel is analytic. PeriodicMaternKernel is 2p times differentiable at x = y (as MaternKernel);
its RKHS on the circle is norm-equivalent to the Sobolev space H^(nu + 1/2) (the trace of H^(nu + 1)(R^2) on a
curve), the same order as the Matérn-nu RKHS on the line.

Inputs: scalars (period and lengthscale scalars) or (d,) points (period, lengthscale scalar or (d,)); use them per
coordinate in TensorProductKernel, e.g. TensorProductKernel([MaternKernel(2, lt), PeriodicKernel(2.0, lx)]) for a
(t, x) problem periodic in x.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softplus

from .base_kernels import Kernel, as_float_array, check_positive, softplus_inverse, fmt
from .matern import matern_phi


def chordal_sqdist(x, y, period, lengthscale):
    """sum_i ((P_i / (pi l_i)) sin(pi (x_i - y_i) / P_i))^2 for scalar or (d,) points."""
    diff = jnp.asarray(x) - jnp.asarray(y)
    for name, v in (("period", period), ("lengthscale", lengthscale)):
        if jnp.ndim(v) == 1 and (diff.ndim > 1 or diff.size != jnp.shape(v)[0]):
            raise ValueError(f"{name} of shape {jnp.shape(v)} does not match points of shape {diff.shape}")
    u = jnp.sin(jnp.pi * diff / period) * (period / (jnp.pi * lengthscale))
    return jnp.sum(u**2)


class _PeriodicBase(Kernel):
    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    raw_period: jax.Array
    min_lengthscale: float = eqx.field(static=True)

    def _init_scales(self, period, lengthscale, variance, min_lengthscale):
        period, lengthscale, variance = map(as_float_array, (period, lengthscale, variance))
        check_positive(period, "period")
        check_positive(lengthscale, "lengthscale", min_lengthscale)
        check_positive(variance, "variance")
        self.raw_variance = softplus_inverse(variance)
        self.raw_lengthscale = softplus_inverse(lengthscale - min_lengthscale)
        self.raw_period = softplus_inverse(period)
        self.min_lengthscale = float(min_lengthscale)

    @property
    def variance(self):
        return softplus(self.raw_variance)

    @property
    def lengthscale(self):
        return softplus(self.raw_lengthscale) + self.min_lengthscale

    @property
    def period(self):
        return softplus(self.raw_period)

    def scale(self, c):
        return eqx.tree_at(lambda k: k.raw_variance, self, softplus_inverse(c * softplus(self.raw_variance)))


class PeriodicKernel(_PeriodicBase):
    """Exp-sine-squared kernel k = variance * exp(-s/2), s the chordal distance^2 (module docstring); analytic.

    period, lengthscale: scalars or (d,) arrays (one per coordinate); lengthscale in units of x (small-distance
    behaviour exp(-(x - y)^2 / (2 l^2))). Periods are learnable leaves: freeze them (hyper.hyperparameter_filter)
    when the period is known.
    """

    def __init__(self, period=1.0, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        self._init_scales(period, lengthscale, variance, min_lengthscale)

    def __call__(self, x, y):
        return self.variance * jnp.exp(-0.5 * chordal_sqdist(x, y, self.period, self.lengthscale))

    def __str__(self):
        return f"{fmt(self.variance)}Periodic(P={fmt(self.period)},{fmt(self.lengthscale)})"


class PeriodicMaternKernel(_PeriodicBase):
    """Matérn-(p + 1/2) kernel of the chordal distance: k = variance * phi_p(s) (module docstring).

    2p times differentiable at x = y (derivatives exact through matern_phi's custom JVP); RKHS on the circle
    ~ H^(p + 1).
    """
    p_order: int = eqx.field(static=True)

    def __init__(self, p, period=1.0, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        if int(p) != p or p < 0:
            raise ValueError(f"p must be a nonnegative integer (nu = p + 1/2), got {p}")
        self.p_order = int(p)
        self._init_scales(period, lengthscale, variance, min_lengthscale)

    def __call__(self, x, y):
        return self.variance * matern_phi(self.p_order, 0, chordal_sqdist(x, y, self.period, self.lengthscale))

    def __str__(self):
        return f"{fmt(self.variance)}PeriodicMatern({self.p_order},P={fmt(self.period)},{fmt(self.lengthscale)})"
