"""Input-warped (non-stationary) kernels with learnable monotone warps.

    WarpedKernel(kernel, warp):  k_w(x, y) = kernel(warp(x), warp(y))

A warp is an equinox module (its array fields are hyperparameters, unlike TransformedKernel's static transform). If
the warp is injective, k_w is positive definite whenever the kernel is. Monotone coordinate warps are injective.

TanhWarp stretches one coordinate around m fronts:

    w(x)_a = x_a + sum_j A_j tanh((x_a - c_j - v_j x_t) / s_j)          (other coordinates unchanged)

with amplitudes A_j >= 0 and widths s_j > 0 (softplus leaves), centers c_j and optional velocities v_j (the fronts
move along a time coordinate t = x[time_axis]; unconstrained leaves). dw_a/dx_a = 1 + sum_j (A_j / s_j)
sech^2(...) >= 1: strictly increasing, so injective, and it only stretches: with a stationary base kernel of
lengthscale l, the local lengthscale near front j is about l / (1 + A_j / s_j) (finer), and l far from the fronts.
Every derivative exists (analytic warp), so the smoothness of k_w is that of the base kernel.

Use: fronts and shocks (a Burgers shock at a known or learnable position, possibly moving with speed v), boundary
layers. Initialise A_j small (A_j -> 0 is the identity warp) and widths near the expected front width. Centers and
velocities are unconstrained leaves: optimise all trainable leaves (hyper.hyperparameter_filter + eqx.partition),
not only the positive ones (hyper.to_log_vector covers only those).

Degenerate direction: for A_j >> s_j the identity part is negligible and only the ratio (base lengthscale) / A_j
matters. If the data are exactly a smooth function of tanh((x - c)/s) (e.g. a pure tanh front), a marginal-likelihood
fit can drift along that ridge to A, l -> infinity while predictions remain good. Bound A (a prior or a box) if
the values matter.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softplus

from .base_kernels import Kernel, as_float_array, check_positive, softplus_inverse, fmt


class TanhWarp(eqx.Module):
    """Monotone warp of coordinate `axis` (module docstring). axis=None for scalar inputs."""
    centers: jax.Array
    raw_amplitudes: jax.Array
    raw_widths: jax.Array
    velocities: jax.Array | None
    axis: int | None = eqx.field(static=True)
    time_axis: int | None = eqx.field(static=True)
    min_widths: float = eqx.field(static=True)

    def __init__(self, centers, amplitudes, widths, axis=None, velocities=None, time_axis=None, min_widths=0.0):
        centers = jnp.atleast_1d(as_float_array(centers))
        amplitudes = jnp.broadcast_to(as_float_array(amplitudes), centers.shape)
        widths = jnp.broadcast_to(as_float_array(widths), centers.shape)
        check_positive(amplitudes, "amplitudes")
        check_positive(widths, "widths", min_widths)
        if (velocities is None) != (time_axis is None):
            raise ValueError("give both velocities and time_axis, or neither")
        self.centers = centers
        self.raw_amplitudes = softplus_inverse(amplitudes)
        self.raw_widths = softplus_inverse(widths - min_widths)
        self.velocities = None if velocities is None else jnp.broadcast_to(as_float_array(velocities), centers.shape)
        self.axis = axis
        self.time_axis = time_axis
        self.min_widths = float(min_widths)

    @property
    def amplitudes(self):
        return softplus(self.raw_amplitudes)

    @property
    def widths(self):
        return softplus(self.raw_widths) + self.min_widths

    def warp_coordinate(self, xa, t=None):
        c = self.centers if self.velocities is None else self.centers + self.velocities * t
        return xa + jnp.sum(self.amplitudes * jnp.tanh((xa - c) / self.widths))

    def __call__(self, x):
        x = jnp.asarray(x)
        if self.axis is None:
            if x.size != 1 or self.time_axis is not None:
                raise ValueError("TanhWarp(axis=None) is for scalar inputs (no time axis)")
            return self.warp_coordinate(x.reshape(())).reshape(x.shape)
        if x.ndim != 1 or not -x.size <= self.axis < x.size:
            raise ValueError(f"TanhWarp axis {self.axis} is invalid for input shape {x.shape}")
        axis = self.axis % x.size
        if self.time_axis is not None:
            if not -x.size <= self.time_axis < x.size or self.time_axis % x.size == axis:
                raise ValueError(f"TanhWarp time_axis {self.time_axis} must be valid and distinct from axis {self.axis}")
            t = x[self.time_axis % x.size]
        else:
            t = None
        return x.at[axis].set(self.warp_coordinate(x[axis], t))

    def __str__(self):
        return f"TanhWarp(axis={self.axis}, c={fmt(self.centers)}, A={fmt(self.amplitudes)}, s={fmt(self.widths)})"


class WarpedKernel(Kernel):
    """k(warp(x), warp(y)) for a learnable warp module (module docstring)."""
    kernel: Kernel
    warp: eqx.Module

    def __call__(self, x, y):
        return self.kernel(self.warp(x), self.warp(y))

    def __str__(self):
        return f"Warped({self.kernel}, {self.warp})"
