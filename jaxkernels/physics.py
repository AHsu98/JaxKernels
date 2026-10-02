"""Kernels whose samples satisfy a linear PDE or constraint exactly.

HeatKernel: space-time covariance of the heat equation u_t = sum_i kappa_i d^2u/dx_i^2 on R^d with a Gaussian-process
initial condition u(0, .) ~ GP(0, variance * exp(-sum_i (x_i - x'_i)^2 / (2 l_i^2))). The heat semigroup is
convolution with a Gaussian of covariance 2 t diag(kappa), so Cov(u(t, x), u(s, x')) convolves the Gaussian k0 with
two more Gaussians:

    k((t, x), (s, x')) = variance * prod_i  l_i / sqrt(v_i) * exp(-(x_i - x'_i)^2 / (2 v_i)),   v_i = l_i^2 + 2 kappa_i (t + s)

(t, s >= 0; positive definite while every v_i > 0). Every function in its RKHS solves the heat equation, so the PDE
need not be imposed as a term, and the diffusivity is a kernel hyperparameter (fit it from data by marginal
likelihood or CV, jaxkernels.objectives). Inputs z = (t, x_1, ..., x_d) (time first); lengthscale scalar or (d,);
diffusivity scalar (isotropic) or (d,) (diagonal diffusion). The initial covariance is the RBF, so these solutions
are analytic in x for t >= 0; a rougher initial condition needs another kernel (no closed form for Matérn).

DivergenceFreeKernel: the 2-D matrix-valued kernel of v = (d psi/dx_2, -d psi/dx_1) for a stream function
psi ~ GP(0, k_psi):

    K(x, y) = [[ d_x2 d_y2 k,  -d_x2 d_y1 k],
               [-d_x1 d_y2 k,   d_x1 d_y1 k]]

Every column is divergence-free in x, so every function of its RKHS is. The stream kernel must be at least twice
differentiable per argument for K to exist (Matérn p >= 1; derivative observations of v need p >= 2).

IndexedMatrixKernel: a matrix-valued kernel as a scalar kernel on (x, c), where the last input coordinate c in
{0, ..., m-1} picks the output component: k((x, c), (y, c')) = K(x, y)[c, c'] (positive definite on R^d x {0..m-1}
iff K is). Scalar-kernel machinery (func_graph_comp's InducingPointRKHS, objectives.Observations) can then represent
vector fields: inducing points (x_j, c) for each component c; derivative functionals act on the x coordinates
(their derivative in c is 0).
"""
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softplus

from .base_kernels import Kernel, as_float_array, check_positive, softplus_inverse, fmt
from .kernels import MaternKernel
from .kerneltools import partial_op
from .periodic import PeriodicMaternKernel


class HeatKernel(Kernel):
    """Space-time kernel of the heat equation u_t = sum_i kappa_i u_(x_i x_i) (module docstring); inputs (t, x)."""
    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    raw_diffusivity: jax.Array
    min_lengthscale: float = eqx.field(static=True)

    def __init__(self, lengthscale=1.0, diffusivity=1.0, variance=1.0, min_lengthscale=0.01):
        lengthscale, diffusivity, variance = map(as_float_array, (lengthscale, diffusivity, variance))
        check_positive(lengthscale, "lengthscale", min_lengthscale)
        check_positive(diffusivity, "diffusivity")
        check_positive(variance, "variance")
        self.raw_variance = softplus_inverse(variance)
        self.raw_lengthscale = softplus_inverse(lengthscale - min_lengthscale)
        self.raw_diffusivity = softplus_inverse(diffusivity)
        self.min_lengthscale = float(min_lengthscale)

    @property
    def variance(self):
        return softplus(self.raw_variance)

    @property
    def lengthscale(self):
        return softplus(self.raw_lengthscale) + self.min_lengthscale

    @property
    def diffusivity(self):
        return softplus(self.raw_diffusivity)

    def __call__(self, z1, z2):
        z1, z2 = jnp.asarray(z1), jnp.asarray(z2)
        if z1.ndim != 1 or z1.shape[0] < 2 or z1.shape != z2.shape:
            raise ValueError(f"HeatKernel takes points (t, x_1, ..., x_d), got shapes {z1.shape}, {z2.shape}")
        d = z1.shape[0] - 1
        ls, kappa = self.lengthscale, self.diffusivity
        for name, v in (("lengthscale", ls), ("diffusivity", kappa)):
            if jnp.ndim(v) == 1 and jnp.shape(v)[0] != d:
                raise ValueError(f"{name} of shape {jnp.shape(v)} for {d} space dimensions")
        ls2 = jnp.broadcast_to(ls**2, (d,))
        v = ls2 + 2.0 * jnp.broadcast_to(kappa, (d,)) * (z1[0] + z2[0])
        diff = z1[1:] - z2[1:]
        return self.variance * jnp.prod(jnp.sqrt(ls2 / v)) * jnp.exp(-0.5 * jnp.sum(diff**2 / v))

    def __str__(self):
        return f"{fmt(self.variance)}Heat({fmt(self.lengthscale)},kappa={fmt(self.diffusivity)})"


@lru_cache(maxsize=None)
def heat_residual_op(diffusivity, dim):
    """The functional u -> u_t - sum_i kappa_i u_(x_i x_i) for points (t, x_1..x_dim) and a concrete diffusivity
    (a float, or a tuple of dim floats); cached, so equal arguments give the same function. For a diffusivity that
    is being traced (a hyperparameter), write the functional inside the traced function instead."""
    kappas = (float(diffusivity),) * dim if not isinstance(diffusivity, tuple) else tuple(map(float, diffusivity))
    dt = partial_op(0)
    dxx = [partial_op(i + 1, i + 1) for i in range(dim)]

    def op(k, index):
        ft = dt(k, index)
        fxx = [L(k, index) for L in dxx]
        return lambda *z: ft(*z) - sum(c * f(*z) for c, f in zip(kappas, fxx))
    op.__name__ = op.__qualname__ = f"heat_residual_{kappas}"
    return op


class MatrixKernel(eqx.Module):
    """Base for matrix-valued kernels K(x, y) of shape (m, m)."""

    @property
    def output_dim(self):
        raise NotImplementedError

    def gram(self, X, Y=None):
        """[K(x_i, y_j)] as an (m n_X, m n_Y) matrix ordered component-major: rows (c, i) -> c * n_X + i."""
        Y = X if Y is None else Y
        B = jax.vmap(jax.vmap(self, (None, 0)), (0, None))(X, Y)          # (nX, nY, m, m)
        m = self.output_dim
        return jnp.transpose(B, (2, 0, 3, 1)).reshape(m * X.shape[0], m * Y.shape[0])


class DivergenceFreeKernel(MatrixKernel):
    """2-D divergence-free matrix-valued kernel from a scalar stream-function kernel (module docstring).

    Known Matérn kernels are checked here; wrappers and custom kernels must satisfy the documented smoothness
    precondition themselves.
    """
    stream_kernel: Kernel

    def __init__(self, stream_kernel):
        if isinstance(stream_kernel, (MaternKernel, PeriodicMaternKernel)) and stream_kernel.p_order < 1:
            raise ValueError("DivergenceFreeKernel requires Matérn p >= 1")
        self.stream_kernel = stream_kernel

    @property
    def output_dim(self):
        return 2

    def __call__(self, x, y):
        if jnp.shape(x) != (2,) or jnp.shape(y) != (2,):
            raise ValueError(f"DivergenceFreeKernel takes points in R^2, got {jnp.shape(x)}, {jnp.shape(y)}")
        H = jax.jacfwd(jax.grad(self.stream_kernel, 0), 1)(x, y)          # H[i, j] = d_xi d_yj k
        return jnp.array([[H[1, 1], -H[1, 0]], [-H[0, 1], H[0, 0]]])

    def __str__(self):
        return f"DivFree({self.stream_kernel})"


class IndexedMatrixKernel(Kernel):
    """Scalar kernel k((x, c), (y, c')) = K(x, y)[c, c'] for a MatrixKernel K; the last coordinate of each input is
    the component index (rounded to the nearest integer)."""
    matrix_kernel: MatrixKernel

    def __call__(self, z1, z2):
        z1, z2 = jnp.asarray(z1), jnp.asarray(z2)
        i = jnp.round(jax.lax.stop_gradient(z1[-1])).astype(int)
        j = jnp.round(jax.lax.stop_gradient(z2[-1])).astype(int)
        return self.matrix_kernel(z1[:-1], z2[:-1])[i, j]

    def __str__(self):
        return f"Indexed({self.matrix_kernel})"
