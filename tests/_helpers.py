"""Shared test helpers: finite differences, nested derivatives, and a catalog of kernels with their smoothness."""
from dataclasses import dataclass, field
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxkernels.kerneltools import vectorize_kfunc


# ------------------------------------------------------------------------------------------------------------
# derivatives
# ------------------------------------------------------------------------------------------------------------
def partial_derivative(f, arg, coord):
    """(x, y) -> d f(x, y) / d arg[coord] (arg 0 = x, 1 = y; coord None for scalar points)."""
    g = jax.grad(f, argnums=arg)
    if coord is None:
        return g
    return lambda x, y: g(x, y)[coord]


def derivative_chain(f, path):
    """[f, D_1 f, D_2 D_1 f, ...] for a path of (arg, coord) directions."""
    out = [f]
    for arg, coord in path:
        out.append(partial_derivative(out[-1], arg, coord))
    return out


def central_difference(f, x, y, arg, coord, h):
    """Fourth-order central difference of f(x, y) in the direction (arg, coord)."""
    def shifted(s):
        v = x if arg == 0 else y
        v = v + s if coord is None else v.at[coord].add(s)
        return f(v, y) if arg == 0 else f(x, v)
    return (-shifted(2 * h) + 8 * shifted(h) - 8 * shifted(-h) + shifted(-2 * h)) / (12 * h)


def gram(k, X, Y=None):
    return vectorize_kfunc(k)(X, X if Y is None else Y)


def raw_leaves_fd_check(loss, kernel, h=1e-6, rtol=1e-6, atol=1e-10):
    """Compare eqx.filter_grad(loss)(kernel) with central differences in every array leaf entry; returns the
    worst relative discrepancy."""
    g = eqx.filter_jit(eqx.filter_grad(loss))(kernel)
    loss_jit = eqx.filter_jit(loss)
    leaves, treedef = jax.tree_util.tree_flatten(eqx.filter(kernel, eqx.is_array))
    gleaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
    worst = 0.0
    for li, (leaf, gl) in enumerate(zip(leaves, gleaves)):
        flat = np.asarray(leaf, dtype=float).ravel()
        for j in range(flat.size):
            def at(delta):
                new = flat.copy()
                new[j] += delta
                new_leaves = list(leaves)
                new_leaves[li] = jnp.asarray(new.reshape(np.shape(leaf)), dtype=leaf.dtype)
                params = jax.tree_util.tree_unflatten(treedef, new_leaves)
                return float(loss_jit(eqx.combine(params, eqx.filter(kernel, eqx.is_array, inverse=True))))
            fd = (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)
            ad = float(np.asarray(gl).ravel()[j])
            err = abs(ad - fd) / (atol + rtol * max(abs(ad), abs(fd), 1.0))
            worst = max(worst, err)
    return worst


def count_traces(kernels, x, y):
    """Number of times eqx.filter_jit traces a function of the kernel when called with each kernel in turn."""
    traces = []

    @eqx.filter_jit
    def f(k, x, y):
        traces.append(1)
        return k(x, y)

    for k in kernels:
        f(k, x, y)
    return len(traces)


# ------------------------------------------------------------------------------------------------------------
# kernel catalog
# ------------------------------------------------------------------------------------------------------------
@dataclass
class Case:
    """A kernel family for the generic tests.

    make(v) builds the kernel from hyperparameter values v (a dict); values[i] are three different settings.
    dim: None for scalar inputs, else d. order: derivative order available per argument at coincident points
    (None = smooth). scale: a typical lengthscale (finite-difference steps, tolerance scales).
    """
    name: str
    make: Callable
    values: list
    dim: Optional[int]
    order: Optional[int] = None
    scale: float = 0.5
    psd: bool = True
    derivative_tol: float = 1.0          # multiplies the default derivative tolerances
    domain: tuple = (0.0, 1.0)
    coords: Optional[tuple] = None       # the two coordinates used in derivative paths (default (0, dim - 1))
    sampler: Optional[Callable] = None   # sampler(rng, n) -> points, instead of uniform on domain
    extra: dict = field(default_factory=dict)

    def kernel(self, i=0):
        return self.make(self.values[i])

    def points(self, n, seed=0):
        rng = np.random.default_rng(seed)
        if self.sampler is not None:
            return jnp.asarray(self.sampler(rng, n))
        lo, hi = self.domain
        shape = (n,) if self.dim is None else (n, self.dim)
        return jnp.asarray(rng.uniform(lo, hi, size=shape))


def _transform(x):                          # module-level: stable structure for TransformedKernel
    return jnp.stack([x[0], x[0] + x[1] ** 2])


def catalog():
    from jaxkernels import (GaussianRBFKernel, RationalQuadraticKernel, ScalarMaternKernel, MaternKernel, LinearKernel,
                            PolynomialKernel, ConstantKernel, SpectralMixtureKernel, TensorProductKernel,
                            TransformedKernel)
    a = jnp.asarray
    cases = [
        Case("rbf", lambda v: GaussianRBFKernel(v["ls"], v["var"]),
             [dict(ls=0.4, var=1.3), dict(ls=0.7, var=0.8), dict(ls=a(0.2), var=a(2.0))], dim=2),
        Case("rbf_scalar", lambda v: GaussianRBFKernel(v["ls"], v["var"]),
             [dict(ls=0.4, var=1.3), dict(ls=0.7, var=0.8), dict(ls=0.3, var=1.0)], dim=None),
        Case("rbf_aniso", lambda v: GaussianRBFKernel(a(v["ls"]), v["var"]),
             [dict(ls=[0.3, 0.8], var=1.3), dict(ls=[0.5, 0.5], var=0.8), dict(ls=[1.0, 0.2], var=1.0)], dim=2),
        Case("rq", lambda v: RationalQuadraticKernel(v["ls"], v["alpha"], v["var"]),
             [dict(ls=0.4, alpha=1.5, var=1.3), dict(ls=0.6, alpha=0.7, var=1.0), dict(ls=0.3, alpha=3.0, var=2.0)],
             dim=2),
        Case("rq_aniso", lambda v: RationalQuadraticKernel(a(v["ls"]), v["alpha"], v["var"]),
             [dict(ls=[0.3, 0.8], alpha=1.5, var=1.3), dict(ls=[0.6, 0.4], alpha=0.7, var=1.0),
              dict(ls=[0.2, 0.9], alpha=3.0, var=2.0)], dim=2),
        Case("linear", lambda v: LinearKernel(v["var"]),
             [dict(var=1.3), dict(var=0.5), dict(var=2.0)], dim=2, psd=True),
        Case("poly3", lambda v: PolynomialKernel(v["var"], v["c"], degree=3),
             [dict(var=1.3, c=0.5), dict(var=0.5, c=1.0), dict(var=2.0, c=2.0)], dim=2),
        Case("constant", lambda v: ConstantKernel(v["var"]),
             [dict(var=1.3), dict(var=0.5), dict(var=2.0)], dim=2),
        Case("specmix", lambda v: eqx.tree_at(lambda k: k.periods, SpectralMixtureKernel(jax.random.PRNGKey(0), 3),
                                              a(v["freq"])),
             [dict(freq=[0.5, 1.0, 2.0]), dict(freq=[0.3, 1.5, 0.1]), dict(freq=[1.0, 1.0, 1.0])], dim=None),
        Case("sum", lambda v: GaussianRBFKernel(v["ls"]) + MaternKernel(2, a(v["ard"])),
             [dict(ls=0.4, ard=[0.5, 0.7]), dict(ls=0.6, ard=[0.3, 0.9]), dict(ls=0.2, ard=[1.0, 1.0])], dim=2,
             order=2),
        Case("product", lambda v: GaussianRBFKernel(v["ls"]) * RationalQuadraticKernel(v["ls2"], 2.0),
             [dict(ls=0.4, ls2=0.8), dict(ls=0.6, ls2=0.5), dict(ls=0.9, ls2=0.3)], dim=2),
        Case("scaled", lambda v: GaussianRBFKernel(v["ls"]) * v["c"],
             [dict(ls=0.4, c=2.0), dict(ls=0.6, c=a(0.5)), dict(ls=0.9, c=3.0)], dim=2),
        Case("tensor_matern", lambda v: TensorProductKernel([ScalarMaternKernel(2, v["l1"]),
                                                             ScalarMaternKernel(3, v["l2"])]),
             [dict(l1=0.4, l2=0.6), dict(l1=0.3, l2=0.8), dict(l1=0.9, l2=0.2)], dim=2, order=2),
        Case("tensor_single", lambda v: TensorProductKernel(GaussianRBFKernel(v["ls"])),
             [dict(ls=0.4), dict(ls=0.6), dict(ls=0.9)], dim=2),
        Case("transformed", lambda v: TransformedKernel(GaussianRBFKernel(v["ls"]), _transform),
             [dict(ls=0.4), dict(ls=0.6), dict(ls=0.9)], dim=2),
    ]
    for p in range(5):
        cases.append(Case(f"matern_scalar_p{p}", lambda v, p=p: ScalarMaternKernel(p, v["ls"], v["var"]),
                          [dict(ls=0.4, var=1.3), dict(ls=0.7, var=0.8), dict(ls=a(0.25), var=a(1.0))],
                          dim=None, order=p))
        cases.append(Case(f"matern_aniso_p{p}", lambda v, p=p: MaternKernel(p, a(v["ls"]), v["var"]),
                          [dict(ls=[0.4, 0.7], var=1.3), dict(ls=[0.6, 0.3], var=0.8), dict(ls=[1.0, 1.0], var=1.0)],
                          dim=2, order=p))
    cases.append(Case("matern_iso_d3_p2", lambda v: MaternKernel(2, v["ls"], v["var"]),
                      [dict(ls=0.5, var=1.3), dict(ls=0.7, var=0.8), dict(ls=0.3, var=1.0)], dim=3, order=2))
    return cases


def extra_cases():
    """Kernels added on ah-hyper beyond the original catalog."""
    from jaxkernels import PeriodicKernel, PeriodicMaternKernel, TensorProductKernel, MaternKernel
    a = jnp.asarray
    cases = [
        Case("periodic", lambda v: PeriodicKernel(v["P"], v["ls"], v["var"]),
             [dict(P=1.0, ls=0.3, var=1.3), dict(P=0.7, ls=0.5, var=0.8), dict(P=a(2.0), ls=a(0.2), var=a(1.0))],
             dim=None, scale=0.3, domain=(0.0, 2.0)),
        Case("periodic_aniso", lambda v: PeriodicKernel(a(v["P"]), a(v["ls"])),
             [dict(P=[1.0, 2.0], ls=[0.3, 0.6]), dict(P=[0.5, 1.0], ls=[0.4, 0.2]), dict(P=[2.0, 2.0], ls=[1.0, 1.0])],
             dim=2, scale=0.3, domain=(0.0, 2.0)),
        Case("tensor_time_periodic", lambda v: TensorProductKernel([MaternKernel(2, v["lt"]),
                                                                    PeriodicKernel(v["P"], v["lx"])]),
             [dict(lt=0.5, P=1.0, lx=0.3), dict(lt=0.3, P=2.0, lx=0.4), dict(lt=0.8, P=1.5, lx=0.2)], dim=2,
             order=2, scale=0.3),
    ]
    for p in (1, 2, 3):
        cases.append(Case(f"periodic_matern_p{p}", lambda v, p=p: PeriodicMaternKernel(p, v["P"], v["ls"], v["var"]),
                          [dict(P=1.0, ls=0.3, var=1.3), dict(P=0.7, ls=0.5, var=0.8), dict(P=2.0, ls=0.2, var=1.0)],
                          dim=None, order=p, scale=0.3, domain=(0.0, 2.0)))
    from jaxkernels import (HeatKernel, DivergenceFreeKernel, IndexedMatrixKernel, WarpedKernel, TanhWarp,
                            WeightedSumKernel, GaussianRBFKernel, RationalQuadraticKernel)

    def indexed_points(rng, n):
        return np.column_stack([rng.uniform(0, 1, (n, 2)), rng.integers(0, 2, n).astype(float)])
    cases += [
        Case("heat_1d", lambda v: HeatKernel(v["ls"], v["kappa"], v["var"]),
             [dict(ls=0.3, kappa=0.1, var=1.3), dict(ls=0.5, kappa=0.05, var=0.8), dict(ls=0.2, kappa=a(0.3), var=1.0)],
             dim=2, scale=0.3),
        Case("heat_2d_aniso", lambda v: HeatKernel(a(v["ls"]), a(v["kappa"])),
             [dict(ls=[0.3, 0.5], kappa=[0.1, 0.02]), dict(ls=[0.4, 0.4], kappa=[0.05, 0.05]),
              dict(ls=[0.2, 0.6], kappa=[0.2, 0.1])], dim=3, scale=0.3),
        Case("divfree_indexed", lambda v: IndexedMatrixKernel(DivergenceFreeKernel(GaussianRBFKernel(v["ls"]))),
             [dict(ls=0.4), dict(ls=0.6), dict(ls=0.3)], dim=3, scale=0.4, coords=(0, 1), sampler=indexed_points),
        Case("divfree_indexed_matern", lambda v: IndexedMatrixKernel(DivergenceFreeKernel(MaternKernel(3, v["ls"]))),
             [dict(ls=0.4), dict(ls=0.6), dict(ls=0.3)], dim=3, order=2, scale=0.4, coords=(0, 1),
             sampler=indexed_points),
        Case("warped_tanh", lambda v: WarpedKernel(GaussianRBFKernel(v["ls"]),
                                                  TanhWarp(v["c"], v["A"], v["s"], axis=1)),
             [dict(ls=0.3, c=0.5, A=0.05, s=0.05), dict(ls=0.4, c=0.3, A=0.1, s=0.1), dict(ls=0.2, c=a(0.6), A=0.02,
                                                                                          s=0.03)], dim=2, scale=0.05),
        Case("warped_moving", lambda v: WarpedKernel(MaternKernel(2, a(v["ls"])),
                                                    TanhWarp(a(v["c"]), 0.05, 0.05, axis=1, velocities=a(v["v"]),
                                                             time_axis=0)),
             [dict(ls=[0.3, 0.3], c=[0.5, 0.2], v=[0.3, -0.1]), dict(ls=[0.2, 0.5], c=[0.4, 0.8], v=[0.0, 0.2]),
              dict(ls=[0.6, 0.3], c=[0.1, 0.9], v=[1.0, 1.0])], dim=2, order=2, scale=0.05),
        Case("weighted_sum", lambda v: WeightedSumKernel([GaussianRBFKernel(0.3), RationalQuadraticKernel(0.5, 2.0),
                                                          MaternKernel(2, 0.4)], a(v["w"])),
             [dict(w=[0.5, 0.3, 0.2]), dict(w=[1.0, 1.0, 1.0]), dict(w=[0.01, 2.0, 0.1])], dim=2, order=2),
        Case("weighted_sum_normalized", lambda v: WeightedSumKernel([GaussianRBFKernel(0.3), MaternKernel(1, 0.4)],
                                                                    a(v["w"]), normalize=True),
             [dict(w=[0.5, 0.5]), dict(w=[0.9, 0.1]), dict(w=[0.2, 3.0])], dim=2, order=1),
    ]
    return cases
