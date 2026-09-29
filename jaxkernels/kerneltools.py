from functools import partial
from types import ModuleType
from typing import Any
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad

def diagpart(M):
    return jnp.diag(jnp.diag(M))

def vectorize_kfunc(k):
    return jax.vmap(jax.vmap(k, in_axes=(None, 0)), in_axes=(0, None))


def op_k_apply(k: Callable[[float, float], float], L_op, R_op):
    return R_op(L_op(k, 0), 1)


def make_block(k, L_op, R_op):
    return vectorize_kfunc(op_k_apply(k, L_op, R_op))


def get_kernel_block_ops(
    k, ops_left, ops_right, output_dim=1, type_pkg: ModuleType = jnp
):
    def k_super(x, y):
        I_mat = type_pkg.eye(output_dim)
        blocks = [
            [
                type_pkg.kron(make_block(k, L_op, R_op)(x, y), I_mat)
                for R_op in ops_right
            ]
            for L_op in ops_left
        ]
        return type_pkg.block(blocks)

    return k_super


def eval_k(k, index):
    return k


def diff_k(k, index):
    return grad(k, index)


def diff2_k(k, index):
    return grad(grad(k, index), index)


def get_selected_grad(k, index, selected_index):
    gradf = grad(k, index)

    def selgrad(*args):
        g = gradf(*args)
        # a static index past the end would be clamped silently under jit (dx_k on 1-D points gave d/dx_0)
        if jnp.ndim(g) != 1 or selected_index >= g.shape[0]:
            raise IndexError(f"derivative in coordinate {selected_index} of points of shape {jnp.shape(g)}; "
                             "dt_k/dx_k/dxx_k use the (t, x) convention (coordinates 0, 1); use partial_op(i) "
                             "for points in R^d or derivative_op(n) for scalar points")
        return g[selected_index]

    return selgrad


def dx_k(k, index):
    return get_selected_grad(k, index, 1)


def dxx_k(k, index):
    return get_selected_grad(get_selected_grad(k, index, 1), index, 1)


def dt_k(k, index):
    return get_selected_grad(k, index, 0)


def nth_derivative_1d(k: Callable, index: int, n: int) -> Callable:
    """
    Computes derivative of order n of k with respect to index and returns the resulting
    function as a callable
    """
    result = k
    for _ in range(n):
        result = jax.grad(result, argnums=index)
    return result


def nth_derivative_operator_1d(n):
    """
    Computes the operator associated to the nth derivative, which maps functions to
    functions. These now match the format of the operators defined above, like diff_k,
    diff2_k.
    """
    return partial(nth_derivative_1d, n=n)


# ---------------------------------------------------------------------------------------------------------------
# Functionals as cached plain functions (ah-hyper). A functional op(k, index) is structure wherever it is stored
# (func_graph_comp's space.operators, objectives.Observations): the same arguments must give the same function
# object, or jit recompiles. functools.partial objects and lambdas made per call are new objects each time; these
# factories are cached.
# ---------------------------------------------------------------------------------------------------------------
from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=None)
def partial_op(*coords):
    """The functional d^n / dx_c1 ... dx_cn for points x in R^d (coords: coordinate indices, repeats allowed):
    partial_op(1) is d/dx_1, partial_op(0, 0) is d^2/dx_0^2, partial_op(0, 1) the mixed second derivative."""
    def op(k, index):
        f = k
        for c in coords:
            f = get_selected_grad(f, index, c)
        return f
    op.__name__ = op.__qualname__ = "d_" + "_".join(str(c) for c in coords) if coords else "eval"
    return op


@lru_cache(maxsize=None)
def derivative_op(n):
    """The functional d^n / dx^n for scalar points (cached nth_derivative_operator_1d)."""
    def op(k, index):
        return nth_derivative_1d(k, index, n)
    op.__name__ = op.__qualname__ = f"d{n}"
    return op


def laplacian(k, index):
    """Trace of the Hessian in argument `index` (same as func_graph_comp.util.laplacian)."""
    def lapk(*x):
        return jnp.trace(jax.hessian(k, argnums=index)(*x))
    return lapk


@lru_cache(maxsize=None)
def linear_combination_op(*terms):
    """sum_j c_j L_j for terms ((c_1, L_1), (c_2, L_2), ...) with Python-float coefficients (structure), e.g.
    linear_combination_op((1.0, dt_k), (-0.1, dxx_k)). For coefficients that are hyperparameters, write a
    module-level functional that reads them from the kernel (see physics.heat_operator)."""
    def op(k, index):
        parts = [(c, L(k, index)) for c, L in terms]
        return lambda *x: sum(c * g(*x) for c, g in parts)
    return op
