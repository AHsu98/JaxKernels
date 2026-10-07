"""Generic properties of every kernel in the catalog (tests/_helpers.py): symmetry, positive semi-definiteness (also
of derivative functionals), derivatives against finite differences up to the order the smoothness allows (at
generic and at coincident points), hyperparameter gradients against finite differences, structure stability
(no retrace for new hyperparameter values), and jit/vmap consistency."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _helpers import (catalog, extra_cases, derivative_chain, central_difference, gram, raw_leaves_fd_check,
                      count_traces, partial_derivative)

CASES = catalog() + extra_cases()
IDS = [c.name for c in CASES]


def _pairs(case, n, seed):
    X = case.points(n, seed)
    Y = case.points(n, seed + 1)
    return X, Y


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_symmetric(case):
    k = case.kernel(0)
    X, Y = _pairs(case, 8, 0)
    kxy = jax.vmap(k)(X, Y)
    kyx = jax.vmap(k)(Y, X)
    np.testing.assert_allclose(kxy, kyx, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("case", [c for c in CASES if c.psd], ids=[c.name for c in CASES if c.psd])
def test_gram_psd(case):
    k = case.kernel(0)
    X = case.points(25, 3)
    K = np.asarray(gram(k, X))
    np.testing.assert_allclose(K, K.T, rtol=1e-13, atol=1e-15)
    ev = np.linalg.eigvalsh((K + K.T) / 2)
    assert ev[0] >= -1e-10 * max(ev[-1], 1e-300), (ev[0], ev[-1])


def _first_derivative_ops(case):
    from jaxkernels.kerneltools import eval_k
    if case.dim is None:
        return [eval_k, lambda k, i: jax.grad(k, i)]
    coords = sorted(set(case.coords)) if case.coords else range(case.dim)
    return [eval_k] + [lambda k, i, c=c: (lambda *a: jax.grad(k, i)(*a)[c]) for c in coords]


@pytest.mark.parametrize("case", [c for c in CASES if c.psd and (c.order is None or c.order >= 1)],
                         ids=[c.name for c in CASES if c.psd and (c.order is None or c.order >= 1)])
def test_derivative_functional_gram_psd(case):
    """Gram matrix of {values, first partials} at 10 points is PSD: catches sign errors in mixed x/y derivatives."""
    from jaxkernels.kerneltools import get_kernel_block_ops
    k = case.kernel(0)
    X = case.points(10, 4)
    ops = _first_derivative_ops(case)
    K = np.asarray(get_kernel_block_ops(k, ops, ops)(X, X))
    np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-10 * np.abs(K).max())
    ev = np.linalg.eigvalsh((K + K.T) / 2)
    assert ev[0] >= -1e-9 * ev[-1], (ev[0], ev[-1])


def _path(case, length):
    if case.dim is None:
        base = [(0, None), (1, None)] * 8
    else:
        a, b = case.coords or (0, case.dim - 1)
        base = [(0, a), (1, b), (0, b), (1, a), (0, a), (1, a), (0, b), (1, b)] * 2
    return base[:length]


def _check_chain(case, X, Y, path, h, rtol):
    k = case.kernel(0)
    chain = derivative_chain(k, path)
    for n in range(1, len(chain)):
        arg, coord = path[n - 1]
        ad = np.asarray(jax.jit(jax.vmap(chain[n]))(X, Y))
        fd = np.asarray(jax.jit(jax.vmap(lambda x, y: central_difference(chain[n - 1], x, y, arg, coord, h)))(X, Y))
        S = (3.0 / case.scale) ** n
        err = np.max(np.abs(ad - fd) / (np.abs(ad) + S))
        assert err < rtol * case.derivative_tol, f"level {n} ({path[:n]}): rel err {err:.2e}"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_derivatives_generic_points(case):
    """Nested derivatives (x and y alternating, 4 levels) against 4th-order central differences of the level
    below, at distinct points (all kernels here are smooth away from x = y)."""
    X, Y = _pairs(case, 3, 10)
    _check_chain(case, X, Y, _path(case, 4), h=1e-3 * case.scale, rtol=1e-7)


COINCIDENT = [pytest.param(c, marks=pytest.mark.slow) if (c.order or 0) >= 3 else c for c in CASES if c.order != 0]


@pytest.mark.parametrize("case", COINCIDENT, ids=[c.name for c in CASES if c.order != 0])
def test_derivatives_coincident_points(case):
    """The same at x = y, up to order 2 * order (each argument differentiated `order` times; 4 for smooth
    kernels). The top derivative of a Matérn kernel is only Lipschitz at 0, so central differences converge
    at O(h): h = 1e-6 * lengthscale, rtol 1e-4. Orders 6 and 8 (Matérn p >= 3) are marked slow; the exact
    Taylor-coefficient tests in test_matern.py cover them by default."""
    X = case.points(3, 20)
    length = 4 if case.order is None else 2 * case.order
    _check_chain(case, X, X, _path(case, length), h=1e-6 * case.scale, rtol=1e-4)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_hyperparameter_gradients(case):
    """d/d(raw leaves) of a weighted sum of kernel values and (where defined) mixed derivatives d2k/dx dy, at
    distinct points and at a coincident point, against central differences."""
    k0 = case.kernel(0)
    X, Y = _pairs(case, 4, 30)
    Y = Y.at[0].set(X[0])
    w = jnp.linspace(0.5, 1.5, 4)
    dxdy = case.order is None or case.order >= 1
    a, b = case.coords or (0, (case.dim or 1) - 1)
    first = (0, None) if case.dim is None else (0, a)
    second = (1, None) if case.dim is None else (1, b)

    def loss(k):
        val = jnp.sum(w * jax.vmap(k)(X, Y))
        if dxdy:
            d2 = partial_derivative(partial_derivative(k, *first), *second)
            val = val + jnp.sum(w * jax.vmap(d2)(X, Y)) * case.scale**2
        return val

    worst = raw_leaves_fd_check(loss, k0, h=1e-5, rtol=1e-6)
    assert worst < 1.0, worst


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_structure_stable(case):
    """Instances with different hyperparameter values are the same pytree structure: jit traces once."""
    kernels = [case.kernel(i) for i in range(3)]
    defs = [jax.tree_util.tree_structure(k) for k in kernels]
    assert defs[0] == defs[1] == defs[2]
    shapes = [[(leaf.shape, leaf.dtype, getattr(leaf, "weak_type", False)) for leaf in jax.tree_util.tree_leaves(k)]
              for k in kernels]
    assert shapes[0] == shapes[1] == shapes[2]
    X, Y = _pairs(case, 1, 40)
    assert count_traces(kernels, X[0], Y[0]) == 1


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_jit_vmap_consistent(case):
    k = case.kernel(1)
    X, Y = _pairs(case, 5, 50)
    loop = np.array([float(k(x, y)) for x, y in zip(X, Y)])
    np.testing.assert_allclose(np.asarray(jax.vmap(k)(X, Y)), loop, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(np.asarray(jax.jit(jax.vmap(k))(X, Y)), loop, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(np.asarray(eqx.filter_jit(lambda kk, x, y: jax.vmap(kk)(x, y))(k, X, Y)), loop,
                               rtol=1e-13, atol=1e-15)
