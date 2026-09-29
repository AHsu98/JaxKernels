"""Regression tests for the fixes of the ah-hyper audit (B1-B12; see the kernels report in func-keql
experiments/hyper/kernels)."""
import os
import subprocess
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxkernels import (GaussianRBFKernel, RationalQuadraticKernel, ScalarMaternKernel, MaternKernel, ProductKernel,
                        SumKernel, PolynomialKernel, SpectralMixtureKernel, ConstantKernel, softplus_inverse)
from _helpers import count_traces


def test_B1_matern_construction_is_cheap_and_structure_is_shared():
    t0 = time.perf_counter()
    kernels = [ScalarMaternKernel(p, 0.3 + 0.1 * i) for p in range(5) for i in range(3)]
    assert time.perf_counter() - t0 < 0.5
    for p in range(5):
        defs = {jax.tree_util.tree_structure(k) for k in kernels[3 * p:3 * p + 3]}
        assert len(defs) == 1
        assert eqx.tree_equal(kernels[3 * p], eqx.tree_at(lambda k: k.raw_lengthscale, kernels[3 * p],
                                                          kernels[3 * p].raw_lengthscale))
    assert count_traces([ScalarMaternKernel(2, ls) for ls in (0.2, 0.3, 0.4)], jnp.array(0.0), jnp.array(0.1)) == 1
    assert hash(jax.tree_util.tree_structure(ScalarMaternKernel(2, 0.2))) == \
        hash(jax.tree_util.tree_structure(ScalarMaternKernel(2, 0.9)))


def test_B2_matern_p0_is_differentiable():
    k = ScalarMaternKernel(0, 0.3)
    g = eqx.filter_grad(lambda kk: kk(0.1, 0.4))(k)
    ls = 0.3
    # k = exp(-|x - y| / ls): d/dls = |x-y|/ls^2 k, and d ls / d raw = sigmoid(raw)
    expected = 0.3 / ls**2 * np.exp(-0.3 / ls) * float(jax.nn.sigmoid(k.raw_lengthscale))
    assert float(g.raw_lengthscale) == pytest.approx(expected, rel=1e-12)
    assert float(jax.grad(k, 0)(0.1, 0.4)) == pytest.approx(np.exp(-1.0) / ls, rel=1e-12)


def test_B3_scalar_matern_on_length_one_points():
    k = ScalarMaternKernel(2, 0.3)
    x, y = jnp.array([0.1]), jnp.array([0.35])
    assert jnp.shape(k(x, y)) == ()
    assert float(jax.grad(k, 0)(x, y)[0]) == pytest.approx(float(jax.grad(k, 0)(0.1, 0.35)), rel=1e-14)
    with pytest.raises(ValueError):
        k(jnp.zeros(2), jnp.zeros(2))


@pytest.mark.parametrize("make", [lambda: ScalarMaternKernel(1, 0.005), lambda: RationalQuadraticKernel(0.005),
                                  lambda: GaussianRBFKernel(0.005), lambda: MaternKernel(2, jnp.array([0.5, 0.001])),
                                  lambda: GaussianRBFKernel(0.3, -1.0), lambda: RationalQuadraticKernel(0.3, -2.0)])
def test_B4_invalid_values_raise(make):
    with pytest.raises(ValueError):
        make()


def test_B4_traced_values_are_not_checked():
    x, y = jnp.array([0.0]), jnp.array([0.5])
    val = jax.jit(lambda ls: MaternKernel(2, ls)(x, y))(0.3)
    assert np.isfinite(float(val))
    assert GaussianRBFKernel(0.01).lengthscale == pytest.approx(0.01)       # equal to the minimum is allowed


def test_B5_leaf_types_do_not_depend_on_input_types():
    ks = [GaussianRBFKernel(0.3), GaussianRBFKernel(jnp.float64(0.4)), GaussianRBFKernel(np.float64(0.5)),
          GaussianRBFKernel(jnp.asarray(0.6)), GaussianRBFKernel(1)]
    assert count_traces(ks, jnp.ones(2), jnp.zeros(2)) == 1
    for k in ks:
        assert all(not leaf.weak_type and leaf.dtype == jnp.float64 for leaf in jax.tree_util.tree_leaves(k))


def test_B6_products_merge_flat():
    k1, k2, k3, k4 = (GaussianRBFKernel(ls) for ls in (0.3, 0.4, 0.5, 0.6))
    p = (k1 * k2) * k3
    assert isinstance(p, ProductKernel) and len(p.kernels) == 3
    assert len(((k1 * k2) * (k3 * k4)).kernels) == 4
    s = k3 + k4
    q = (k1 * k2) * s
    assert len(q.kernels) == 3 and isinstance(q.kernels[2], SumKernel)     # a sum stays one factor
    x, y = jnp.array([0.1, 0.2]), jnp.array([0.4, -0.1])
    assert float(q(x, y)) == pytest.approx(float(k1(x, y) * k2(x, y) * (k3(x, y) + k4(x, y))), rel=1e-14)
    assert len(((k1 * k2) * 2.0).kernels) == 3


def test_B7_multiplication_by_traced_scalar():
    k = GaussianRBFKernel(0.3)
    x, y = jnp.zeros(2), jnp.ones(2) * 0.2
    val = jax.jit(lambda c: (k * c)(x, y))(2.0)
    assert float(val) == pytest.approx(2.0 * float(k(x, y)), rel=1e-14)
    assert float(jax.grad(lambda c: (c * k)(x, y))(2.0)) == pytest.approx(float(k(x, y)), rel=1e-12)


def test_B8_str():
    assert str(PolynomialKernel(2.0, 1.0, 3)) == "2.00Poly(1.00,3)"
    assert "SpecMix(n=3)" in str(SpectralMixtureKernel(jax.random.PRNGKey(0), 3))
    assert str(GaussianRBFKernel(jnp.array([0.3, 0.5]))) == "1.00GRBF([0.3,0.5])"
    assert str(GaussianRBFKernel(0.3) + ConstantKernel(2.0)) == "1.00GRBF(0.30) + 2.000"


@pytest.mark.parametrize("v", [1e-300, 1e-12, 1e-8, 1e-4, 1.0, 30.0, 800.0])
def test_B9_softplus_inverse_round_trip(v):
    r = softplus_inverse(jnp.array(v))
    assert float(jax.nn.softplus(r)) == pytest.approx(v, rel=4e-15)


def test_B10_package_parses_on_python_310():
    """No f-string nesting of the same quotes (a SyntaxError before Python 3.12)."""
    import ast
    pkg = Path(__file__).resolve().parents[1] / "jaxkernels"
    for f in pkg.glob("*.py"):
        src = f.read_text()
        ast.parse(src, feature_version=(3, 10))


def test_B12_sum_builtin():
    ks = [GaussianRBFKernel(ls) for ls in (0.3, 0.4, 0.5)]
    s = sum(ks)
    assert isinstance(s, SumKernel) and len(s.kernels) == 3


@pytest.mark.parametrize("env,expected", [({}, "True True"), ({"JAX_ENABLE_X64": "0"}, "False False"),
                                          ({"JAX_ENABLE_X64": "1"}, "True False")])
def test_x64_policy(env, expected):
    """Importing jaxkernels enables x64 unless JAX_ENABLE_X64 is set (then the user's choice stands)."""
    code = "import jax, jaxkernels; print(jax.config.jax_enable_x64, jaxkernels.X64_SET_BY_JAXKERNELS)"
    full_env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    full_env.update(env)
    full_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run([sys.executable, "-c", code], env=full_env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == expected


def test_B13_selected_derivative_out_of_range_raises():
    from jaxkernels.kerneltools import dx_k, dt_k, get_kernel_block_ops, eval_k, partial_op
    k = GaussianRBFKernel(0.3)
    X1 = jnp.linspace(0, 1, 4)[:, None]                  # 1-D points: there is no coordinate 1
    with pytest.raises(IndexError):
        jax.jit(lambda X: get_kernel_block_ops(k, [eval_k, dx_k], [eval_k])(X, X))(X1)
    K = get_kernel_block_ops(k, [dt_k, dx_k], [eval_k])(jnp.ones((3, 2)), jnp.zeros((3, 2)))
    assert K.shape == (6, 3)
    np.testing.assert_allclose(get_kernel_block_ops(k, [partial_op(0)], [eval_k])(X1, X1),
                               get_kernel_block_ops(k, [dt_k], [eval_k])(X1, X1))    # coordinate 0 is fine


def test_B14_matern_built_inside_jit_through_nested_filter_jit():
    """func-keql audit a15: a Matérn built inside jax.jit and differentiated inside a nested eqx.filter_jit leaked
    sympy2jax constants as tracers ("No constant handler for type DynamicJaxprTracer")."""
    def inner(k):
        return eqx.filter_jit(lambda k: jax.grad(jax.grad(lambda x: k(x, jnp.asarray(0.5))))(jnp.asarray(0.1)))(k)
    val = jax.jit(lambda l: inner(ScalarMaternKernel(2, l)))(0.3)
    ref = inner(ScalarMaternKernel(2, 0.3))
    assert float(val) == pytest.approx(float(ref), rel=1e-14)
    g = jax.jit(jax.grad(lambda l: inner(MaternKernel(3, l))))(0.3)
    fd = (inner(MaternKernel(3, 0.3 + 1e-6)) - inner(MaternKernel(3, 0.3 - 1e-6))) / 2e-6
    assert float(g) == pytest.approx(float(fd), rel=1e-6)
