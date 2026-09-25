"""Regression tests for kernel hyperparameter handling. Run with pytest, or: python tests/test_kernel_fixes.py"""
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from jaxkernels import (GaussianRBFKernel, RationalQuadraticKernel, ProductKernel, TensorProductKernel,
                        TransformedKernel, softplus_inverse)


def rbf(x, y, ls):
    return jnp.exp(-0.5 * jnp.sum((x - y) ** 2) / ls**2)


def test_tensor_product_matches_product_of_components():
    k = TensorProductKernel([GaussianRBFKernel(0.3), GaussianRBFKernel(0.7)])
    x, y = jnp.array([0.1, 0.4]), jnp.array([0.3, -0.2])
    assert jnp.allclose(k(x, y), rbf(x[:1], y[:1], 0.3) * rbf(x[1:], y[1:], 0.7))
    k1 = TensorProductKernel(GaussianRBFKernel(0.3))
    assert jnp.allclose(k1(x, y), rbf(x[:1], y[:1], 0.3) * rbf(x[1:], y[1:], 0.3))


def test_tensor_product_hyperparameters_are_live_leaves():
    k = TensorProductKernel([GaussianRBFKernel(0.3), GaussianRBFKernel(0.7)])
    x, y = jnp.array([0.1, 0.4]), jnp.array([0.3, -0.2])
    k2 = eqx.tree_at(lambda k: k._kernels[1].raw_lengthscale, k, softplus_inverse(jnp.array(0.5) - 0.01))
    assert jnp.allclose(k2(x, y), rbf(x[:1], y[:1], 0.3) * rbf(x[1:], y[1:], 0.5))
    g = eqx.filter_grad(lambda k: k(x, y))(k)
    assert jnp.abs(g._kernels[0].raw_lengthscale) > 0 and jnp.abs(g._kernels[1].raw_lengthscale) > 0


def test_tensor_product_does_not_retrace_for_new_hyperparameters():
    traces = []

    @eqx.filter_jit
    def f(k, x):
        traces.append(1)
        return k(x, x + 0.1)

    for ls in [0.2, 0.3, 0.4]:
        f(TensorProductKernel([GaussianRBFKernel(ls), TransformedKernel(GaussianRBFKernel(ls), jnp.sin)]),
          jnp.ones(2))
    assert len(traces) == 1


def test_constructors_accept_traced_values():
    x, y = jnp.array([0.0]), jnp.array([0.5])
    val = jax.jit(lambda ls: GaussianRBFKernel(ls)(x, y))(0.3)
    assert jnp.allclose(val, rbf(x, y, 0.3))
    g = jax.grad(lambda ls: GaussianRBFKernel(ls)(x, y))(0.3)
    assert jnp.allclose(g, jax.grad(lambda ls: rbf(x, y, ls))(0.3))
    try:
        GaussianRBFKernel(0.001)
        raise AssertionError("expected ValueError for a lengthscale below the minimum")
    except ValueError:
        pass


def test_rational_quadratic_lengthscale():
    k = RationalQuadraticKernel(lengthscale=0.2, alpha=1.0)
    x, y = jnp.array([0.0]), jnp.array([0.3])
    assert jnp.allclose(k(x, y), (1 + 0.3**2 / (2 * 0.2**2)) ** -1.0)


def test_product_kernel_scale():
    k = ProductKernel(GaussianRBFKernel(0.3), GaussianRBFKernel(0.5))
    x, y = jnp.array([0.0]), jnp.array([0.2])
    assert jnp.allclose(k.scale(2.0)(x, y), 2.0 * k(x, y))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
