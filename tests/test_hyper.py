"""Named hyperparameter access (jaxkernels.hyper)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxkernels as jk
from jaxkernels import (GaussianRBFKernel, MaternKernel, TensorProductKernel, FrozenKernel, PolynomialKernel,
                        PeriodicKernel, RationalQuadraticKernel)
from jaxkernels.hyper import (hyperparameters, with_hyperparameters, hyperparameter_filter, describe, select,
                              positive_names, to_log_vector, from_log_vector)
from _helpers import count_traces


def composite():
    return (TensorProductKernel([GaussianRBFKernel(0.3), FrozenKernel(MaternKernel(2, jnp.array([0.4])))])
            + PolynomialKernel(c=2.0) + PeriodicKernel(1.0, jnp.array([0.2, 0.3])))


def test_names_and_values():
    k = composite()
    h = hyperparameters(k)
    assert list(h) == ["kernels[0]._kernels[0].variance", "kernels[0]._kernels[0].lengthscale",
                       "kernels[0]._kernels[1].kernel.variance", "kernels[0]._kernels[1].kernel.lengthscale",
                       "kernels[1].variance", "kernels[1].c", "kernels[2].variance", "kernels[2].lengthscale",
                       "kernels[2].period"]
    assert float(h["kernels[0]._kernels[0].lengthscale"]) == pytest.approx(0.3, rel=1e-14)
    assert float(h["kernels[1].c"]) == 2.0
    np.testing.assert_allclose(h["kernels[2].lengthscale"], [0.2, 0.3], rtol=1e-14)
    assert "frozen" in describe(k).splitlines()[3]
    assert list(hyperparameters(GaussianRBFKernel(0.3))) == ["variance", "lengthscale"]


def test_set_by_name_equals_construction():
    k = RationalQuadraticKernel(0.3, 2.0, 1.5)
    k2 = with_hyperparameters(k, {"lengthscale": 0.45, "alpha": 0.7})
    ref = RationalQuadraticKernel(0.45, 0.7, 1.5)
    x, y = jnp.array([0.1, 0.2]), jnp.array([0.5, -0.3])
    assert float(k2(x, y)) == pytest.approx(float(ref(x, y)), rel=1e-14)
    assert jax.tree_util.tree_structure(k2) == jax.tree_util.tree_structure(k)
    # scalars broadcast to ARD leaves
    a = with_hyperparameters(MaternKernel(2, jnp.array([0.3, 0.5])), {"lengthscale": 0.7})
    np.testing.assert_allclose(a.lengthscale, [0.7, 0.7], rtol=1e-14)


def test_errors():
    k = GaussianRBFKernel(0.3)
    with pytest.raises(KeyError):
        with_hyperparameters(k, {"length": 0.3})
    with pytest.raises(ValueError):
        with_hyperparameters(k, {"lengthscale": 0.001})          # below min_lengthscale
    with pytest.raises(ValueError, match="above its minimum"):
        with_hyperparameters(k, {"lengthscale": k.min_lengthscale})
    with pytest.raises(ValueError):
        with_hyperparameters(k, {"variance": -1.0})
    with pytest.raises(ValueError):
        with_hyperparameters(MaternKernel(2, jnp.array([0.3, 0.5])), {"lengthscale": jnp.ones(3)})


def test_traceable_and_no_retrace():
    k = composite()
    x, y = jnp.array([0.1, 0.2]), jnp.array([0.5, -0.3])
    name = "kernels[0]._kernels[0].lengthscale"

    def f(ls):
        return with_hyperparameters(k, {name: ls})(x, y)
    g = jax.grad(f)(0.3)
    fd = (f(0.3 + 1e-6) - f(0.3 - 1e-6)) / 2e-6
    assert float(g) == pytest.approx(float(fd), rel=1e-7)
    kernels = [with_hyperparameters(k, {name: v}) for v in (0.3, jnp.float64(0.4), np.float32(0.5))]
    assert count_traces(kernels, x, y) == 1


def test_filter_partition_and_frozen():
    k = composite()
    filt = hyperparameter_filter(k, exclude=["*variance"])
    trainable, static = eqx.partition(k, filt)
    names = select(k, exclude=["*variance"])
    assert names == ["kernels[0]._kernels[0].lengthscale", "kernels[1].c", "kernels[2].lengthscale",
                     "kernels[2].period"]                                   # frozen Matérn excluded
    x, y = jnp.array([0.1, 0.2]), jnp.array([0.5, -0.3])
    g = jax.grad(lambda t: eqx.combine(t, static)(x, y))(trainable)
    n_grad_leaves = len(jax.tree_util.tree_leaves(g))
    assert n_grad_leaves == 4
    assert select(k, include="kernels[0]._kernels[1].*", include_frozen=True) == [
        "kernels[0]._kernels[1].kernel.variance", "kernels[0]._kernels[1].kernel.lengthscale"]
    assert select(k, include="*[2].p?riod") == ["kernels[2].period"]          # brackets literal, ? wildcard


def test_log_vector_round_trip_and_gradient():
    k = composite()
    names = positive_names(k)
    assert "kernels[0]._kernels[1].kernel.lengthscale" not in names
    z = to_log_vector(k, names)
    assert z.shape == (7,)                  # 4 scalars + the (2,) periodic lengthscale + the period
    k2 = from_log_vector(k, names, z)
    for n, v in hyperparameters(k).items():
        np.testing.assert_allclose(hyperparameters(k2)[n], v, rtol=1e-13)
    x, y = jnp.array([0.1, 0.2]), jnp.array([0.5, -0.3])
    f = lambda z: from_log_vector(k, names, z)(x, y)
    g = jax.grad(f)(z)
    for i in range(z.shape[0]):
        e = jnp.zeros_like(z).at[i].set(1e-6)
        assert float(g[i]) == pytest.approx(float((f(z + e) - f(z - e)) / 2e-6), rel=1e-6, abs=1e-12)
    with pytest.raises(ValueError):
        to_log_vector(k, ["kernels[1].c"])
    with pytest.raises(ValueError):
        from_log_vector(k, names, z[:3])


def test_log_coordinates_have_no_invalid_region():
    """z = log(value - minimum): values stay above the lengthscale floor for every z (a line search cannot produce
    NaN), and z = log(value) when there is no floor."""
    k = GaussianRBFKernel(0.3)                                   # min_lengthscale 0.01
    names = positive_names(k)
    z = to_log_vector(k, names)
    np.testing.assert_allclose(z, [0.0, np.log(0.3 - 0.01)], rtol=1e-14, atol=1e-15)
    for zl in (-50.0, -5.0, 0.0, 3.0):
        kk = from_log_vector(k, names, jnp.array([0.0, zl]))
        assert float(kk.lengthscale) == pytest.approx(0.01 + np.exp(zl), rel=1e-12)
        assert np.isfinite(float(kk(jnp.zeros(1), jnp.ones(1))))
    g = jax.grad(lambda z: from_log_vector(k, names, z)(jnp.zeros(1), 0.1 * jnp.ones(1)))(jnp.array([0.0, -8.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    k0 = GaussianRBFKernel(0.3, min_lengthscale=0.0)
    np.testing.assert_allclose(to_log_vector(k0), [0.0, np.log(0.3)], rtol=1e-14, atol=1e-15)


def test_top_level_exports():
    assert jk.hyperparameters is hyperparameters and jk.with_hyperparameters is with_hyperparameters
