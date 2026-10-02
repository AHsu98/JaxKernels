import jax
import jax.numpy as jnp
from jax.nn import softplus
import equinox as eqx
from .matern import matern_phi
from .base_kernels import (Kernel, softplus_inverse, is_concrete, as_float_array, check_positive, scaled_sqdist,
                           fmt)


class TranslationInvariantKernel(Kernel):
    """
    Not used for anything yet, but maybe unifies some of the other kernels
    Kernels defined by k(x,y) = var * h( (x-y)/ls )
    """
    core_func:callable
    raw_variance: jax.Array
    raw_lengthscale: jax.Array

    min_lengthscale: jax.Array = eqx.field(static=True)
    fix_variance:bool = eqx.field(static=True)
    fix_lengthscale:bool = eqx.field(static=True)

    def __init__(
            self,
            core_func,
            lengthscale,
            variance,
            min_lengthscale,
            fix_variance = False,
            fix_lengthscale = False,
            ):
        self.raw_variance = softplus_inverse(jnp.array(variance))
        if is_concrete(lengthscale) and lengthscale <= min_lengthscale:
            raise ValueError("Initial lengthscale must be above minimum")
        self.raw_lengthscale = softplus_inverse(jnp.array(lengthscale) - min_lengthscale)
        self.min_lengthscale = min_lengthscale
        self.fix_variance = fix_variance
        self.fix_lengthscale = fix_lengthscale
        self.core_func = core_func

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        if self.fix_variance is True:
            var = jax.lax.stop_gradient(var)

        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        if self.fix_lengthscale is True:
            ls = jax.lax.stop_gradient(ls)

        scaled_diff = (y-x)/ls
        return var*self.core_func(scaled_diff)


class _StationaryKernel(Kernel):
    """Shared fields of the lengthscale/variance kernels: softplus-positive raw leaves, lengthscale >
    min_lengthscale (a static float). The lengthscale may be a scalar (isotropic) or a (d,) array (ARD: one
    lengthscale per input coordinate)."""
    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    min_lengthscale: float = eqx.field(static=True)

    def _init_scales(self, lengthscale, variance, min_lengthscale):
        lengthscale, variance = as_float_array(lengthscale), as_float_array(variance)
        check_positive(lengthscale, "lengthscale", min_lengthscale)
        check_positive(variance, "variance")
        self.raw_variance = softplus_inverse(variance)
        self.raw_lengthscale = softplus_inverse(lengthscale - min_lengthscale)
        self.min_lengthscale = float(min_lengthscale)

    @property
    def variance(self):
        return softplus(self.raw_variance)

    @property
    def lengthscale(self):
        return softplus(self.raw_lengthscale) + self.min_lengthscale

    def scale(self, c):
        new_raw_var = softplus_inverse(c*softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)


class MaternKernel(_StationaryKernel):
    """
    Half-integer Matérn kernel on scalars or R^d, radial in the lengthscale-scaled distance, nu = p + 1/2:

        k(x, y) = variance * phi_p(s),   s = sum_i ((x_i - y_i) / lengthscale_i)^2

    phi_p(s) = exp(-z) p!/(2p)! sum_{i=0}^p (p+i)!/(i!(p-i)!) (2z)^(p-i), z = sqrt(2 nu s) (Rasmussen & Williams
    4.16): p = 0 exponential, 1: (1 + z) e^-z, 2: (1 + z + z^2/3) e^-z. lengthscale: scalar, or (d,) for ARD.

    Smoothness: k is 2p times differentiable at x = y (and analytic elsewhere), so an operator of order m applied
    to both arguments needs 2m <= 2p: the Laplacian needs p >= 2, third derivatives p >= 3. Derivatives of every
    order up to 2p are exact at x = y through a closed-form custom JVP (see matern.py); higher ones are not
    defined there. The RKHS on R^d is norm-equivalent to the Sobolev space H^(nu + d/2).
    """
    p_order: int = eqx.field(static=True)

    def __init__(self, p, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        if int(p) != p or p < 0:
            raise ValueError(f"p must be a nonnegative integer (nu = p + 1/2), got {p}")
        self.p_order = int(p)
        self._init_scales(lengthscale, variance, min_lengthscale)

    @property
    def nu(self):
        return self.p_order + 0.5

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        return var * matern_phi(self.p_order, 0, scaled_sqdist(x, y, ls))

    def __str__(self):
        return f"{fmt(self.variance)}Matern({self.p_order},{fmt(self.lengthscale)})"


class ScalarMaternKernel(MaternKernel):
    """
    Scalar half-integer order matern kernel
    order = p+(1/2)

    Parameters:
        p: int
        variance > 0
        lengthscale > 0
    Internally stored as "raw_" after applying softplus_inverse.

    The Matérn kernel of MaternKernel restricted to scalar inputs (shape () or (1,)); for points in R^d use
    MaternKernel (radial, optionally ARD) or TensorProductKernel of ScalarMaternKernels (separable). Since
    ah-hyper: closed form (no sympy), the same structure for every instance of a given p (jit does not retrace),
    differentiable for p = 0, and scalar output for shape-(1,) inputs; values agree with the former sympy
    implementation to 3.3e-16 and derivatives up to order 2p to 3.4e-13 relative.
    """

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        if jnp.size(x) != 1 or jnp.size(y) != 1:
            raise ValueError(f"ScalarMaternKernel takes scalar inputs (shape () or (1,)), got {jnp.shape(x)} and "
                             f"{jnp.shape(y)}; use MaternKernel or TensorProductKernel for points in R^d")
        return super().__call__(x, y)

    @property
    def core_matern(self):
        """The Matérn profile as a function of the scaled difference d = (y - x) / lengthscale (the former
        static field of the same name)."""
        p = self.p_order
        return lambda d: matern_phi(p, 0, d * d)


class GaussianRBFKernel(_StationaryKernel):
    """
    RBF (squared exponential) kernel:
        k(x, y) = variance * exp(-||x - y||^2 / (2*lengthscale^2))

    Parameters:
        variance > 0
        lengthscale > 0: a scalar, or a (d,) array for ARD, k = variance * exp(-sum_i (x_i - y_i)^2 / (2 l_i^2))
    Internally stored as "raw_" after applying softplus_inverse.
    """

    def __init__(self, lengthscale=1.0,variance=1.0,min_lengthscale = 0.01):
        self._init_scales(lengthscale, variance, min_lengthscale)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale)+self.min_lengthscale
        if jnp.ndim(ls) == 0:           # the original expression: bitwise identical values for scalar lengthscales
            sqdist = jnp.sum((x - y) ** 2)
            return var * jnp.exp(-0.5 * sqdist / (ls**2))
        return var * jnp.exp(-0.5 * scaled_sqdist(x, y, ls))

    def __str__(self):
        return f"{fmt(self.variance)}GRBF({fmt(self.lengthscale)})"


class RationalQuadraticKernel(_StationaryKernel):
    """
    Rational Quadratic kernel:
      k(x, y) = variance * [1 + (||x - y||^2 / (2 * alpha * lengthscale^2))]^(-alpha)

    Parameters:
        variance > 0
        lengthscale > 0: a scalar, or a (d,) array for ARD (||x - y||^2 / l^2 -> sum_i (x_i - y_i)^2 / l_i^2)
        alpha > 0
    Internally stored as "raw_" after applying softplus_inverse.
    """
    raw_alpha: jax.Array

    def __init__(self, lengthscale=1.0, alpha=1.0,variance=1.0,min_lengthscale = 0.01):
        self._init_scales(lengthscale, variance, min_lengthscale)
        alpha = as_float_array(alpha)
        check_positive(alpha, "alpha")
        self.raw_alpha = softplus_inverse(alpha)

    @property
    def alpha(self):
        return softplus(self.raw_alpha)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        a = softplus(self.raw_alpha)
        if jnp.ndim(ls) == 0:           # the original expression: bitwise identical values for scalar lengthscales
            sqdist = jnp.sum((x - y) ** 2)
            factor = 1.0 + (sqdist / (2.0 * a * ls**2))
        else:
            factor = 1.0 + scaled_sqdist(x, y, ls) / (2.0 * a)
        return var * jnp.power(factor, -a)

    def __str__(self):
        return f"{fmt(self.variance)}RQ({fmt(self.alpha)},{fmt(self.lengthscale)})"


class SpectralMixtureKernel(Kernel):
    """
    Spectral Mixture kernel for scalar inputs:
      k(x, y) = sum_{m=1..M} w_m * exp(-2 * (pi*sigma_m)^2 * (x-y)^2) * cos(2 pi (x-y) * periods_m)
    where tau = x - y.

    Note: `periods` are frequencies (cycles per unit of x), unconstrained.
    Internally stored as "raw_" after applying softplus_inverse.
    """
    raw_weights: jnp.ndarray
    raw_freq_sigmas: jnp.ndarray
    periods: jnp.ndarray

    def __init__(
            self,
            key,
            num_mixture=20,
            period_variance = 10.
            ):
        key1, key2, key3 = jax.random.split(key, 3)
        self.raw_weights = jax.random.normal(key1, shape=(num_mixture,))
        self.raw_freq_sigmas = jax.random.normal(key2, shape=(num_mixture,))
        self.periods = jnp.sqrt(period_variance)*jax.random.normal(key3, shape=(num_mixture,))

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        tau = x - y
        weights = softplus(self.raw_weights)
        freq_sigmas = softplus(self.raw_freq_sigmas)

        kernel_components = (
            jnp.exp(-2.0 * (jnp.pi * freq_sigmas)**2 * tau**2)
            *jnp.cos(2.0 * jnp.pi * tau * self.periods)
        )
        return jnp.sum(weights * kernel_components)

    def scale(self, c):
        new_raw_weights = softplus_inverse(c*softplus(self.raw_weights))
        return eqx.tree_at(lambda x: x.raw_weights, self, new_raw_weights)

    def __str__(self):
        weights = softplus(self.raw_weights)
        return f"{fmt(jnp.sum(weights))}SpecMix(n={len(self.periods)})"


class LinearKernel(Kernel):
    """
    Linear Kernel k(x, y) = v* <x,y>

    Params:
        variance, variance
    Internally stored as "raw_" after applying softplus_inverse.
    """
    raw_variance: jnp.ndarray

    def __init__(self, variance: float = 1.0):
        """
        :param constant: A positive float specifying the kernel's variance
        """
        if is_concrete(variance) and variance <= 0:
            raise ValueError("LinearKernel requires a strictly positive constant.")
        # Store an unconstrained parameter via softplus-inverse
        self.raw_variance = softplus_inverse(as_float_array(variance))

    @property
    def variance(self):
        return softplus(self.raw_variance)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        v = softplus(self.raw_variance)  # guaranteed positive
        return v*jnp.dot(x,y)

    def scale(self, c):
        new_raw_var = softplus_inverse(c*softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __str__(self):
        v = softplus(self.raw_variance)
        return f"{fmt(v)}Lin()"


class PolynomialKernel(Kernel):
    """
    Polynomial Kernel k(x, y) = v * (<x,y>+c)^p

    Params:
        variance, variance
        c: offset, stored unconstrained. The kernel is positive semi-definite only for c >= 0 (for degree >= 1);
           a gradient-based fit can make it negative.
    Internally stored as "raw_" after applying softplus_inverse.
    """
    raw_variance: jnp.ndarray
    degree:int = eqx.field(static=True)
    c: jnp.ndarray

    def __init__(self, variance: float = 1.0,c:float = 1.,degree: int = 2):
        if is_concrete(variance) and variance <= 0:
            raise ValueError("PolynomialKernel requires a strictly positive constant.")
        self.raw_variance = softplus_inverse(as_float_array(variance))
        self.c = as_float_array(c)
        self.degree = degree

    @property
    def variance(self):
        return softplus(self.raw_variance)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        v = softplus(self.raw_variance)  # guaranteed positive
        return v*jnp.pow(jnp.dot(x,y)+self.c,self.degree)

    def scale(self, c):
        new_raw_var = softplus_inverse(c*softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __str__(self):
        v = softplus(self.raw_variance)  # guaranteed positive
        return f"{fmt(v)}Poly({fmt(self.c)},{self.degree})"
