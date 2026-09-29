import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from abc import abstractmethod
from jax.nn import softplus


def softplus_inverse(y: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: log(exp(y) - 1) = y + log(-expm1(-y)).

    expm1 keeps full relative precision for small y (the former y + log1p(-exp(-y)) lost it: the round trip
    softplus(softplus_inverse(y)) was off by 2e-5 relative at y = 1e-12 and 1e-9 at 1e-8)."""
    return y + jnp.log(-jnp.expm1(-y))


def is_concrete(x) -> bool:
    """False for values being traced by jit/grad/vmap, where Python comparisons are not possible."""
    return not isinstance(x, jax.core.Tracer)


def as_float_array(x) -> jax.Array:
    """x as a strongly typed array of the default float dtype (float64 when x64 is enabled).

    Hyperparameter leaves are stored this way so that kernels built from Python floats, NumPy scalars or JAX
    arrays are the same pytree for jit: dtype and weak type are part of the cache key, and a Python float gives a
    weakly typed array while a JAX float64 scalar does not, so mixing them compiled twice."""
    return jnp.asarray(x, dtype=jnp.result_type(float))


def check_positive(value, name, minimum=0.0):
    """Raise ValueError if a concrete hyperparameter value is not above `minimum` (lengthscales: >= minimum is
    allowed, as before; positivity: > 0). Traced values are not checked."""
    if not is_concrete(value):
        return
    v = np.asarray(value)
    bad = np.any(v < minimum) if minimum > 0 else np.any(v <= 0)
    if bad:
        what = f"below the minimum {minimum}" if minimum > 0 else "not positive"
        raise ValueError(f"{name} {v} is {what}")


def scaled_sqdist(x, y, lengthscale):
    """sum_i ((x_i - y_i) / lengthscale_i)^2 for scalar or (d,) points; lengthscale scalar or (d,) (ARD)."""
    diff = jnp.asarray(x) - jnp.asarray(y)
    if jnp.ndim(lengthscale) == 0:
        return jnp.sum(diff**2) / lengthscale**2
    if diff.ndim > 1 or diff.size != jnp.shape(lengthscale)[0]:
        raise ValueError(f"ARD lengthscale of shape {jnp.shape(lengthscale)} does not match points of shape "
                         f"{diff.shape}")
    return jnp.sum((diff / lengthscale) ** 2)


def fmt(x, precision=2):
    """Short text for a (possibly array-valued, possibly traced) hyperparameter value."""
    try:
        a = np.asarray(x)
    except Exception:  # traced
        return "?"
    if a.ndim == 0:
        return f"{float(a):.{precision}f}"
    return np.array2string(a, precision=precision, separator=",")


def _is_scalar(other) -> bool:
    try:
        return jnp.ndim(other) == 0 and not isinstance(other, Kernel)
    except Exception:
        return isinstance(other, (int, float))


class Kernel(eqx.Module):
    """Abstract base class for kernels in JAX + Equinox."""

    @abstractmethod
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute k(x, y). Must be overridden by subclasses."""
        pass

    def __add__(self, other: "Kernel"):
        """
        Overload the '+' operator so we can do k1 + k2.
        Internally, we return a SumKernel object containing both.
        Also handles the case if `other` is already a SumKernel, in
        which case we combine everything into one big sum.
        """
        if isinstance(other, SumKernel):
            # Combine self with an existing SumKernel's list
            return SumKernel(*( [self] + list(other.kernels) ))
        elif isinstance(other, Kernel):
            return SumKernel(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        """0 + k = k, so that sum([k1, k2, ...]) works."""
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    def __mul__(self, other: "Kernel"):
        """
        Overload the '*' operator so we can do k1 * k2.
        Handles:
          - Kernel * Kernel -> ProductKernel(self, other)
          - Kernel * ProductKernel -> merge into one ProductKernel
          - Kernel * scalar -> ProductKernel(self, ConstantKernel(scalar)); the scalar may be traced
        """
        if isinstance(other, ProductKernel):
            return ProductKernel(*( [self] + list(other.kernels) ))
        elif isinstance(other, Kernel):
            return ProductKernel(self, other)
        elif _is_scalar(other):
            return ProductKernel(self, ConstantKernel(other))
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Ensure scalar * kernel and Kernel * scalar behave the same way.
        """
        return self.__mul__(other)

    def transform(self,f):
        """
        Creates a transformed kernel, returning a kernel function
        k_transformed(x,y) = k(f(x),f(y))
        """
        return TransformedKernel(self,f)

    def scale(self,c):
        """
        returns a kernel rescaled by a constant factor c
            really should be implemented better
            but the abstract Kernel doesn't include the variances yet
        Thus, we return a product kernel with the constant kernel,
        abusing the __mul__ overloading
        """
        kc = ConstantKernel(c)
        return kc * self


class TransformedKernel(Kernel):
    """
    Transformed kernel, representing the
    composition of a kernel with another
    fixed function

    The transform is structure (a static field): use a module-level function, not a lambda or closure created per
    build (each new function object is new structure, so jit recompiles). For a transform with learnable
    parameters use WarpedKernel.
    """
    kernel: Kernel
    transform: callable = eqx.field(static=True)

    def __init__(self,kernel,transform):
        self.kernel = kernel
        self.transform = transform

    def __call__(self, x, y):
        return self.kernel(self.transform(x),self.transform(y))

    def __str__(self):
        return f"Transformed({self.kernel.__str__()})"


class SumKernel(Kernel):
    """
    Represents the sum of multiple kernels:
      k_sum(x, y) = sum_{k in kernels} k(x, y)
    """
    kernels: tuple[Kernel, ...]

    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return sum(k(x, y) for k in self.kernels)

    def __add__(self, other: "Kernel"):
        """
        If we do (k1 + k2) + k3, the left side is a SumKernel, so
        we define its __add__ to merge again into one SumKernel.
        """
        if isinstance(other, SumKernel):
            return SumKernel(*(list(self.kernels) + list(other.kernels)))
        elif isinstance(other, Kernel):
            return SumKernel(*(list(self.kernels) + [other]))
        else:
            return NotImplemented

    def scale(self,c):
        """
        Push scaling down a level
        """
        return SumKernel(*[k.scale(c) for k in self.kernels])

    def __str__(self):
        component_str = [k.__str__() for k in self.kernels]
        return " + ".join(component_str)


class ProductKernel(Kernel):
    """
    Represents the product of multiple kernels:
      k_prod(x, y) = prod_{k in kernels} k(x, y)
    """
    kernels: tuple[Kernel, ...]

    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.prod(jnp.array([k(x, y) for k in self.kernels]))

    def __mul__(self, other: "Kernel"):
        """
        (k1*k2)*k3: the left side is a ProductKernel, so merge into one flat ProductKernel. (Until 2b015b7 this
        method was named __prod__, which Python never calls, so products nested; and it merged a SumKernel's
        components as factors.)
        """
        if isinstance(other, ProductKernel):
            return ProductKernel(*(list(self.kernels) + list(other.kernels)))
        elif isinstance(other, Kernel):
            return ProductKernel(*(list(self.kernels) + [other]))
        elif _is_scalar(other):
            return ProductKernel(*(list(self.kernels) + [ConstantKernel(other)]))
        else:
            return NotImplemented

    def scale(self,c):
        """
        Scale the first kernel
        """
        return ProductKernel(self.kernels[0].scale(c), *self.kernels[1:])

    def __str__(self):
        component_str = ["(" + k.__str__() + ")" for k in self.kernels]
        return "*".join(component_str)


class FrozenKernel(Kernel):
    """A kernel whose hyperparameters receive no gradient (jax.lax.stop_gradient); hyper.hyperparameter_filter
    also marks its leaves as frozen."""
    kernel:Kernel
    def __init__(self,kernel):
        self.kernel = kernel

    def __call__(self, x, y):
        return jax.lax.stop_gradient(self.kernel)(x, y)

    def __str__(self):
        return self.kernel.__str__()


class ConstantKernel(Kernel):
    """
    Constant kernel k(x, y) = c for all x, y.

    Params:
        variance, variance of the constant shift
    Internally stored as "raw_" after applying softplus_inverse.
    """
    raw_variance: jnp.ndarray

    def __init__(self, variance: float = 1.0):
        """
        :param variance: A positive float specifying the kernel's constant value.
        """
        if is_concrete(variance) and variance <= 0:
            raise ValueError("ConstantKernel requires a strictly positive constant.")
        # Store an unconstrained parameter via softplus-inverse
        self.raw_variance = softplus_inverse(as_float_array(variance))

    @property
    def variance(self):
        return softplus(self.raw_variance)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        v = softplus(self.raw_variance)
        return v

    def scale(self,c):
        return ConstantKernel(c*softplus(self.raw_variance))

    def __str__(self):
        v = softplus(self.raw_variance)
        return fmt(v, 3)


class TensorProductKernel(Kernel):
    """
    Tensor-product (separable) kernel across coordinates.

    If initialized with [k1,...,kd]:
        K(x,y) = Π_i k_i(x[i], y[i])

    If initialized with a single kernel k:
        K(x,y) = Π_i k(x[i], y[i])

    The component kernels are ordinary pytree leaves, so their hyperparameters can be
    changed with eqx.tree_at, differentiated, and passed through jit without retracing.
    """

    _kernels: object

    def __init__(self, kernels):
        if isinstance(kernels, Kernel):
            self._kernels = kernels
            return
        if not isinstance(kernels, (list, tuple)):
            raise TypeError("TensorProductKernel expects a Kernel or a list/tuple of Kernels.")
        if len(kernels) == 0:
            raise ValueError("TensorProductKernel requires at least one kernel.")
        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("TensorProductKernel(list) requires Kernel instances.")
        self._kernels = tuple(kernels)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(
                f"TensorProductKernel expects 1D inputs. Got x.ndim={x.ndim}, y.ndim={y.ndim}."
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"TensorProductKernel expects x,y same length. Got {x.shape[0]} and {y.shape[0]}."
            )
        if isinstance(self._kernels, Kernel):
            return jnp.prod(jax.vmap(self._kernels)(x, y))
        if x.shape[0] != len(self._kernels):
            raise ValueError(
                f"TensorProductKernel initialized with {len(self._kernels)} kernels "
                f"but input dimension is {x.shape[0]}."
            )
        out = self._kernels[0](x[0], y[0])
        for i, k in enumerate(self._kernels[1:], start=1):
            out = out * k(x[i], y[i])
        return out

    def __repr__(self):

        if isinstance(self._kernels, Kernel):
            return f"TensorProductKernel({self._kernels})"

        names = " ⊗ ".join(str(k) for k in self._kernels)
        return f"TensorProductKernel({names})"

    __str__ = __repr__


class WeightedSumKernel(Kernel):
    """
    k(x, y) = sum_j w_j k_j(x, y) with learnable weights: continuous selection among kernel families.

    weights: w_j = softplus(raw_weights_j) (each component's own variance), or, with normalize=True,
    w_j = softplus(raw_j) / sum_i softplus(raw_i) (a convex combination: the total variance stays that of the
    components, which fits func_graph_comp's convention of unit kernel variances with the prior scale in the reg
    weight). Build the components with variance 1 and freeze their variances (hyper.hyperparameter_filter(k,
    exclude="kernels*variance")), or the weights and component variances are one direction.

    Selection: fit the weights by marginal likelihood or CV (jaxkernels.objectives); a family whose weight goes
    to ~0 is deselected. Components must accept the same inputs. With free lengthscales per component the mixture
    is also a multi-scale model, and then the weights are not family labels: on a Matérn-1/2 sample path (300 points)
    the fit put 0.65 on a long-lengthscale RBF (the trend) and 0.35 on the Matérn-1/2, although single-family
    marginal likelihoods rank Matérn-1/2 first by 80 nats (func-keql experiments/hyper/kernels/demos.py D). For
    family selection, give the components one shared lengthscale, or compare single-family fits.
    """
    kernels: tuple
    raw_weights: jax.Array
    normalize: bool = eqx.field(static=True)

    def __init__(self, kernels, weights=None, normalize=False):
        kernels = tuple(kernels)
        if not kernels or not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("WeightedSumKernel takes a non-empty sequence of kernels")
        w = jnp.full((len(kernels),), 1.0 / len(kernels) if normalize else 1.0) if weights is None else weights
        w = as_float_array(w)
        if w.shape != (len(kernels),):
            raise ValueError(f"{len(kernels)} kernels but weights of shape {w.shape}")
        check_positive(w, "weights")
        self.kernels = kernels
        self.raw_weights = softplus_inverse(w)
        self.normalize = bool(normalize)

    @property
    def weights(self):
        w = softplus(self.raw_weights)
        return w / jnp.sum(w) if self.normalize else w

    def __call__(self, x, y):
        w = self.weights
        return sum(w[j] * k(x, y) for j, k in enumerate(self.kernels))

    def __str__(self):
        return " + ".join(f"{fmt(w)}*({k})" for w, k in zip(self.weights, self.kernels))
