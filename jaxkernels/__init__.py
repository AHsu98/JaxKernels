import os as _os

import jax as _jax

# Enable float64 (kernel matrices are too ill-conditioned for float32 Cholesky factorizations) unless the user has
# chosen: if JAX_ENABLE_X64 is set in the environment, the setting is left alone.
X64_SET_BY_JAXKERNELS = "JAX_ENABLE_X64" not in _os.environ and not _jax.config.jax_enable_x64
if X64_SET_BY_JAXKERNELS:
    _jax.config.update("jax_enable_x64", True)

from .kernels import (  # noqa: E402
    GaussianRBFKernel,
    MaternKernel,
    ScalarMaternKernel,
    RationalQuadraticKernel,
    SpectralMixtureKernel,
    LinearKernel,
    PolynomialKernel
)
from .base_kernels import (Kernel, softplus_inverse, ConstantKernel, SumKernel, ProductKernel,  # noqa: E402
                           TransformedKernel, TensorProductKernel, FrozenKernel, WeightedSumKernel)
from .fit_kernel import fit_kernel,build_loocv,build_neg_marglike,fit_kernel_partialobs  # noqa: E402
from .periodic import PeriodicKernel, PeriodicMaternKernel  # noqa: E402
from .physics import (HeatKernel, heat_residual_op, MatrixKernel, DivergenceFreeKernel,  # noqa: E402
                      IndexedMatrixKernel)
from .warped import TanhWarp, WarpedKernel  # noqa: E402
from . import hyper, objectives  # noqa: E402
from .hyper import (hyperparameters, with_hyperparameters, hyperparameter_filter, describe,  # noqa: E402
                    to_log_vector, from_log_vector)

__all__ = [
    "Kernel",
    "ConstantKernel",
    "SumKernel",
    "ProductKernel",
    "TransformedKernel",
    "TensorProductKernel",
    "FrozenKernel",
    "WeightedSumKernel",
    "GaussianRBFKernel",
    "MaternKernel",
    "ScalarMaternKernel",
    "RationalQuadraticKernel",
    "LinearKernel",
    "PolynomialKernel",
    "SpectralMixtureKernel",
    "PeriodicKernel",
    "PeriodicMaternKernel",
    "HeatKernel",
    "heat_residual_op",
    "MatrixKernel",
    "DivergenceFreeKernel",
    "IndexedMatrixKernel",
    "TanhWarp",
    "WarpedKernel",
    "objectives",
    "hyper",
    "hyperparameters",
    "with_hyperparameters",
    "hyperparameter_filter",
    "describe",
    "to_log_vector",
    "from_log_vector",
    "fit_kernel",
    "build_loocv",
    "build_neg_marglike",
    "softplus_inverse",
    "fit_kernel_partialobs"
]