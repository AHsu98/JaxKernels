import jax
import jax.numpy as jnp
from jax.nn import softplus

from jaxopt import LBFGS

from .base_kernels import as_float_array, check_positive, softplus_inverse
from .kerneltools import vectorize_kfunc
from .tree_opt import run_gradient_descent, run_jaxopt_solver


SIGMA2_FLOOR = 1e-6


def _raw_noise_variance(init_sigma2):
    """Map a requested total variance to its raw excess above ``SIGMA2_FLOOR``."""
    init_sigma2 = as_float_array(init_sigma2)
    check_positive(init_sigma2, "init_sigma2", SIGMA2_FLOOR)
    return softplus_inverse(init_sigma2 - SIGMA2_FLOOR)


def noise_variance(params):
    """The floored noise variance stored in a parameter dict. Every builder uses this mapping so fitted and
    reported variances agree."""
    return softplus(params['transformed_sigma2']) + SIGMA2_FLOOR


def _output_count(y):
    if jnp.ndim(y) == 1:
        return 1
    if jnp.ndim(y) == 2:
        return y.shape[1]
    raise ValueError("y must be either a 1 or two dimensional array")


def _neg_marglike(K, y, sigma2, m):
    C = jax.scipy.linalg.cholesky(K + sigma2 * jnp.eye(len(K)), lower=True)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(C)))
    yTKinvY = jnp.sum(jax.scipy.linalg.solve_triangular(C, y, lower=True) ** 2)
    return m * logdet + yTKinvY


def build_neg_marglike(X, y):
    """Objective 2 * (-log N(y | 0, K + sigma2 I)) - n m log(2 pi) = m log det(K + sigma2 I) + tr(Y^T (K + sigma2 I)^-1 Y)
    for y of shape (n,) or (n, m) (m independent outputs sharing the kernel)."""
    m = _output_count(y)

    def loss(params):
        K = vectorize_kfunc(params['kernel'])(X, X)
        return _neg_marglike(K, y, noise_variance(params), m)

    return loss


def build_neg_marglike_partialobs(t, y, v):
    m = _output_count(y)

    def loss(params):
        K = vectorize_kfunc(params['kernel'])(t, t) * (v @ v.T)
        return _neg_marglike(K, y, noise_variance(params), m)

    return loss


def build_loocv(X, y):
    """Mean squared leave-one-out residual of GP regression: e_i = [C^-1 y]_i / [C^-1]_ii, C = K + sigma2 I
    (the prediction of y_i from the other points is y_i - e_i). Cholesky-based: diag(C^-1) is the column norms
    of L^-1 (until 2b015b7: jnp.linalg.inv and the algebraically equal K P y - diag(K P)/diag(P) * P y)."""
    def loss(params):
        K = vectorize_kfunc(params['kernel'])(X, X)
        L = jnp.linalg.cholesky(K + noise_variance(params) * jnp.eye(len(X)))
        Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(len(X)), lower=True)
        alpha = Linv.T @ (Linv @ y)
        diag_Cinv = jnp.sum(Linv**2, axis=0)
        e = alpha / (diag_Cinv if jnp.ndim(y) == 1 else diag_Cinv[:, None])
        return jnp.mean(e**2)

    return loss


def _build_split_obj(X, y, train_idx, val_idx):
    Xtrain = X[train_idx]
    ytrain = y[train_idx]
    Xval = X[val_idx]
    yval = y[val_idx]

    def loss(params):
        kernel = params['kernel']
        K = vectorize_kfunc(kernel)(Xtrain, Xtrain)
        c = jnp.linalg.solve(K + noise_variance(params) * jnp.eye(len(ytrain)), ytrain)
        ypred = vectorize_kfunc(kernel)(Xval, Xtrain) @ c
        return jnp.mean((ypred - yval) ** 2)

    return loss

def build_random_split_obj(X, y, p=0.2, rng_key=None):
    """Mean squared prediction error on a fixed random train-validation split."""
    n = X.shape[0]
    if rng_key is None:
        rng_key = jax.random.key(1)
    perm = jax.random.permutation(rng_key, n)
    n_val = int(jnp.round(p * n))
    return _build_split_obj(X, y, perm[n_val:], perm[:n_val])


def build_every_other_obj(X, y):
    return _build_split_obj(X, y, slice(None, None, 2), slice(1, None, 2))


def _fit(loss, init_kernel, init_sigma2, gd_tol, lbfgs_tol, max_gd_iter, max_lbfgs_iter, show_progress):
    params = {
        'kernel': init_kernel,
        'transformed_sigma2': _raw_noise_variance(init_sigma2),
    }
    params, history_gd = run_gradient_descent(
        loss, params, tol=gd_tol, maxiter=max_gd_iter, show_progress=show_progress, init_stepsize=1e-4
    )
    solver = LBFGS(loss, maxiter=max_lbfgs_iter, tol=lbfgs_tol)
    params, history_bfgs, _ = run_jaxopt_solver(solver, params, show_progress=show_progress)
    return params['kernel'], noise_variance(params), [history_gd, history_bfgs]

def fit_kernel(
        init_kernel,
        init_sigma2,
        X,
        y,
        loss_builder=build_neg_marglike,
        gd_tol=1e-4,
        lbfgs_tol=1e-6,
        max_gd_iter=3000,
        max_lbfgs_iter=1000,
        show_progress=True,
        ):
    return _fit(loss_builder(X, y), init_kernel, init_sigma2, gd_tol, lbfgs_tol, max_gd_iter, max_lbfgs_iter,
                show_progress)

def fit_kernel_partialobs(
        init_kernel,
        init_sigma2,
        t, y, v,
        gd_tol=1e-4,
        lbfgs_tol=1e-6,
        max_gd_iter=3000,
        max_lbfgs_iter=1000,
        show_progress=True,
        ):
    return _fit(build_neg_marglike_partialobs(t, y, v), init_kernel, init_sigma2, gd_tol, lbfgs_tol, max_gd_iter,
                max_lbfgs_iter, show_progress)
