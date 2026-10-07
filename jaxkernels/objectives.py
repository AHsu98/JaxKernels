"""Data-only hyperparameter objectives: Gaussian-process regression on observations of linear functionals.

    y_i = (L_i u)(x_i) + e_i,    u ~ GP(0, k),    e ~ N(0, diag(noise))

Observations come in groups of (functional, points): Observations([(eval_k, X0), (dx_k, X1), ...]). A functional is
a kerneltools-style operator op(k, index) -> function (eval_k, dx_k, partial_op(i), laplacian, ...). The covariance
of the observations is

    C = K + diag(noise) + jitter * diag(K),    K = [L_a L_b' k (X_a, X_b)]  (blocks over groups a, b)

with the jitter relative per row, so it suits mixed value/derivative rows. Each objective factors C once by
Cholesky and never forms C^-1. All are plain traceable functions of (kernel, noise, ...), so they can be jitted and
differentiated with respect to the kernel pytree and the noise:

    neg_log_marginal_likelihood(kernel, noise, obs, y)  -log N(y | 0, C)  (total, not per observation)
    loo_residuals(kernel, noise, obs, y)                exact leave-one-out residuals and variances
    loo_mse, loo_nlpd                                   mean squared LOO residual; mean negative log predictive
                                                        density of the held-out y_i
    kfold_residuals(kernel, noise, obs, y, folds)       exact K-fold residuals
    kfold_mse, kfold_nlpd
    posterior(kernel, noise, obs, y, targets)           posterior mean and variance of target functionals

noise: a scalar, one value per group (tuple/list), or an (n,) array. y: (n,) or (n, m) for m independent outputs
sharing the kernel (the objectives sum over outputs).

Pitfalls:
- A derivative of order m on both arguments needs a kernel 2m times differentiable at x = y (Matérn p >= m);
  beyond that the values are finite but wrong.
- Functionals and folds are static structure: use module-level or cached functionals (kerneltools.partial_op),
  not lambdas created per call, and close over folds when jitting, or jit recompiles. Under eqx.filter_jit, pass a
  changing scalar noise as an array, since Python floats are static. Without jit, nested derivatives are slow.
- For nearly noise-free data the LOO and K-fold objectives, like brute-force refits, lose accuracy as the condition
  number of C grows.
"""
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular

from .kerneltools import make_block, op_k_apply


class Observations(eqx.Module):
    """Groups of (functional, points); rows are ordered by group, then by point."""
    ops: tuple = eqx.field(static=True)
    points: tuple

    def __init__(self, groups):
        groups = list(groups)
        self.ops = tuple(op for op, _ in groups)
        self.points = tuple(jnp.asarray(X) for _, X in groups)

    @classmethod
    def product(cls, ops, X):
        """Every functional at every point."""
        return cls([(op, X) for op in ops])

    @property
    def sizes(self):
        return tuple(int(X.shape[0]) for X in self.points)

    def __len__(self):
        return sum(self.sizes)

    def gram(self, kernel, other=None):
        """[L_a L_b' k (X_a, Y_b)] over the groups of self (rows) and other (columns; default self)."""
        other = self if other is None else other
        return jnp.block([[make_block(kernel, La, Lb)(Xa, Xb) for Lb, Xb in zip(other.ops, other.points)]
                          for La, Xa in zip(self.ops, self.points)])

    def gram_diagonal(self, kernel):
        """diag(self.gram(kernel)) without forming the matrix."""
        return jnp.concatenate([jax.vmap(lambda x, f=op_k_apply(kernel, L, L): f(x, x))(X)
                                for L, X in zip(self.ops, self.points)])

    def noise_diagonal(self, noise):
        """Per-row noise variances from a scalar, one value per group, or an (n,) array."""
        if isinstance(noise, (tuple, list)):
            if len(noise) != len(self.ops):
                raise ValueError(f"{len(noise)} noise values for {len(self.ops)} groups")
            return jnp.concatenate([jnp.full((n,), v, dtype=jnp.result_type(float))
                                    for n, v in zip(self.sizes, noise)])
        return jnp.broadcast_to(jnp.asarray(noise, dtype=jnp.result_type(float)), (len(self),))


def _as_obs(obs):
    return obs if isinstance(obs, Observations) else Observations([(_eval, obs)])


def _eval(k, index):
    return k


def covariance(kernel, noise, obs, jitter=1e-10):
    """C = K + diag(noise) + jitter * diag(K) (K symmetrized)."""
    obs = _as_obs(obs)
    K = obs.gram(kernel)
    K = 0.5 * (K + K.T)
    return K + jnp.diag(obs.noise_diagonal(noise) + jitter * jnp.diag(K))


def _cholesky(kernel, noise, obs, jitter):
    return jnp.linalg.cholesky(covariance(kernel, noise, obs, jitter))


def neg_log_marginal_likelihood(kernel, noise, obs, y, jitter=1e-10):
    """-log N(y | 0, C) = 1/2 y^T C^-1 y + 1/2 log det C + n/2 log 2 pi (summed over the columns of y)."""
    L = _cholesky(kernel, noise, obs, jitter)
    a = solve_triangular(L, y, lower=True)
    m = 1 if jnp.ndim(y) == 1 else y.shape[1]
    n = L.shape[0]
    return 0.5 * jnp.sum(a**2) + m * jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * n * m * jnp.log(2 * jnp.pi)


def _inverse_factor(kernel, noise, obs, jitter):
    L = _cholesky(kernel, noise, obs, jitter)
    return solve_triangular(L, jnp.eye(L.shape[0], dtype=L.dtype), lower=True)       # L^-1, C^-1 = L^-T L^-1


def loo_residuals(kernel, noise, obs, y, jitter=1e-10):
    """(e, var): e_i = y_i - E[y_i | y_-i] = [C^-1 y]_i / [C^-1]_ii and var_i = Var[y_i | y_-i] = 1 / [C^-1]_ii."""
    Li = _inverse_factor(kernel, noise, obs, jitter)
    alpha = Li.T @ (Li @ y)
    dinv = jnp.sum(Li**2, axis=0)
    var = 1.0 / dinv
    return alpha * (var if jnp.ndim(y) == 1 else var[:, None]), var


def loo_mse(kernel, noise, obs, y, jitter=1e-10):
    e, _ = loo_residuals(kernel, noise, obs, y, jitter)
    return jnp.mean(e**2)


def loo_nlpd(kernel, noise, obs, y, jitter=1e-10):
    """Mean over held-out entries of -log N(y_i | E[y_i | y_-i], Var[y_i | y_-i])."""
    e, var = loo_residuals(kernel, noise, obs, y, jitter)
    if jnp.ndim(y) == 2:
        var = jnp.broadcast_to(var[:, None], e.shape)
    return jnp.mean(0.5 * jnp.log(2 * jnp.pi * var) + 0.5 * e**2 / var)


def kfold_indices(n, n_folds, seed=None):
    """Folds as a list of index arrays: contiguous blocks, or a random partition if seed is given."""
    idx = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(idx, n_folds)]


def kfold_residuals(kernel, noise, obs, y, folds: Sequence, jitter=1e-10):
    """[(F, e_F, A_F)] per fold F: e_F = y_F - E[y_F | y_-F] = A_F^-1 [C^-1 y]_F with A_F = [C^-1]_FF, and
    Cov[y_F | y_-F] = A_F^-1. Folds are index arrays (NumPy, static)."""
    obs = _as_obs(obs)
    Li = _inverse_factor(kernel, noise, obs, jitter)
    alpha = Li.T @ (Li @ y)
    out = []
    for F in folds:
        F = np.asarray(F)
        if (F.ndim != 1 or not np.issubdtype(F.dtype, np.integer) or len(np.unique(F)) != len(F)
                or np.any((F < 0) | (F >= len(obs)))):
            raise ValueError(f"invalid fold indices {F}")
        B = Li[:, F]
        A = B.T @ B
        cA = jnp.linalg.cholesky(A)
        out.append((F, cho_solve((cA, True), alpha[F]), A))
    return out


def kfold_mse(kernel, noise, obs, y, folds, jitter=1e-10):
    res = kfold_residuals(kernel, noise, obs, y, folds, jitter)
    return sum(jnp.sum(e**2) for _, e, _ in res) / sum(e.size for _, e, _ in res)


def kfold_nlpd(kernel, noise, obs, y, folds, jitter=1e-10):
    """Mean over folds and outputs of -log N(y_F | E[y_F | y_-F], Cov[y_F | y_-F]), divided by the fold sizes
    (i.e. per held-out entry)."""
    total, count = 0.0, 0
    for F, e, A in kfold_residuals(kernel, noise, obs, y, folds, jitter):
        cA = jnp.linalg.cholesky(A)
        m = 1 if e.ndim == 1 else e.shape[1]
        quad = jnp.sum(e * (A @ e))
        total = total + 0.5 * m * len(F) * jnp.log(2 * jnp.pi) - m * jnp.sum(jnp.log(jnp.diag(cA))) + 0.5 * quad
        count += len(F) * m
    return total / count


def posterior(kernel, noise, obs, y, targets, jitter=1e-10):
    """(mean, var) of the target functionals (an Observations; noise-free) given the observations y."""
    obs = _as_obs(obs)
    targets = _as_obs(targets)
    L = _cholesky(kernel, noise, obs, jitter)
    Ks = targets.gram(kernel, obs)
    mean = Ks @ cho_solve((L, True), y)
    V = solve_triangular(L, Ks.T, lower=True)
    var = targets.gram_diagonal(kernel) - jnp.sum(V**2, axis=0)
    return mean, var
