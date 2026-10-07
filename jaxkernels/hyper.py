"""Named access to kernel hyperparameters.

Every array leaf is a hyperparameter. A leaf stored as ``raw_<name>`` is positive, with value
``softplus(raw) + min_<name>``; other array leaves are unconstrained. Names are pytree paths with the ``raw_`` prefix
dropped. Leaves inside a FrozenKernel are excluded from filters and log vectors by default.

Include and exclude patterns are globs with only ``*`` and ``?`` special; brackets are literal. The accessors work on
any pytree and preserve its leaf order and structure, so parameter replacement remains traceable.

Log coordinates use ``z = log(value - minimum)`` and ``value = minimum + exp(z)``. Every coordinate is valid, so an
optimizer cannot step below a positive parameter's floor.
"""
import re
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import softplus

from .base_kernels import FrozenKernel, as_float_array, is_concrete, softplus_inverse


def _softplus_inverse_exp(z):
    """softplus_inverse(exp(z)), using its asymptote before exp(z) can underflow."""
    cutoff = jnp.log(jnp.finfo(z.dtype).eps)
    safe_z = jnp.where(z < cutoff, cutoff, z)
    return jnp.where(z < cutoff, z, softplus_inverse(jnp.exp(safe_z)))


@dataclass(frozen=True)
class Hyperparameter:
    name: str
    index: int          # position in jax.tree_util.tree_leaves(tree)
    shape: tuple
    positive: bool      # value = softplus(raw) + minimum; else value = leaf
    minimum: float
    frozen: bool        # inside a FrozenKernel

    def value(self, leaf):
        return softplus(leaf) + self.minimum if self.positive else leaf

    def raw(self, value):
        return softplus_inverse(value - self.minimum) if self.positive else value


def _key_str(key):
    if isinstance(key, jax.tree_util.GetAttrKey):
        return "." + key.name
    if isinstance(key, jax.tree_util.SequenceKey):
        return f"[{key.idx}]"
    if isinstance(key, jax.tree_util.DictKey):
        return f"[{key.key!r}]"
    return f"[{key}]"


def _child(obj, key):
    if isinstance(key, jax.tree_util.GetAttrKey):
        return getattr(obj, key.name)
    if isinstance(key, jax.tree_util.SequenceKey):
        return obj[key.idx]
    if isinstance(key, jax.tree_util.DictKey):
        return obj[key.key]
    return None


def hyperparameter_info(tree):
    """[Hyperparameter] for every array leaf of tree, in pytree order."""
    out = []
    for index, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
        if not eqx.is_array(leaf):
            continue
        obj, frozen = tree, isinstance(tree, FrozenKernel)
        for key in path[:-1]:
            obj = _child(obj, key)
            frozen = frozen or isinstance(obj, FrozenKernel)
        parts = [_key_str(key) for key in path]
        last = path[-1] if path else None
        positive, minimum = False, 0.0
        if isinstance(last, jax.tree_util.GetAttrKey) and last.name.startswith("raw_"):
            public = last.name[4:]
            positive = True
            minimum = float(getattr(obj, "min_" + public, 0.0) or 0.0)
            parts[-1] = "." + public
        out.append(Hyperparameter("".join(parts).lstrip("."), index, leaf.shape, positive, minimum, frozen))
    return out


def _by_name(tree):
    info = hyperparameter_info(tree)
    table = {h.name: h for h in info}
    if len(table) != len(info):
        raise ValueError("duplicate hyperparameter names")
    return table


def hyperparameters(tree):
    """{name: constrained value} for every hyperparameter, in pytree order."""
    leaves = jax.tree_util.tree_leaves(tree)
    return {h.name: h.value(leaves[h.index]) for h in hyperparameter_info(tree)}


def describe(tree):
    """A text table of the hyperparameters of tree."""
    leaves = jax.tree_util.tree_leaves(tree)
    rows = []
    for h in hyperparameter_info(tree):
        v = h.value(leaves[h.index])
        try:
            vs = np.array2string(np.asarray(v), precision=4, separator=",") if np.ndim(v) else f"{float(v):.6g}"
        except Exception:
            vs = "(traced)"
        kind = (f"softplus + {h.minimum:g}" if h.minimum else "softplus") if h.positive else "unconstrained"
        rows.append((h.name, str(h.shape), vs, kind, "frozen" if h.frozen else ""))
    widths = [max([len(r[i]) for r in rows] + [len(t)]) for i, t in enumerate(("name", "shape", "value", "constraint",
                                                                                ""))]
    fmt_row = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    return "\n".join([fmt_row(("name", "shape", "value", "constraint", ""))] + [fmt_row(r) for r in rows])


def with_hyperparameters(tree, values):
    """A copy of tree with the named hyperparameters set to the given (constrained) values.

    Values are broadcast to the leaf's shape (a scalar sets every entry of a per-coordinate lengthscale) and
    stored with the leaf's dtype, so the result has the same structure as tree (jit does not retrace). Traceable in
    the values."""
    table = _by_name(tree)
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    for name, v in values.items():
        if name not in table:
            raise KeyError(f"unknown hyperparameter {name!r}; known: {list(table)}")
        h = table[name]
        v = jnp.broadcast_to(as_float_array(v), h.shape)
        if h.positive and is_concrete(v):
            a = np.asarray(v)
            if np.any(a <= h.minimum):
                raise ValueError(f"{name} = {a} must be above its minimum {h.minimum}")
        leaves[h.index] = h.raw(v).astype(leaves[h.index].dtype)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _glob(pattern):
    return re.compile(re.escape(pattern).replace(r"\*", ".*").replace(r"\?", "."))


def select(tree, include=None, exclude=None, include_frozen=False):
    """Names of the hyperparameters matching any `include` glob (default all) and no `exclude` glob."""
    inc = None if include is None else [_glob(p) for p in ([include] if isinstance(include, str) else include)]
    exc = [] if exclude is None else [_glob(p) for p in ([exclude] if isinstance(exclude, str) else exclude)]
    out = []
    for h in hyperparameter_info(tree):
        if h.frozen and not include_frozen:
            continue
        if inc is not None and not any(p.fullmatch(h.name) for p in inc):
            continue
        if any(p.fullmatch(h.name) for p in exc):
            continue
        out.append(h.name)
    return out


def hyperparameter_filter(tree, include=None, exclude=None, include_frozen=False):
    """Pytree of bools with tree's structure, True at the selected hyperparameters (see select):

        trainable, static = eqx.partition(kernel, hyperparameter_filter(kernel, exclude="*variance"))
        grads = jax.grad(lambda t: loss(eqx.combine(t, static)))(trainable)
    """
    table = _by_name(tree)
    chosen = {table[n].index for n in select(tree, include, exclude, include_frozen)}
    treedef = jax.tree_util.tree_structure(tree)
    return jax.tree_util.tree_unflatten(treedef, [i in chosen for i in range(treedef.num_leaves)])


def positive_names(tree, include_frozen=False):
    """Names of the positive (softplus-constrained) hyperparameters, in pytree order."""
    return [h.name for h in hyperparameter_info(tree) if h.positive and (include_frozen or not h.frozen)]


def to_log_vector(tree, names=None):
    """Concatenated log(value - minimum) of the named positive hyperparameters (default: positive_names(tree))."""
    names = positive_names(tree) if names is None else list(names)
    table = _by_name(tree)
    leaves = jax.tree_util.tree_leaves(tree)
    parts = []
    for n in names:
        h = table[n]
        if not h.positive:
            raise ValueError(f"{n} is unconstrained: no log coordinate")
        parts.append(jnp.log(softplus(leaves[h.index])).ravel())
    return jnp.concatenate(parts) if parts else jnp.zeros((0,))


def from_log_vector(tree, names, z):
    """Inverse of to_log_vector: a copy of tree with the named hyperparameters set to minimum + exp(z) (same
    order). Defined for every z (no floor to step below)."""
    table = _by_name(tree)
    names = positive_names(tree) if names is None else list(names)
    sizes = [int(np.prod(table[n].shape)) for n in names]
    if sum(sizes) != jnp.shape(z)[0]:
        raise ValueError(f"z has {jnp.shape(z)[0]} entries, the names need {sum(sizes)}")
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    offset = 0
    for n, size in zip(names, sizes):
        h = table[n]
        if not h.positive:
            raise ValueError(f"{n} is unconstrained: no log coordinate")
        coordinates = jnp.asarray(z[offset:offset + size], dtype=leaves[h.index].dtype).reshape(h.shape)
        leaves[h.index] = _softplus_inverse_exp(coordinates)
        offset += size
    return jax.tree_util.tree_unflatten(treedef, leaves)
