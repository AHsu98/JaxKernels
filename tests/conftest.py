"""Tests run on the CPU in float64 by default (small problems; no GPU contention). JAX_PLATFORMS=cuda to override.

Tests marked `slow` (deep nested derivative chains, e.g. 8th-order Matérn derivatives, ~20 s each to compile) run
only with JAXKERNELS_SLOW=1."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long compile times; run with JAXKERNELS_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("JAXKERNELS_SLOW"):
        return
    skip = pytest.mark.skip(reason="slow; set JAXKERNELS_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
