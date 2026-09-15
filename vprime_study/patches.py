"""Compatibility hooks for reusing the production cluster ED+GW/JF solvers.

The production modules import ``ruby_cluster_interactions`` by value.  The
V-prime studies keep the baseline package untouched and replace only those
module-local references while a launcher is running.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable

from .model import vprime_cluster_interactions


_PATCH_MODULES = (
    "rubycgw.cluster_ed_gw",
    "rubycgw.cluster_ed_gw_fast",
    "rubycgw.cluster_ed_gw_covariant",
    "rubycgw.cluster_ed_gw_jf",
    "rubycgw.cluster_ed_gw_jf_consistent",
    "rubycgw.cluster_restart_solver",
)


def install_cluster_interaction_hooks(
    interaction_builder: Callable,
) -> dict[str, object]:
    """Replace package-local cluster interaction builders with ``interaction_builder``.

    The six-site impurity and cluster-GW double-counting pieces both call the
    same ``ruby_cluster_interactions`` symbol.  Passing the model-specific q=0
    projection here therefore keeps the impurity and cluster subtraction
    consistent while leaving the lattice ``V(q)`` implementation untouched.

    Returns a dictionary of the replaced callables for optional restoration.
    """
    old: dict[str, object] = {}
    for name in _PATCH_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "ruby_cluster_interactions"):
            old[name] = getattr(module, "ruby_cluster_interactions")
            setattr(module, "ruby_cluster_interactions", interaction_builder)
    return old


def install_vprime_cluster_hooks() -> dict[str, object]:
    """Install q=0-projected V-prime impurity interactions package-wide."""
    return install_cluster_interaction_hooks(vprime_cluster_interactions)


def restore_vprime_cluster_hooks(old: dict[str, object]) -> None:
    """Restore callables returned by either hook installer."""
    for name, func in old.items():
        module = importlib.import_module(name)
        setattr(module, "ruby_cluster_interactions", func)
