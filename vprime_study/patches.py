"""Compatibility hooks for reusing the production cluster ED+GW/JF solvers.

The production modules import ``ruby_cluster_interactions`` by value.  The
V-prime study keeps the baseline package untouched and replaces only those
module-local references while a V-prime launcher is running.
"""
from __future__ import annotations

import importlib

from .model import vprime_cluster_interactions


_PATCH_MODULES = (
    "rubycgw.cluster_ed_gw",
    "rubycgw.cluster_ed_gw_fast",
    "rubycgw.cluster_ed_gw_covariant",
    "rubycgw.cluster_ed_gw_jf",
    "rubycgw.cluster_ed_gw_jf_consistent",
    "rubycgw.cluster_restart_solver",
)


def install_vprime_cluster_hooks() -> dict[str, object]:
    """Install q=0-projected V-prime impurity interactions package-wide.

    Returns a dictionary of the replaced callables.  Launchers are short-lived,
    so restoration is normally unnecessary, but the return value is convenient
    for tests and interactive use.
    """
    old: dict[str, object] = {}
    for name in _PATCH_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "ruby_cluster_interactions"):
            old[name] = getattr(module, "ruby_cluster_interactions")
            setattr(module, "ruby_cluster_interactions", vprime_cluster_interactions)
    return old


def restore_vprime_cluster_hooks(old: dict[str, object]) -> None:
    """Restore callables returned by :func:`install_vprime_cluster_hooks`."""
    for name, func in old.items():
        module = importlib.import_module(name)
        setattr(module, "ruby_cluster_interactions", func)
