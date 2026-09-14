"""Attractive inter-triangle V' extension used for the Ruby JF study.

This package is intentionally separate from :mod:`rubycgw` so the baseline
V-only implementation remains untouched.  Root-level launchers install the
small compatibility hooks needed by cluster ED+GW/JF and then delegate to the
production solvers.
"""

from .model import (
    VPrimeParameters,
    build_vprime_interaction,
    vprime_cluster_interactions,
    vprime_interaction_bonds,
)

__all__ = [
    "VPrimeParameters",
    "build_vprime_interaction",
    "vprime_cluster_interactions",
    "vprime_interaction_bonds",
]
