"""Compatibility layer for the historical ``vprime_study`` import path.

The canonical extended Ruby model now lives in :mod:`rubycgw.models.ruby`.
This module intentionally keeps the old names so existing research scripts and
saved workflows continue to run unchanged.
"""
from __future__ import annotations

from rubycgw.models.ruby import (
    CROSS_INTERTRIANGLE_BONDS,
    INTERTRIANGLE_BONDS,
    INTRATRIANGLE_BONDS,
    REFERENCE_CROSS_BONDS,
    REFERENCE_STRAIGHT_BONDS,
    ExtendedRubyParameters,
    build_extended_interaction,
    extended_cluster_interactions,
    extended_cluster_matrix,
    extended_interaction_bonds,
    physical_pair_cluster_interactions,
    reference_pair_effective_couplings,
)

VPrimeCrossParameters = ExtendedRubyParameters
vprime_vcross_interaction_bonds = extended_interaction_bonds
build_vprime_vcross_interaction = build_extended_interaction
vprime_vcross_cluster_interactions = extended_cluster_interactions
vprime_vcross_cluster_matrix = extended_cluster_matrix
vprime_vcross_physical_pair_interactions = physical_pair_cluster_interactions

__all__ = [
    "VPrimeCrossParameters",
    "INTRATRIANGLE_BONDS",
    "INTERTRIANGLE_BONDS",
    "CROSS_INTERTRIANGLE_BONDS",
    "REFERENCE_STRAIGHT_BONDS",
    "REFERENCE_CROSS_BONDS",
    "vprime_vcross_interaction_bonds",
    "build_vprime_vcross_interaction",
    "vprime_vcross_cluster_interactions",
    "vprime_vcross_cluster_matrix",
    "vprime_vcross_physical_pair_interactions",
    "reference_pair_effective_couplings",
]
