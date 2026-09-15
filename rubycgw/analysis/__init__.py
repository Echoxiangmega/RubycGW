"""Maintained analysis helpers for public workflows."""

from .results import summarize_background, summarize_effective_ed
from .self_energy import (
    decompose_kernel,
    dyson_kernel,
    green_from_kernel,
    k_to_realspace,
    realspace_to_k,
)

# Backward-compatible aliases retained for early public-API callers.
background_summary = summarize_background
effective_ed_summary = summarize_effective_ed

__all__ = [
    "summarize_background",
    "summarize_effective_ed",
    "background_summary",
    "effective_ed_summary",
    "realspace_to_k",
    "k_to_realspace",
    "dyson_kernel",
    "decompose_kernel",
    "green_from_kernel",
]
