"""Small, stable public API for RubycGW.

Research modules remain available under their historical import paths, but new
scripts, notebooks, and user interfaces should prefer this module.  Keeping the
surface intentionally small lets the internal research implementation evolve
without forcing users to track file-level refactors.
"""

from .io import background_metadata, save_background
from .models import ExtendedRubyParameters, RubyModel, RubyParameters
from .solvers import EffectiveEDResult
from .workflows import (
    BackgroundConfig,
    BackgroundRun,
    ClusterConfig,
    EffectiveEDConfig,
    GWConfig,
    GridConfig,
    load_restart_for_run,
    run_cluster_background,
    run_effective_ed,
)

__all__ = [
    "RubyParameters",
    "ExtendedRubyParameters",
    "RubyModel",
    "GridConfig",
    "GWConfig",
    "ClusterConfig",
    "BackgroundConfig",
    "BackgroundRun",
    "load_restart_for_run",
    "run_cluster_background",
    "save_background",
    "background_metadata",
    "EffectiveEDConfig",
    "EffectiveEDResult",
    "run_effective_ed",
]
