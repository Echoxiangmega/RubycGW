"""High-level cluster ED+GW background workflow.

This module is intentionally thin: it assembles the existing, validated
numerical kernels without duplicating their physics.  It is the entry point that
CLI scripts, notebooks, and a future GUI should call.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..grids import MatsubaraGrid
from ..cluster_ed_gw import ruby_cluster_interactions
from ..models import RubyModel
from ..solvers.cluster import (
    ClusterEDGWFastOptions,
    ClusterEDGWFastResult,
    ClusterEDGWRestartState,
    load_cluster_ed_gw_restart,
    solve_cluster_ed_gw_fast,
    solve_cluster_ed_gw_fast_continued,
    solve_cluster_ed_gw_fast_restarted,
)
from ..solvers.gw import GWOptions
from vprime_study.patches import (
    install_cluster_interaction_hooks,
    restore_vprime_cluster_hooks,
)


@dataclass(frozen=True)
class GridConfig:
    """Momentum/frequency grid for production calculations."""

    Lx: int = 3
    Ly: int = 3
    nw: int = 55
    nOmega: int = 12
    T: float = 0.08

    def build(self) -> MatsubaraGrid:
        return MatsubaraGrid(
            nk1=int(self.Lx),
            nk2=int(self.Ly),
            nw=int(self.nw),
            nOmega=int(self.nOmega),
            T=float(self.T),
        )


@dataclass(frozen=True)
class GWConfig:
    max_iter: int = 160
    tol: float = 1e-8
    mixing: float = 0.25
    mixing_method: str = "pulay"
    pulay_history: int = 6
    pulay_start: int = 3
    pulay_regularization: float = 1e-10
    momentum_backend: str = "fft"
    verbose: bool = True

    def build(self, *, filling: float) -> GWOptions:
        return GWOptions(
            target_filling=float(filling),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            mixing=float(self.mixing),
            mixing_method=str(self.mixing_method),
            pulay_history=int(self.pulay_history),
            pulay_start=int(self.pulay_start),
            pulay_regularization=float(self.pulay_regularization),
            momentum_backend=str(self.momentum_backend),
            verbose=bool(self.verbose),
        )


@dataclass(frozen=True)
class ClusterConfig:
    max_iter: int = 200
    tol: float = 2e-5
    mixing: float = 0.80
    mixing_method: str = "pulay"
    pulay_history: int = 8
    pulay_start: int = 3
    pulay_regularization: float = 1e-7
    pulay_step_cap: float = 3.0
    impurity_mixing: float = 1.0
    nbath: int = 6
    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 300
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_fit_xtol: float = 1e-9
    discard_weight_tol: float = 1e-11
    verbose: bool = True

    def build(self) -> ClusterEDGWFastOptions:
        return ClusterEDGWFastOptions(
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            mixing=float(self.mixing),
            mixing_method=str(self.mixing_method),
            pulay_history=int(self.pulay_history),
            pulay_start=int(self.pulay_start),
            pulay_regularization=float(self.pulay_regularization),
            pulay_step_cap=float(self.pulay_step_cap),
            impurity_mixing=float(self.impurity_mixing),
            nbath=int(self.nbath),
            bath_fit_nfreq=int(self.bath_fit_nfreq),
            bath_fit_max_nfev=int(self.bath_fit_max_nfev),
            bath_energy_window=float(self.bath_energy_window),
            bath_coupling_bound=float(self.bath_coupling_bound),
            bath_fit_xtol=float(self.bath_fit_xtol),
            discard_weight_tol=float(self.discard_weight_tol),
            verbose=bool(self.verbose),
        )


@dataclass(frozen=True)
class BackgroundConfig:
    """Complete numerical configuration for one cluster ED+GW background."""

    filling: float = 2.0
    grid: GridConfig = field(default_factory=GridConfig)
    gw: GWConfig = field(default_factory=GWConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


@dataclass
class BackgroundRun:
    model: RubyModel
    config: BackgroundConfig
    grid: MatsubaraGrid
    h0: np.ndarray
    Vq: np.ndarray
    result: ClusterEDGWFastResult
    restart_mode: str = "fresh"
    restart_source: str = ""


@contextmanager
def _v_only_cluster_interactions_enabled():
    """Use intra-triangle V only in ED and cluster-GW double counting.

    The lattice h0/V(q) still comes from the full extended Ruby model, so
    Vprime/Vcross remain active in lattice Hartree, screening and GW self-energy.
    """
    old = install_cluster_interaction_hooks(ruby_cluster_interactions)
    try:
        yield
    finally:
        restore_vprime_cluster_hooks(old)


def load_restart_for_run(path: str | Path, model: RubyModel, config: BackgroundConfig):
    """Load a checkpoint with structural validation for the requested run."""
    grid = config.grid.build()
    return load_cluster_ed_gw_restart(
        Path(path),
        Lx=int(config.grid.Lx),
        Ly=int(config.grid.Ly),
        filling=float(config.filling),
        T=float(config.grid.T),
        params=model.parameters(),
        grid=grid,
        nbath=int(config.cluster.nbath),
        allow_interaction_change=True,
    )


def run_cluster_background(
    model: RubyModel,
    config: BackgroundConfig = BackgroundConfig(),
    *,
    restart: ClusterEDGWRestartState | None = None,
    restart_mode: str = "continuation",
) -> BackgroundRun:
    """Run one cluster ED+GW background.

    ``restart_mode`` may be ``"continuation"`` (allow a changed interaction
    point), ``"restart"`` (same physical point), or ``"fresh"``.  Passing no
    restart always performs a fresh solve.
    """
    grid = config.grid.build()
    params = model.parameters()
    h0 = model.build_h0(grid.kmesh())
    Vq = model.build_interaction(grid.qmesh())
    gw_opts = config.gw.build(filling=float(config.filling))
    embed_opts = config.cluster.build()

    mode = "fresh" if restart is None else str(restart_mode).lower()
    if mode not in {"fresh", "restart", "continuation"}:
        raise ValueError("restart_mode must be 'fresh', 'restart', or 'continuation'")

    with _v_only_cluster_interactions_enabled():
        if restart is None or mode == "fresh":
            result = solve_cluster_ed_gw_fast(
                h0, Vq, params, grid, gw_opts=gw_opts, embed_opts=embed_opts
            )
        elif mode == "restart":
            result = solve_cluster_ed_gw_fast_restarted(
                h0,
                Vq,
                params,
                grid,
                gw_opts=gw_opts,
                embed_opts=embed_opts,
                restart=restart,
            )
        else:
            result = solve_cluster_ed_gw_fast_continued(
                h0,
                Vq,
                params,
                grid,
                gw_opts=gw_opts,
                embed_opts=embed_opts,
                restart=restart,
            )

    return BackgroundRun(
        model=model,
        config=config,
        grid=grid,
        h0=np.asarray(h0),
        Vq=np.asarray(Vq),
        result=result,
        restart_mode=mode,
        restart_source="" if restart is None else str(restart.source_path or ""),
    )


__all__ = [
    "GridConfig",
    "GWConfig",
    "ClusterConfig",
    "BackgroundConfig",
    "BackgroundRun",
    "load_restart_for_run",
    "run_cluster_background",
]
