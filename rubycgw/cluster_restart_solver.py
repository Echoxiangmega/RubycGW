"""Warm-start wrappers for accelerated cluster ED+GW.

Two modes are supported:

* ``restart`` continues the same physical point and keeps an ordinary converged
  lattice SC-GW object as the diagnostic background;
* ``continuation`` starts a new interaction point directly from a previously
  converged embedded state and deliberately skips the standalone SC-GW solve.

In both cases Pulay history is rebuilt from scratch.  The continuation path is
intended for scans in ``V`` or extensions such as attractive ``V'`` where a
large-interaction SC-GW fixed point can be a poor initializer for the embedded
branch.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .cluster_restart import ClusterEDGWRestartState
from .gw import GWResult


def _seeded_solve(
    h0,
    Vq,
    params,
    grid,
    *,
    gw_opts,
    embed_opts,
    restart: ClusterEDGWRestartState,
    carrier_background: GWResult,
    final_background: GWResult,
    label: str,
):
    """Run the production solver while seeding impurity self-energy and bath."""
    from . import cluster_ed_gw_fast as fast

    original_cluster_gw = fast.cluster_gw_self_energy
    original_fit_bath = fast.fit_finite_bath
    cluster_calls = 0
    bath_calls = 0

    def cluster_gw_seeded(*args, **kwargs):
        nonlocal cluster_calls
        out = original_cluster_gw(*args, **kwargs)
        cluster_calls += 1
        if cluster_calls == 1:
            # The first call initializes sigma_imp only.  The first outer
            # iteration immediately recomputes the current cluster-GW double
            # counting map for the requested interaction.
            return (np.array(restart.Sigma_imp, copy=True), out[1], out[2])
        return out

    def fit_bath_seeded(*args, **kwargs):
        nonlocal bath_calls
        bath_calls += 1
        if bath_calls == 1 and kwargs.get("initial") is None:
            kwargs = dict(kwargs)
            kwargs["initial"] = restart.bath
        return original_fit_bath(*args, **kwargs)

    if embed_opts.verbose:
        print(
            f"[cluster-ED+GW] {label} from {restart.source_path or '<memory>'}; "
            "restoring G/Sigma/mu/bath and rebuilding Pulay history",
            flush=True,
        )

    fast.cluster_gw_self_energy = cluster_gw_seeded
    fast.fit_finite_bath = fit_bath_seeded
    try:
        result = fast.solve_cluster_ed_gw_fast(
            h0,
            Vq,
            params,
            grid,
            gw_opts=gw_opts,
            embed_opts=embed_opts,
            background=carrier_background,
        )
    finally:
        fast.cluster_gw_self_energy = original_cluster_gw
        fast.fit_finite_bath = original_fit_bath

    result.background = final_background
    return result


def solve_cluster_ed_gw_fast_restarted(
    h0,
    Vq,
    params,
    grid,
    *,
    gw_opts,
    embed_opts,
    restart: ClusterEDGWRestartState,
    background=None,
):
    """Continue the same cluster ED+GW point using a fresh Pulay history."""
    from . import cluster_ed_gw_fast as fast

    if background is None:
        if embed_opts.verbose:
            print(
                "[cluster-ED+GW] restart requested but no cached lattice GW "
                "background was supplied; solving it once ...",
                flush=True,
            )
        background = fast.solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not background.converged:
        raise RuntimeError(
            "restart requires a converged lattice GW background; "
            f"got residual={background.final_error:.3e}"
        )

    restart_background = replace(
        background,
        G=np.array(restart.G, copy=True),
        Sigma_H=np.array(restart.Sigma_H, copy=True),
        Sigma_GW=np.array(restart.Sigma_emb, copy=True),
        mu=float(restart.mu),
    )
    return _seeded_solve(
        h0,
        Vq,
        params,
        grid,
        gw_opts=gw_opts,
        embed_opts=embed_opts,
        restart=restart,
        carrier_background=restart_background,
        final_background=background,
        label="warm restart",
    )


def _continuation_background(
    restart: ClusterEDGWRestartState,
    grid,
) -> GWResult:
    """Create an initialization carrier without solving standalone SC-GW.

    The production embedding solver historically receives its initial ``G``,
    ``Sigma_H``, ``Sigma_GW`` and ``mu`` through a ``GWResult``.  For parameter
    continuation we construct that carrier directly from the saved embedded
    state.  ``W`` and ``P`` are placeholders only: both are recomputed from the
    current interaction at the beginning of the first embedding iteration.
    """
    norb = int(np.asarray(restart.Sigma_H).shape[-1])
    bosonic_shape = (grid.nb, grid.nk1, grid.nk2, norb, norb)
    return GWResult(
        G=np.array(restart.G, copy=True),
        W=np.zeros(bosonic_shape, dtype=complex),
        P=np.zeros(bosonic_shape, dtype=complex),
        Sigma_H=np.array(restart.Sigma_H, copy=True),
        Sigma_GW=np.array(restart.Sigma_emb, copy=True),
        mu=float(restart.mu),
        density=np.zeros(norb, dtype=float),
        converged=True,
        iterations=0,
        final_error=0.0,
        mixing_method="parameter-continuation-seed",
        min_screening_singular_value=np.nan,
        min_screening_m=0,
        min_screening_Omega=np.nan,
        min_screening_q1=np.nan,
        min_screening_q2=np.nan,
        min_screening_mode=np.zeros(norb, dtype=complex),
        min_density_mode=np.zeros(norb, dtype=complex),
        min_density_mode_residual=np.nan,
    )


def solve_cluster_ed_gw_fast_continued(
    h0,
    Vq,
    params,
    grid,
    *,
    gw_opts,
    embed_opts,
    restart: ClusterEDGWRestartState,
):
    """Continue an embedded solution to a new interaction point.

    This path performs **no standalone SC-GW solve**.  The saved embedded
    ``G``, Hartree self-energy, embedded dynamic self-energy, impurity
    self-energy, chemical potential and bath are used as the initial state.
    The first outer iteration immediately evaluates the *new* lattice GW,
    cluster-GW and ED maps with the requested interaction, after which the usual
    fixed-filling and Pulay iterations proceed unchanged.
    """
    carrier = _continuation_background(restart, grid)
    if embed_opts.verbose:
        print(
            "[cluster-ED+GW] parameter continuation: skipping standalone SC-GW "
            "background and entering the coupled embedding map directly",
            flush=True,
        )
    return _seeded_solve(
        h0,
        Vq,
        params,
        grid,
        gw_opts=gw_opts,
        embed_opts=embed_opts,
        restart=restart,
        carrier_background=carrier,
        final_background=carrier,
        label="parameter continuation",
    )


__all__ = [
    "solve_cluster_ed_gw_fast_restarted",
    "solve_cluster_ed_gw_fast_continued",
]
