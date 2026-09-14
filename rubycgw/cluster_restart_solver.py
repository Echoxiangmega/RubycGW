"""Warm-restart wrapper for the accelerated cluster-ED+GW solver.

The core solver historically initializes from the converged lattice-GW state.
For a continuation run we instead want to start from a saved embedded state
without changing the core fixed-point equations.  This module injects the saved
state only during initialization and then restores the ordinary functions.

Pulay history is intentionally rebuilt from scratch.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .cluster_restart import ClusterEDGWRestartState


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
    """Continue cluster ED+GW from ``restart`` using a fresh Pulay history.

    ``background`` remains the ordinary converged lattice-GW result used for
    diagnostics/output.  A temporary copy supplies the saved embedded dynamic
    state to the historical solver initializer.  The first cluster self-energy
    call and first bath fit are likewise seeded from the saved impurity
    self-energy and finite bath.  All subsequent calls use the unmodified
    production maps.
    """
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

    original_cluster_gw = fast.cluster_gw_self_energy
    original_fit_bath = fast.fit_finite_bath
    cluster_calls = 0
    bath_calls = 0

    def cluster_gw_seeded(*args, **kwargs):
        nonlocal cluster_calls
        out = original_cluster_gw(*args, **kwargs)
        cluster_calls += 1
        if cluster_calls == 1:
            # The first call is used only to initialize sigma_imp.  The first
            # outer iteration immediately recomputes the actual cluster-GW
            # double-counting self-energy with the current G.
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
            f"[cluster-ED+GW] warm restart from {restart.source_path or '<memory>'}; "
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
            background=restart_background,
        )
    finally:
        fast.cluster_gw_self_energy = original_cluster_gw
        fast.fit_finite_bath = original_fit_bath

    # Preserve the real SC-GW reference in saved output and benchmarks; the
    # restart_background object was only an initialization carrier.
    result.background = background
    return result


__all__ = ["solve_cluster_ed_gw_fast_restarted"]
