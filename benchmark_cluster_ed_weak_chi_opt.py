#!/usr/bin/env python3
"""Optimized wrapper for ``benchmark_cluster_ed_weak_chi.py``.

The saved zero-field embedding may use G0-based bath fitting and/or the sparse
Lanczos impurity solver.  This wrapper preserves those numerical choices during
the finite-source fixed-point response so the response differentiates the same
approximation that produced the saved state.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

import benchmark_cluster_ed_weak_chi as base
from rubycgw.bath_fit_complex_optimized import install_complex_bath_fit
from rubycgw.impurity_ed_lanczos import install_impurity_solver


def _input_path(argv) -> Path:
    for arg in argv[1:]:
        if not str(arg).startswith("-"):
            return Path(arg)
    raise SystemExit("expected embedding NPZ as first positional argument")


def main():
    path = _input_path(sys.argv)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        nbath = int(len(np.asarray(z["bath_energies"])))
        metric = (
            str(np.asarray(z["bath_fit_metric"]).reshape(()).item())
            if "bath_fit_metric" in z.files
            else "delta"
        )
        saved_solver = (
            str(np.asarray(z["impurity_solver"]).reshape(()).item())
            if "impurity_solver" in z.files
            else "dense"
        )
        thermal_tol = (
            float(z["impurity_thermal_tol"])
            if "impurity_thermal_tol" in z.files
            else 1.0e-10
        )
        thermal_max = (
            int(z["impurity_thermal_max_states"])
            if "impurity_thermal_max_states" in z.files
            else 64
        )
        krylov_steps = (
            int(z["impurity_krylov_steps"])
            if "impurity_krylov_steps" in z.files
            else 18
        )
        low_nfreq = (
            int(z["bath_g0_low_nfreq"])
            if "bath_g0_low_nfreq" in z.files
            else 4
        )

    solver_cls, resolved = install_impurity_solver(
        mode=saved_solver,
        nbath=nbath,
        thermal_state_tol=thermal_tol,
        thermal_max_states=thermal_max,
        krylov_steps=krylov_steps,
    )
    # The response modules import the impurity class into their own namespaces.
    import rubycgw.cluster_ed_gw_covariant as gw_cov
    import rubycgw.cluster_ed_weak_covariant as weak_cov
    gw_cov.FiniteBathImpurityED = solver_cls
    weak_cov.FiniteBathImpurityED = solver_cls

    original = base.solve_cluster_source_warm_weak

    def consistent_source_solver(
        h0_source,
        Vq,
        params,
        grid,
        K,
        target_filling,
        initial,
        opts,
        *,
        weak_solver="gw",
        gf2_sox_opts,
    ):
        h_cluster = np.mean(np.asarray(h0_source, dtype=complex), axis=(0, 1))
        h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
        install_complex_bath_fit(
            h_cluster,
            metric=metric,
            low_nfit=low_nfreq,
        )
        return original(
            h0_source,
            Vq,
            params,
            grid,
            K,
            target_filling,
            initial,
            opts,
            weak_solver=weak_solver,
            gf2_sox_opts=gf2_sox_opts,
        )

    base.solve_cluster_source_warm_weak = consistent_source_solver
    print(
        f"[response numerics] bath_metric={metric}, impurity_solver={resolved}, nbath={nbath}",
        flush=True,
    )
    try:
        base.main()
    finally:
        base.solve_cluster_source_warm_weak = original


if __name__ == "__main__":
    main()
