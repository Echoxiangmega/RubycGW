#!/usr/bin/env python3
"""Robust wrapper for scan_primitive_finite_source_vs_nk.py.

Runs the ordinary primitive finite-source mesh scan, but if one SC-GW solve
fails it retries the *same* (mesh,h) point with progressively more conservative
mixing.  The failed iterate is used as the seed for the next retry, while the
outer scan still controls h-continuation exactly as in the base script.

This wrapper is diagnostic: it does not change the GW fixed-point equations.
A point is accepted only when the raw fixed-point residual satisfies the same
``opts.tol`` as the base scan.
"""
from __future__ import annotations

from dataclasses import replace

import scan_primitive_finite_source_vs_nk as scan
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast as _solve_base


def _solve_with_retries(h0, Vq, grid, opts, initial=None):
    """Retry a failed GW point without changing the target fixed point."""
    plans = [
        ("base", opts),
        (
            "pulay-0.12",
            replace(
                opts,
                mixing=min(float(opts.mixing), 0.12),
                mixing_method="pulay",
                max_iter=max(int(opts.max_iter), 1200),
                pulay_history=max(int(opts.pulay_history), 8),
                pulay_start=max(int(opts.pulay_start), 4),
                pulay_regularization=max(float(opts.pulay_regularization), 1e-9),
            ),
        ),
        (
            "pulay-0.06",
            replace(
                opts,
                mixing=min(float(opts.mixing), 0.06),
                mixing_method="pulay",
                max_iter=max(int(opts.max_iter), 1600),
                pulay_history=max(int(opts.pulay_history), 8),
                pulay_start=max(int(opts.pulay_start), 5),
                pulay_regularization=max(float(opts.pulay_regularization), 1e-8),
            ),
        ),
        (
            "linear-0.03",
            replace(
                opts,
                mixing=min(float(opts.mixing), 0.03),
                mixing_method="linear",
                max_iter=max(int(opts.max_iter), 2000),
            ),
        ),
    ]

    seed = initial
    best = None
    for ia, (label, trial_opts) in enumerate(plans):
        result = _solve_base(h0, Vq, grid, opts=trial_opts, initial=seed)
        print(
            f"    retry[{ia}] {label:12s}: "
            f"{'OK' if result.converged else 'FAIL':4s} "
            f"iter={result.iterations:4d} r={result.final_error:.3e} "
            f"smin={result.min_screening_singular_value:.3e} "
            f"q*=({result.min_screening_q1:.4f},{result.min_screening_q2:.4f})"
        )
        if best is None or float(result.final_error) < float(best.final_error):
            best = result
        if result.converged:
            return result
        # A failed iterate can still be a useful nearby seed for the next,
        # more strongly damped fixed-point iteration.
        seed = result

    return best


def main():
    scan.solve_matrix_gw_fast = _solve_with_retries
    scan.main()


if __name__ == "__main__":
    main()
