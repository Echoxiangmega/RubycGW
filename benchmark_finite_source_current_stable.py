#!/usr/bin/env python3
"""Stable finite-source current benchmark with separate GW and GW+SOX mixing.

This is a thin wrapper around ``benchmark_finite_source_current.py``.  The
original benchmark uses one ``GWOptions`` object for both ordinary GW and
GW+SOX, so changing ``--mixing`` to stabilize SOX also changes the ordinary-GW
iteration.  Near strong-coupling current branches that is often counterproductive.

This wrapper keeps all original benchmark arguments, while adding SOX-specific
background-solver controls:

    --sox-mixing
    --sox-mixing-method
    --sox-max-iter
    --sox-pulay-history
    --sox-pulay-start
    --sox-pulay-regularization

The ordinary-GW controls remain the original options, e.g. ``--mixing`` and
``--gw-max-iter``.  The source, ED calculation, observables, outputs and h
continuation are otherwise unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import sys

import benchmark_finite_source_current as benchmark
from rubycgw.gw_sox import solve_matrix_gw_sox as _solve_matrix_gw_sox


def _split_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sox-mixing", type=float, default=0.08)
    p.add_argument(
        "--sox-mixing-method",
        choices=["linear", "pulay"],
        default="pulay",
    )
    p.add_argument("--sox-max-iter", type=int, default=None)
    p.add_argument("--sox-pulay-history", type=int, default=None)
    p.add_argument("--sox-pulay-start", type=int, default=None)
    p.add_argument("--sox-pulay-regularization", type=float, default=None)
    known, remaining = p.parse_known_args(argv)
    return known, remaining


def main():
    sox_cfg, remaining = _split_args(sys.argv[1:])
    if not (0.0 < float(sox_cfg.sox_mixing) <= 1.0):
        raise ValueError("--sox-mixing must lie in (0,1]")
    if sox_cfg.sox_max_iter is not None and int(sox_cfg.sox_max_iter) < 1:
        raise ValueError("--sox-max-iter must be positive")
    if sox_cfg.sox_pulay_history is not None and int(sox_cfg.sox_pulay_history) < 2:
        raise ValueError("--sox-pulay-history must be at least 2")
    if sox_cfg.sox_pulay_start is not None and int(sox_cfg.sox_pulay_start) < 1:
        raise ValueError("--sox-pulay-start must be at least 1")

    # The wrapped benchmark must only see arguments it already knows.
    sys.argv = [sys.argv[0]] + remaining

    def solve_matrix_gw_sox_split(
        h0,
        Vq,
        grid,
        opts,
        sox_opts,
        initial=None,
    ):
        updates = {
            "mixing": float(sox_cfg.sox_mixing),
            "mixing_method": str(sox_cfg.sox_mixing_method),
        }
        if sox_cfg.sox_max_iter is not None:
            updates["max_iter"] = int(sox_cfg.sox_max_iter)
        if sox_cfg.sox_pulay_history is not None:
            updates["pulay_history"] = int(sox_cfg.sox_pulay_history)
        if sox_cfg.sox_pulay_start is not None:
            updates["pulay_start"] = int(sox_cfg.sox_pulay_start)
        if sox_cfg.sox_pulay_regularization is not None:
            updates["pulay_regularization"] = float(
                sox_cfg.sox_pulay_regularization
            )
        tuned_opts = replace(opts, **updates)
        return _solve_matrix_gw_sox(
            h0,
            Vq,
            grid,
            opts=tuned_opts,
            sox_opts=sox_opts,
            initial=initial,
        )

    benchmark.solve_matrix_gw_sox = solve_matrix_gw_sox_split

    print(
        "[stable wrapper] ordinary GW uses original --mixing/--gw-max-iter; "
        f"GW+SOX uses mixing={sox_cfg.sox_mixing:g}, "
        f"method={sox_cfg.sox_mixing_method}, "
        f"max_iter={sox_cfg.sox_max_iter if sox_cfg.sox_max_iter is not None else 'same as GW'}"
    )
    benchmark.main()


if __name__ == "__main__":
    main()
