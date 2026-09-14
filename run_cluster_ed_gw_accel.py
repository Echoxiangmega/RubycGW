#!/usr/bin/env python3
"""Run cluster-ED+GW with scale-invariant late-stage Pulay acceleration.

This is a drop-in launcher for ``run_cluster_ed_gw.py``.  It keeps the same
physics and command-line arguments, replaces the Pulay coefficient solve by a
residual-scale-invariant form, and additionally supports warm continuation from
an existing cluster-ED+GW result through ``--restart-from``.

Examples
--------
Fresh run::

    python run_cluster_ed_gw_accel.py --Lx 2 --Ly 1 --V 1.0 --filling 2 \
        --embed-mixing-method pulay --embed-pulay-history 8

Continue the same physical calculation with a fresh Pulay history::

    python run_cluster_ed_gw_accel.py --Lx 2 --Ly 1 --V 1.0 --filling 2 \
        --restart-from results/cluster_ed_gw/cluster_ed_gw_L2x1_V1_fill2.npz \
        --embed-max 100 --embed-tol 2e-5

The restart restores G, Sigma_H, Sigma_emb, the impurity self-energy, chemical
potential and finite bath.  Pulay/DIIS vectors are intentionally rebuilt from
scratch, so a checkpoint made with the historical mixer can safely continue
with the scale-invariant implementation.
"""
from __future__ import annotations

from pathlib import Path
import sys

from rubycgw.pulay_accel import install_scale_invariant_pulay


install_scale_invariant_pulay()

import run_cluster_ed_gw as _driver  # noqa: E402
from rubycgw.cluster_restart import load_cluster_ed_gw_restart  # noqa: E402
from rubycgw.cluster_restart_solver import (  # noqa: E402
    solve_cluster_ed_gw_fast_restarted,
)


def _extract_restart(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Remove our extra option before delegating to the historical parser."""
    restart = None
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--restart-from":
            if i + 1 >= len(argv):
                raise SystemExit("--restart-from requires a checkpoint path")
            if restart is not None:
                raise SystemExit("--restart-from may be specified only once")
            restart = Path(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--restart-from="):
            if restart is not None:
                raise SystemExit("--restart-from may be specified only once")
            value = arg.split("=", 1)[1]
            if not value:
                raise SystemExit("--restart-from requires a checkpoint path")
            restart = Path(value)
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return restart, cleaned


def main() -> None:
    restart_path, cleaned_argv = _extract_restart(list(sys.argv))
    if "--help" in cleaned_argv or "-h" in cleaned_argv:
        # The delegated parser does not know about this launcher's extra option.
        # Print it explicitly before argparse emits the standard help text.
        print(
            "extra accelerated-launcher option:\n"
            "  --restart-from PATH   continue from a saved cluster-ED+GW .npz state\n"
        )

    sys.argv[:] = cleaned_argv
    original_solve = _driver.solve_cluster_ed_gw_fast

    if restart_path is not None:
        def _solve_with_restart(
            h0,
            Vq,
            params,
            grid,
            *,
            gw_opts,
            embed_opts,
            background=None,
        ):
            if gw_opts.target_filling is None:
                raise ValueError("cluster ED+GW restart requires fixed target filling")
            restart = load_cluster_ed_gw_restart(
                restart_path,
                Lx=int(grid.nk1),
                Ly=int(grid.nk2),
                filling=float(gw_opts.target_filling),
                T=float(grid.T),
                params=params,
                grid=grid,
                nbath=int(embed_opts.nbath),
            )
            return solve_cluster_ed_gw_fast_restarted(
                h0,
                Vq,
                params,
                grid,
                gw_opts=gw_opts,
                embed_opts=embed_opts,
                restart=restart,
                background=background,
            )

        _driver.solve_cluster_ed_gw_fast = _solve_with_restart

    try:
        _driver.main()
    finally:
        _driver.solve_cluster_ed_gw_fast = original_solve


if __name__ == "__main__":
    main()
