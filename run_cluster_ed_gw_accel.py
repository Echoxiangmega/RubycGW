#!/usr/bin/env python3
"""Run cluster-ED+GW with scale-invariant late-stage Pulay acceleration.

This is a drop-in launcher for ``run_cluster_ed_gw.py``.  It keeps the same
physics and command-line arguments, replaces the Pulay coefficient solve by a
residual-scale-invariant form, and adds two warm-start modes:

* ``--restart-from``: same physical point, strict parameter validation;
* ``--continue-from``: new interaction strength, reuse the embedded state and
  skip the standalone SC-GW background solve entirely.

Examples
--------
Fresh run::

    python run_cluster_ed_gw_accel.py --Lx 2 --Ly 1 --V 1.0 --filling 2 \
        --embed-mixing-method pulay --embed-pulay-history 8

Continue the same physical calculation::

    python run_cluster_ed_gw_accel.py --Lx 2 --Ly 1 --V 1.0 --filling 2 \
        --restart-from results/cluster_ed_gw/cluster_ed_gw_L2x1_V1_fill2.npz

Follow the embedded branch to a new V without solving SC-GW first::

    python run_cluster_ed_gw_accel.py --Lx 2 --Ly 1 --V 1.1 --filling 2 \
        --continue-from results/cluster_ed_gw/cluster_ed_gw_L2x1_V1_fill2.npz

Both modes restore G, Sigma_H, Sigma_emb, impurity self-energy, chemical
potential and finite bath.  Pulay/DIIS vectors are intentionally rebuilt from
scratch.
"""
from __future__ import annotations

from pathlib import Path
import sys

from rubycgw.pulay_accel import install_scale_invariant_pulay


install_scale_invariant_pulay()

import run_cluster_ed_gw as _driver  # noqa: E402
from rubycgw.cluster_restart import load_cluster_ed_gw_restart  # noqa: E402
from rubycgw.cluster_restart_solver import (  # noqa: E402
    solve_cluster_ed_gw_fast_continued,
    solve_cluster_ed_gw_fast_restarted,
)


def _extract_warm_start(
    argv: list[str],
) -> tuple[Path | None, Path | None, list[str]]:
    """Remove launcher-only warm-start options before delegated argparse."""
    restart = None
    continuation = None
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        matched = None
        value = None
        if arg in ("--restart-from", "--continue-from"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a checkpoint path")
            matched = arg
            value = argv[i + 1]
            i += 2
        elif arg.startswith("--restart-from=") or arg.startswith("--continue-from="):
            matched, value = arg.split("=", 1)
            if not value:
                raise SystemExit(f"{matched} requires a checkpoint path")
            i += 1
        else:
            cleaned.append(arg)
            i += 1
            continue

        if matched == "--restart-from":
            if restart is not None:
                raise SystemExit("--restart-from may be specified only once")
            restart = Path(value)
        else:
            if continuation is not None:
                raise SystemExit("--continue-from may be specified only once")
            continuation = Path(value)

    if restart is not None and continuation is not None:
        raise SystemExit("use only one of --restart-from and --continue-from")
    return restart, continuation, cleaned


def main() -> None:
    restart_path, continue_path, cleaned_argv = _extract_warm_start(list(sys.argv))
    if "--help" in cleaned_argv or "-h" in cleaned_argv:
        print(
            "extra accelerated-launcher options:\n"
            "  --restart-from PATH    continue from a saved cluster-ED+GW .npz state (same physical point)\n"
            "  --continue-from PATH   continue to a new V without standalone SC-GW\n"
        )

    # A continuation deliberately has no standalone SC-GW background.  Disable
    # the delegated SC-GW cache so its initialization carrier is never written
    # under a misleading GW cache key.
    if continue_path is not None and "--no-cache" not in cleaned_argv:
        cleaned_argv.append("--no-cache")

    sys.argv[:] = cleaned_argv
    original_solve = _driver.solve_cluster_ed_gw_fast

    warm_path = restart_path if restart_path is not None else continue_path
    if warm_path is not None:
        is_continuation = continue_path is not None

        def _solve_with_warm_start(
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
                raise ValueError("cluster ED+GW warm start requires fixed target filling")
            restart = load_cluster_ed_gw_restart(
                warm_path,
                Lx=int(grid.nk1),
                Ly=int(grid.nk2),
                filling=float(gw_opts.target_filling),
                T=float(grid.T),
                params=params,
                grid=grid,
                nbath=int(embed_opts.nbath),
                allow_interaction_change=bool(is_continuation),
            )
            if is_continuation:
                if embed_opts.verbose:
                    print(
                        f"[cluster-ED+GW] continue V: {restart.source_V:g} -> {float(params.V):g}",
                        flush=True,
                    )
                return solve_cluster_ed_gw_fast_continued(
                    h0,
                    Vq,
                    params,
                    grid,
                    gw_opts=gw_opts,
                    embed_opts=embed_opts,
                    restart=restart,
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

        _driver.solve_cluster_ed_gw_fast = _solve_with_warm_start

    try:
        _driver.main()
    finally:
        _driver.solve_cluster_ed_gw_fast = original_solve


if __name__ == "__main__":
    main()
