#!/usr/bin/env python3
"""Warm-start wrapper for :mod:`run_cluster_ed_weak`.

Usage is identical to ``run_cluster_ed_weak.py`` with one additional option::

    --warm-start PATH.npz

The saved state may come from a smaller/larger k mesh.  On a different mesh the
local cluster correction ``Sigma_ED^C-Sigma_weak^C`` plus the impurity/bath
state are transplanted onto a freshly converged weak background on the new
mesh.  On the same mesh and same V the complete saved G/Sigma_emb state is
reused.
"""
from __future__ import annotations

from pathlib import Path
import sys

import run_cluster_ed_weak as base
from rubycgw.cluster_ed_weak_warm import (
    load_cluster_ed_warm_start,
    solve_cluster_ed_weak_warm,
)


def _pop_warm_start(argv: list[str]) -> Path | None:
    out: list[str] = [argv[0]]
    warm: Path | None = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--warm-start":
            if i + 1 >= len(argv):
                raise SystemExit("--warm-start requires a path")
            warm = Path(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--warm-start="):
            warm = Path(arg.split("=", 1)[1])
            i += 1
            continue
        out.append(arg)
        i += 1
    argv[:] = out
    return warm


def main():
    warm_path = _pop_warm_start(sys.argv)
    if warm_path is None:
        base.main()
        return

    original_solver = base.solve_cluster_ed_weak_fast

    def warm_solver(
        h0,
        Vq,
        params,
        grid,
        *,
        weak_solver="gw",
        gw_opts,
        embed_opts,
        gf2_sox_opts,
        background=None,
    ):
        warm = load_cluster_ed_warm_start(
            warm_path,
            weak_solver=weak_solver,
            grid=grid,
            filling=float(gw_opts.target_filling),
            T=float(grid.T),
            ti=float(params.ti),
            t1=float(params.t1),
            t2=float(params.t2),
            V=float(params.V),
            nbath=int(embed_opts.nbath),
        )
        print(
            f"[warm-start] loaded {warm_path}: weak={warm.weak_solver}, "
            f"old mesh={warm.old_nk1}x{warm.old_nk2}, old V={warm.old_V:g}, "
            f"bath={'yes' if warm.bath is not None else 'refit'}",
            flush=True,
        )
        return solve_cluster_ed_weak_warm(
            h0,
            Vq,
            params,
            grid,
            warm_start=warm,
            weak_solver=weak_solver,
            gw_opts=gw_opts,
            embed_opts=embed_opts,
            gf2_sox_opts=gf2_sox_opts,
            background=background,
        )

    base.solve_cluster_ed_weak_fast = warm_solver
    try:
        base.main()
    finally:
        base.solve_cluster_ed_weak_fast = original_solver


if __name__ == "__main__":
    main()
