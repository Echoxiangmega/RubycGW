#!/usr/bin/env python3
"""Run one cluster ED+GW background through the public RubycGW API."""
from __future__ import annotations

import argparse
from pathlib import Path

from rubycgw.analysis import summarize_background
from rubycgw.api import (
    BackgroundConfig,
    ClusterConfig,
    GWConfig,
    GridConfig,
    RubyModel,
    load_restart_for_run,
    run_cluster_background,
    save_background,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--V", type=float, default=1.8)
    p.add_argument("--Vp", "--Vprime", dest="Vprime", type=float, default=0.0)
    p.add_argument("--Vx", "--Vcross", dest="Vcross", type=float, default=0.0)
    p.add_argument("--embed-max", type=int, default=200)
    p.add_argument("--embed-tol", type=float, default=2e-5)
    p.add_argument("--embed-mixing", type=float, default=0.8)
    p.add_argument("--nbath", type=int, default=6)
    p.add_argument("--continue-from", type=Path, default=None)
    p.add_argument("--restart-mode", choices=("continuation", "restart"), default="continuation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    model = RubyModel(
        ti=a.ti,
        t1=a.t1,
        t2=a.t2,
        V=a.V,
        Vprime=a.Vprime,
        Vcross=a.Vcross,
    )
    cfg = BackgroundConfig(
        filling=a.filling,
        grid=GridConfig(Lx=a.Lx, Ly=a.Ly, nw=a.nw, nOmega=a.nomega, T=a.T),
        gw=GWConfig(verbose=not a.quiet),
        cluster=ClusterConfig(
            max_iter=a.embed_max,
            tol=a.embed_tol,
            mixing=a.embed_mixing,
            nbath=a.nbath,
            verbose=not a.quiet,
        ),
    )
    restart = None
    if a.continue_from is not None:
        restart = load_restart_for_run(a.continue_from, model, cfg)
    run = run_cluster_background(
        model,
        cfg,
        restart=restart,
        restart_mode=a.restart_mode,
    )
    path = save_background(a.out, run)
    summary = summarize_background(run.result)
    print(f"saved {path}")
    print(
        "converged={converged}, iterations={iterations}, residual={final_error:.3e}, "
        "mu={mu:+.10f}, bath={bath_fit_error:.3e}, Gimp/Gc={impurity_mismatch:.3e}".format(
            **summary
        )
    )
    print("density=", summary["density"])


if __name__ == "__main__":
    main()
