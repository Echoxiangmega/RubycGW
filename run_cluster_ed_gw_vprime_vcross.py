#!/usr/bin/env python3
"""Run cluster ED+GW with straight V' and crossed Vx interactions.

For each neighboring A/B triangle pair, V' acts on the same two links as the
inter-triangle t1/t2 hopping, while Vx acts on the two crossed density links of
that same pair.  The three bond orientations are included with their true cell
offsets.  Negative V' or Vx is attractive.

Use ``--continue-from`` to follow a converged embedded branch while changing V,
V' and/or Vx.  Continuation skips the standalone SC-GW initializer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_gw_fast import ClusterEDGWFastOptions, solve_cluster_ed_gw_fast
from rubycgw.cluster_restart import load_cluster_ed_gw_restart
from rubycgw.cluster_restart_solver import solve_cluster_ed_gw_fast_continued
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import build_h0
from rubycgw.pulay_accel import install_scale_invariant_pulay
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
    reference_pair_effective_couplings,
    vprime_vcross_cluster_interactions,
)
from vprime_study.patches import install_cluster_interaction_hooks


install_scale_invariant_pulay()
install_cluster_interaction_hooks(vprime_vcross_cluster_interactions)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--V", type=float, default=1.8, help="intra-triangle repulsion")
    p.add_argument(
        "--Vprime", "--Vp", dest="Vprime", type=float, default=-0.10,
        help="straight inter-triangle density interaction on the hopping links",
    )
    p.add_argument(
        "--Vcross", "--Vx", dest="Vcross", type=float, default=-0.05,
        help="crossed density interaction inside each neighboring triangle pair",
    )
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max", type=int, default=160)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")

    p.add_argument("--embed-max", type=int, default=100)
    p.add_argument("--embed-tol", type=float, default=2e-5)
    p.add_argument("--embed-mixing", type=float, default=0.80)
    p.add_argument("--embed-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--embed-pulay-history", type=int, default=8)
    p.add_argument("--embed-pulay-start", type=int, default=3)
    p.add_argument("--embed-pulay-regularization", type=float, default=1e-7)
    p.add_argument("--embed-pulay-step-cap", type=float, default=3.0)
    p.add_argument("--impurity-mixing", type=float, default=1.0)

    p.add_argument("--nbath", type=int, default=6)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--quiet-gw", action="store_true")
    p.add_argument("--quiet-embed", action="store_true")
    p.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        help="reuse a converged embedded checkpoint while changing V/V'/Vx; skips standalone SC-GW",
    )
    p.add_argument("--out", type=Path, default=Path("results/vprime_vcross_cluster_bg"))
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _saved_interactions(path: Path) -> tuple[float, float]:
    """Return source (V',Vx); absent metadata means zero."""
    with np.load(path, allow_pickle=False) as z:
        if "Vprime" in z:
            vp = float(np.asarray(z["Vprime"]).reshape(()))
        elif "Vp" in z:
            vp = float(np.asarray(z["Vp"]).reshape(()))
        else:
            vp = 0.0
        if "Vcross" in z:
            vx = float(np.asarray(z["Vcross"]).reshape(()))
        elif "Vx" in z:
            vx = float(np.asarray(z["Vx"]).reshape(()))
        else:
            vx = 0.0
    return vp, vx


def main():
    args = _args()
    if args.Lx < 1 or args.Ly < 1:
        raise ValueError("Lx and Ly must be positive")

    params = VPrimeCrossParameters(
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        V=float(args.V), Vprime=float(args.Vprime), Vcross=float(args.Vcross),
    )
    grid = MatsubaraGrid(
        nk1=int(args.Lx), nk2=int(args.Ly), nw=int(args.nw),
        nOmega=int(args.nomega), T=float(args.T),
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)

    gw_opts = GWOptions(
        target_filling=float(args.filling),
        max_iter=int(args.gw_max), tol=float(args.gw_tol),
        mixing=float(args.gw_mixing), mixing_method=str(args.gw_mixing_method),
        verbose=not bool(args.quiet_gw), momentum_backend="fft",
    )
    embed_opts = ClusterEDGWFastOptions(
        max_iter=int(args.embed_max), tol=float(args.embed_tol),
        mixing=float(args.embed_mixing), mixing_method=str(args.embed_mixing_method),
        pulay_history=int(args.embed_pulay_history),
        pulay_start=int(args.embed_pulay_start),
        pulay_regularization=float(args.embed_pulay_regularization),
        pulay_step_cap=float(args.embed_pulay_step_cap),
        impurity_mixing=float(args.impurity_mixing),
        nbath=int(args.nbath), bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_energy_window=float(args.bath_energy_window),
        bath_coupling_bound=float(args.bath_coupling_bound),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not bool(args.quiet_embed),
    )

    print(
        "=== Ruby V-prime + V-cross cluster ED+GW ===\n"
        f"L={args.Lx}x{args.Ly}, V={args.V:g}, V'={args.Vprime:g}, "
        f"Vx={args.Vcross:g}, filling={args.filling:g}, T={args.T:g}\n"
        "V': two straight hopping links per neighboring triangle pair; "
        "Vx: two crossed density links in the same pair",
        flush=True,
    )
    if np.isclose(float(args.t1), float(args.t2), rtol=0.0, atol=1e-14) and args.V != 0.0:
        jn, jm, jz = reference_pair_effective_couplings(
            t=float(args.t1), V=float(args.V),
            Vprime=float(args.Vprime), Vcross=float(args.Vcross),
        )
        print(
            "[strong-coupling diagnostic, reference bond] "
            f"Jn={jn:+.6e}, Jm={jm:+.6e}, Jz={jz:+.6e}",
            flush=True,
        )

    continuation_source = ""
    source_V = np.nan
    source_Vprime = np.nan
    source_Vcross = np.nan
    if args.continue_from is None:
        result = solve_cluster_ed_gw_fast(
            h0, Vq, params, grid, gw_opts=gw_opts, embed_opts=embed_opts
        )
    else:
        restart = load_cluster_ed_gw_restart(
            args.continue_from,
            Lx=int(args.Lx),
            Ly=int(args.Ly),
            filling=float(args.filling),
            T=float(args.T),
            params=params,
            grid=grid,
            nbath=int(args.nbath),
            allow_interaction_change=True,
        )
        continuation_source = str(args.continue_from)
        source_V = float(restart.source_V)
        source_Vprime, source_Vcross = _saved_interactions(args.continue_from)
        print(
            "[cluster-ED+GW] continue interactions: "
            f"(V,V',Vx)=({source_V:g},{source_Vprime:g},{source_Vcross:g}) -> "
            f"({args.V:g},{args.Vprime:g},{args.Vcross:g})",
            flush=True,
        )
        result = solve_cluster_ed_gw_fast_continued(
            h0,
            Vq,
            params,
            grid,
            gw_opts=gw_opts,
            embed_opts=embed_opts,
            restart=restart,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / (
        f"cluster_ed_gw_vprime_vcross_L{args.Lx}x{args.Ly}_V{_tag(args.V)}_"
        f"Vp{_tag(args.Vprime)}_Vx{_tag(args.Vcross)}_fill{_tag(args.filling)}.npz"
    )
    np.savez_compressed(
        outfile,
        interaction_model=np.asarray("V_intra_plus_Vprime_straight_plus_Vcross_diagonal"),
        cluster_projection=np.asarray("q0_primitive_cell_projection_sum_repeated_cross_pairs"),
        continuation_source=np.asarray(continuation_source),
        continuation_source_V=float(source_V),
        continuation_source_Vprime=float(source_Vprime),
        continuation_source_Vcross=float(source_Vcross),
        background_kind=np.asarray(
            "parameter_continuation_seed" if args.continue_from is not None else "standalone_scgw"
        ),
        Lx=int(args.Lx), Ly=int(args.Ly),
        V=float(args.V),
        Vprime=float(args.Vprime), Vp=float(args.Vprime),
        Vcross=float(args.Vcross), Vx=float(args.Vcross),
        filling=float(args.filling), T=float(args.T),
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        omega=np.asarray(grid.omega), Omega=np.asarray(grid.Omega),
        converged=bool(result.converged), iterations=int(result.iterations),
        final_error=float(result.final_error),
        impurity_mismatch=float(result.impurity_mismatch),
        bath_fit_error=float(result.bath_fit_error),
        mixing_method=str(result.mixing_method),
        pulay_fallbacks=int(result.pulay_fallbacks),
        residual_history=np.asarray(result.residual_history),
        impurity_residual_history=np.asarray(result.impurity_residual_history),
        impurity_mismatch_history=np.asarray(result.impurity_mismatch_history),
        bath_fit_history=np.asarray(result.bath_fit_history),
        mu_history=np.asarray(result.mu_history),
        bath_nfev_history=np.asarray(result.bath_nfev_history),
        elapsed_history=np.asarray(result.elapsed_history),
        mu=float(result.mu), density=np.asarray(result.density),
        G=np.asarray(result.G), W=np.asarray(result.W), P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H), Sigma_emb=np.asarray(result.Sigma_emb),
        Sigma_GW_lattice=np.asarray(result.Sigma_GW_lattice),
        Sigma_GW_cluster=np.asarray(result.Sigma_GW_cluster),
        Sigma_ED_cluster=np.asarray(result.Sigma_ED_cluster),
        G_cluster=np.asarray(result.G_cluster), G_impurity=np.asarray(result.G_impurity),
        bath_energies=np.asarray(result.bath.energies),
        bath_couplings=np.asarray(result.bath.couplings),
        G_background=np.asarray(result.background.G),
        mu_background=float(result.background.mu),
    )
    print(
        f"saved {outfile}\n"
        f"converged={result.converged}, residual={result.final_error:.3e}, "
        f"bath={result.bath_fit_error:.3e}, mu={result.mu:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
