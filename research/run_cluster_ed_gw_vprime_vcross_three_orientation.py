#!/usr/bin/env python3
"""Run the simultaneous C3-symmetric three-orientation physical-pair ED+GW map."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import rubycgw.cluster_ed_gw_three_orientation as _three
from rubycgw.cluster_ed_gw_covariant import fit_finite_bath_complex
from rubycgw.cluster_ed_gw_fast import ClusterEDGWFastOptions
from rubycgw.cluster_ed_gw_three_orientation import solve_three_orientation_ed_gw
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import build_h0
from rubycgw.pulay_accel import install_scale_invariant_pulay
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
    reference_pair_effective_couplings,
)

install_scale_invariant_pulay()


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--V", type=float, default=1.8)
    p.add_argument("--Vprime", "--Vp", dest="Vprime", type=float, default=-0.10)
    p.add_argument("--Vcross", "--Vx", dest="Vcross", type=float, default=-0.07)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max", type=int, default=400)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")

    p.add_argument("--embed-max", type=int, default=80)
    p.add_argument("--embed-tol", type=float, default=5e-7)
    p.add_argument("--embed-mixing", type=float, default=0.5)
    p.add_argument("--embed-mixing-method", choices=("linear", "pulay", "broyden"), default="broyden")
    p.add_argument("--embed-pulay-history", type=int, default=8)
    p.add_argument("--embed-pulay-start", type=int, default=3)
    p.add_argument("--embed-pulay-regularization", type=float, default=1e-7)
    p.add_argument("--embed-pulay-step-cap", type=float, default=3.0)
    p.add_argument("--embed-broyden-history", type=int, default=8)
    p.add_argument("--embed-broyden-regularization", type=float, default=1e-8)
    p.add_argument("--embed-broyden-step-cap", type=float, default=3.0)
    p.add_argument("--embed-broyden-reset-growth", type=float, default=1.5)
    p.add_argument("--impurity-mixing", type=float, default=1.0)

    p.add_argument("--nbath", type=int, default=6)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--bath-fit-metric", choices=("delta", "g0"), default="delta")
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument("--complex-bath", action="store_true")
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--quiet-gw", action="store_true")
    p.add_argument("--quiet-embed", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/vprime_vcross_cluster_three_orientation"),
    )
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def main():
    args = _args()
    if args.Lx != args.Ly:
        raise ValueError("three-orientation C3-symmetric embedding requires Lx=Ly")

    params = VPrimeCrossParameters(
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        V=float(args.V), Vprime=float(args.Vprime), Vcross=float(args.Vcross),
    )
    if args.complex_bath:
        if args.bath_fit_metric != "delta":
            raise ValueError("--complex-bath currently supports only --bath-fit-metric delta")
        _three.fit_finite_bath = fit_finite_bath_complex

    grid = MatsubaraGrid(
        nk1=int(args.Lx), nk2=int(args.Ly), nw=int(args.nw),
        nOmega=int(args.nomega), T=float(args.T),
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)

    gw_opts = GWOptions(
        target_filling=float(args.filling),
        max_iter=int(args.gw_max),
        tol=float(args.gw_tol),
        mixing=float(args.gw_mixing),
        mixing_method=str(args.gw_mixing_method),
        verbose=not bool(args.quiet_gw),
        momentum_backend="fft",
    )
    embed_opts = ClusterEDGWFastOptions(
        max_iter=int(args.embed_max),
        tol=float(args.embed_tol),
        mixing=float(args.embed_mixing),
        mixing_method=str(args.embed_mixing_method),
        pulay_history=int(args.embed_pulay_history),
        pulay_start=int(args.embed_pulay_start),
        pulay_regularization=float(args.embed_pulay_regularization),
        pulay_step_cap=float(args.embed_pulay_step_cap),
        broyden_history=int(args.embed_broyden_history),
        broyden_regularization=float(args.embed_broyden_regularization),
        broyden_step_cap=float(args.embed_broyden_step_cap),
        broyden_reset_growth=float(args.embed_broyden_reset_growth),
        impurity_mixing=float(args.impurity_mixing),
        nbath=int(args.nbath),
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_fit_metric=str(args.bath_fit_metric),
        bath_energy_window=float(args.bath_energy_window),
        bath_coupling_bound=float(args.bath_coupling_bound),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not bool(args.quiet_embed),
    )

    print(
        "=== C3-symmetric three-orientation physical-pair ED+GW ===\n"
        f"L={args.Lx}x{args.Ly}, V={args.V:g}, V'={args.Vprime:g}, "
        f"Vx={args.Vcross:g}, filling={args.filling:g}, T={args.T:g}\n"
        "map = lattice GW + (1/3) sum_r [ED(pair_r)-GW(pair_r)] mapped to common gauge",
        flush=True,
    )
    if np.isclose(args.t1, args.t2, rtol=0.0, atol=1e-14) and args.V != 0:
        jn, jm, jz = reference_pair_effective_couplings(
            t=float(args.t1), V=float(args.V),
            Vprime=float(args.Vprime), Vcross=float(args.Vcross),
        )
        print(f"[strong-coupling] Jn={jn:+.6e}, Jm={jm:+.6e}, Jz={jz:+.6e}", flush=True)

    result = solve_three_orientation_ed_gw(
        h0, Vq, params, grid, gw_opts=gw_opts, embed_opts=embed_opts
    )

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / (
        f"cluster_ed_gw_threeori_L{args.Lx}x{args.Ly}_"
        f"V{_tag(args.V)}_Vp{_tag(args.Vprime)}_Vx{_tag(args.Vcross)}_"
        f"fill{_tag(args.filling)}.npz"
    )
    np.savez_compressed(
        outfile,
        interaction_model=np.asarray("lattice_full_V_Vprime_Vcross__three_physical_pairs"),
        cluster_projection=np.asarray("three_orientation_physical_pair_average"),
        embedding_scheme=np.asarray(
            "GW(full)+(1/3)sum_r[ED(V+pair_r)-GW(V+pair_r)]"
        ),
        orientation_method=np.asarray("simultaneous_C3_symmetric_map"),
        Lx=int(args.Lx), Ly=int(args.Ly),
        V=float(args.V), Vprime=float(args.Vprime), Vp=float(args.Vprime),
        Vcross=float(args.Vcross), Vx=float(args.Vcross),
        filling=float(args.filling), T=float(args.T),
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        omega=np.asarray(grid.omega), Omega=np.asarray(grid.Omega),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        mu=float(result.mu),
        density=np.asarray(result.density),
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H),
        Sigma_emb=np.asarray(result.Sigma_emb),
        Sigma_GW_lattice=np.asarray(result.Sigma_GW_lattice),
        Sigma_GW_cluster_by_orientation=np.asarray(result.Sigma_GW_cluster_by_orientation),
        Sigma_ED_cluster_by_orientation=np.asarray(result.Sigma_ED_cluster_by_orientation),
        G_cluster_by_orientation=np.asarray(result.G_cluster_by_orientation),
        G_impurity_by_orientation=np.asarray(result.G_impurity_by_orientation),
        impurity_static_shift_by_orientation=np.asarray(
            result.impurity_static_shift_by_orientation
        ),
        impurity_mismatch_by_orientation=np.asarray(
            result.impurity_mismatch_by_orientation
        ),
        bath_fit_error_by_orientation=np.asarray(
            result.bath_fit_error_by_orientation
        ),
        bath_symmetry_mode=np.asarray("representative_ori0_plus_exact_C3"),
        bath_energies_representative=np.asarray(result.baths[0].energies),
        bath_couplings_representative=np.asarray(result.baths[0].couplings),
        bath_coupling_kind=np.asarray("complex" if args.complex_bath else "real"),
        bath_fit_metric=np.asarray(str(args.bath_fit_metric)),
        mixing_method=np.asarray(str(result.mixing_method)),
        mixer_fallbacks=int(result.mixer_fallbacks),
        residual_history=np.asarray(result.residual_history),
        impurity_residual_history=np.asarray(result.impurity_residual_history),
        impurity_mismatch_history=np.asarray(result.impurity_mismatch_history),
        bath_fit_history=np.asarray(result.bath_fit_history),
        mu_history=np.asarray(result.mu_history),
        elapsed_history=np.asarray(result.elapsed_history),
        c3_residual_history=np.asarray(result.c3_residual_history),
        G_background=np.asarray(result.background.G),
        mu_background=float(result.background.mu),
    )
    print(
        f"saved {outfile}\n"
        f"converged={result.converged}, residual={result.final_error:.3e}, "
        f"mu={result.mu:+.10f}, density={np.array2string(result.density, precision=7)}\n"
        f"Gimp/Gc ori0/1/2="
        f"{result.impurity_mismatch_by_orientation[0]:.3e}/"
        f"{result.impurity_mismatch_by_orientation[1]:.3e}/"
        f"{result.impurity_mismatch_by_orientation[2]:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
