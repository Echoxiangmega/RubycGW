#!/usr/bin/env python3
"""Run ordinary cluster ED+GW in one of three C3-related cluster gauges.

This is the symmetry-safe alternative to forcing a single oriented six-site
impurity map onto a C3-projected lattice fixed point.  The impurity ED and its cluster-GW double-counting subtraction treat the strong
intra-triangle V together with exactly one real neighbouring A-B pair carrying
two Vprime and two Vcross bonds.  Distinct intercell neighbours are never
collapsed onto the same impurity orbital pair; the remaining directions stay
in the full lattice GW interaction.  The physical lattice
Hamiltonian is unchanged; only the primitive-cell assignment of the B triangle
is shifted so that the six-site impurity internalizes one of the three
C3-related neighbouring A-B pairs.

Run orientations 0,1,2 separately.  Each calculation uses the ordinary
unconstrained production embedding and should therefore retain its usual
convergence properties.  The three outputs can subsequently be transformed
back to one common gauge and averaged/compared.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution as:
#   python research\run_cluster_ed_gw_vprime_vcross_orientation.py ...
# without requiring an editable package install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import rubycgw.cluster_ed_gw_fast as _fast_solver
from rubycgw.cluster_ed_gw_covariant import fit_finite_bath_complex
from rubycgw.cluster_ed_gw_fast import ClusterEDGWFastOptions, solve_cluster_ed_gw_fast
from rubycgw.cluster_ed_gw_free_energy import evaluate_cluster_ed_gw_free_energy
from rubycgw.cluster_orientation import (
    build_oriented_lattice_fields,
    orientation_b_shift,
    rotate_local_between_orientations,
    rotate_solution_between_orientations,
)
from rubycgw.cluster_restart import ClusterEDGWRestartState, load_cluster_ed_gw_restart
from rubycgw.cluster_restart_solver import (
    _continuation_background,
    solve_cluster_ed_gw_fast_continued,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import build_h0
from rubycgw.models.ruby import physical_pair_cluster_interactions
from rubycgw.pulay_accel import install_scale_invariant_pulay
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
    reference_pair_effective_couplings,
)
from vprime_study.patches import install_cluster_interaction_hooks


install_scale_invariant_pulay()


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orientation", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=3)
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

    p.add_argument("--gw-max", type=int, default=160)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")

    p.add_argument("--embed-max", type=int, default=150)
    p.add_argument("--embed-tol", type=float, default=2e-5)
    p.add_argument("--embed-mixing", type=float, default=0.80)
    p.add_argument("--embed-mixing-method", choices=("linear", "pulay", "broyden"), default="pulay")
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
    p.add_argument(
        "--bath-fit-metric",
        choices=("delta", "g0"),
        default="delta",
        help="fit bath hybridization Delta or the low-frequency Weiss Green function G0",
    )
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument(
        "--complex-bath",
        action="store_true",
        help="fit the same number of bath orbitals with complex rather than real couplings",
    )
    p.add_argument(
        "--source-mode-file",
        type=Path,
        default=None,
        help="tracking-ready lambda_allq NPZ containing mode_static_matrix",
    )
    p.add_argument(
        "--source-mode-row",
        type=int,
        default=None,
        help="global mode row in --source-mode-file; currently q=(0,0) only",
    )
    p.add_argument(
        "--source-strength",
        type=float,
        default=0.0,
        help="static one-body source h in H_source=-h O_mode",
    )
    p.add_argument(
        "--evaluate-free-energy",
        action="store_true",
        help="evaluate the cluster-ED+GW Luttinger-Ward functional for the returned state",
    )
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--quiet-gw", action="store_true")
    p.add_argument("--quiet-embed", action="store_true")
    p.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        help=(
            "seed from an existing V'+Vx checkpoint.  If its saved cluster "
            "orientation differs from the requested one, the C3-related "
            "lattice G/Sigma/Hartree state is used as the warm start while "
            "the impurity/bath is rebuilt for the new cluster cut"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/vprime_vcross_cluster_orientation"),
    )
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _saved_scalar(path: Path, names, default):
    with np.load(path, allow_pickle=False) as z:
        for name in names:
            if name in z:
                return np.asarray(z[name]).reshape(()).item()
    return default


def _saved_interactions(path: Path) -> tuple[float, float]:
    vp = float(_saved_scalar(path, ("Vprime", "Vp"), 0.0))
    vx = float(_saved_scalar(path, ("Vcross", "Vx"), 0.0))
    return vp, vx


def _saved_orientation(path: Path) -> int:
    return int(_saved_scalar(path, ("cluster_orientation",), 0))


def _load_q0_source_template(path: Path, row: int) -> tuple[np.ndarray, tuple[int, int], str]:
    with np.load(path, allow_pickle=False) as z:
        if "mode_static_matrix" not in z or "mode_q_index" not in z:
            raise ValueError(
                f"{path}: JF file lacks mode_static_matrix; rerun the updated JF analyzer"
            )
        mats = np.asarray(z["mode_static_matrix"], dtype=complex)
        q = np.asarray(z["mode_q_index"], dtype=int)
        sectors = (
            np.asarray(z["mode_sector"]).astype(str)
            if "mode_sector" in z
            else np.full(len(mats), "unknown")
        )
    row = int(row)
    if row < 0 or row >= len(mats):
        raise IndexError(f"source mode row {row} outside [0,{len(mats)})")
    qn = tuple(int(x) for x in q[row])
    if qn != (0, 0):
        raise ValueError(
            "primitive-cell source runner can only realize q=(0,0) order; "
            f"requested q={qn}. Use the finite-q supercell branch workflow."
        )
    M = np.asarray(mats[row], dtype=complex)
    M = 0.5 * (M + M.conj().T)
    M = M - np.trace(M) * np.eye(M.shape[0], dtype=complex) / float(M.shape[0])
    scale = float(np.max(np.abs(M), initial=0.0))
    if not np.isfinite(scale) or scale <= 1e-14:
        raise ValueError("selected JF mode has a vanishing static traceless source form factor")
    M = M / scale
    return M, qn, str(sectors[row])


def _lattice_reseed_for_orientation(
    restart: ClusterEDGWRestartState,
    *,
    source_orientation: int,
    target_orientation: int,
) -> ClusterEDGWRestartState:
    """Map a saved branch to its C3-related partner in the target cluster frame.

    A pure unit-cell gauge change would keep the same symmetry-broken physical
    state while changing only which B triangle is called intracell.  That is
    not the desired warm start when comparing the three C3-related cluster
    orientations: the target should instead receive the actively C3-rotated
    physical partner, expressed in its own orientation gauge.
    """
    G = rotate_solution_between_orientations(
        restart.G, source_orientation, target_orientation
    )
    Sigma_emb = rotate_solution_between_orientations(
        restart.Sigma_emb, source_orientation, target_orientation
    )
    Sigma_H = rotate_local_between_orientations(
        restart.Sigma_H,
        source_orientation,
        target_orientation,
        nk1=restart.G.shape[-4],
        nk2=restart.G.shape[-3],
    )
    return ClusterEDGWRestartState(
        G=np.asarray(G),
        Sigma_H=np.asarray(Sigma_H),
        Sigma_emb=np.asarray(Sigma_emb),
        Sigma_imp=np.array(restart.Sigma_imp, copy=True),
        mu=float(restart.mu),
        bath=restart.bath,
        source_path=restart.source_path,
        source_V=float(restart.source_V),
    )


def main():
    args = _args()
    if args.Lx < 1 or args.Ly < 1:
        raise ValueError("Lx and Ly must be positive")

    params = VPrimeCrossParameters(
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        V=float(args.V), Vprime=float(args.Vprime), Vcross=float(args.Vcross),
    )

    source_matrix = None
    source_q = (-1, -1)
    source_sector = ""
    if args.source_mode_file is not None:
        if args.source_mode_row is None:
            raise ValueError("--source-mode-file requires --source-mode-row")
        source_matrix, source_q, source_sector = _load_q0_source_template(
            args.source_mode_file, int(args.source_mode_row)
        )
        if np.max(np.abs(source_matrix.imag), initial=0.0) > 1e-12:
            args.complex_bath = True
            print(
                "[source] complex q=0 form factor detected: enabling complex bath",
                flush=True,
            )
    elif args.source_mode_row is not None:
        raise ValueError("--source-mode-row requires --source-mode-file")
    elif not np.isclose(float(args.source_strength), 0.0):
        raise ValueError("nonzero --source-strength requires --source-mode-file")

    if bool(args.complex_bath):
        if str(args.bath_fit_metric) != "delta":
            raise ValueError("--complex-bath currently supports only --bath-fit-metric delta")
        _fast_solver.fit_finite_bath = fit_finite_bath_complex

    # Treat exactly one real A-B neighbour pair in the six-site impurity.
    # The other two directions remain in the full lattice GW interaction.
    install_cluster_interaction_hooks(
        lambda p: physical_pair_cluster_interactions(p, int(args.orientation))
    )
    grid = MatsubaraGrid(
        nk1=int(args.Lx), nk2=int(args.Ly), nw=int(args.nw),
        nOmega=int(args.nomega), T=float(args.T),
    )
    h0_base = build_h0(grid.kmesh(), params)
    Vq_base = build_vprime_vcross_interaction(grid.qmesh(), params)
    h0, Vq = build_oriented_lattice_fields(h0_base, Vq_base, int(args.orientation))
    if source_matrix is not None and not np.isclose(float(args.source_strength), 0.0):
        h0 = np.asarray(h0, dtype=complex) - float(args.source_strength) * source_matrix[
            None, None, :, :
        ]

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
        broyden_history=int(args.embed_broyden_history),
        broyden_regularization=float(args.embed_broyden_regularization),
        broyden_step_cap=float(args.embed_broyden_step_cap),
        broyden_reset_growth=float(args.embed_broyden_reset_growth),
        impurity_mixing=float(args.impurity_mixing),
        nbath=int(args.nbath), bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_fit_metric=str(args.bath_fit_metric),
        bath_energy_window=float(args.bath_energy_window),
        bath_coupling_bound=float(args.bath_coupling_bound),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not bool(args.quiet_embed),
    )

    shift = orientation_b_shift(int(args.orientation))
    pair_terms = physical_pair_cluster_interactions(params, int(args.orientation))
    pair_intercell = [(i, j, u) for i, j, u in pair_terms if (i < 3) != (j < 3)]
    print(
        "=== Ruby V-prime + V-cross orientation-resolved cluster ED+GW ===\n"
        f"orientation={args.orientation}, B-cell shift=({shift[0]},{shift[1]})\n"
        f"L={args.Lx}x{args.Ly}, V={args.V:g}, V'={args.Vprime:g}, "
        f"Vx={args.Vcross:g}, filling={args.filling:g}, T={args.T:g}\n"
        f"ED physical intercell pair terms={pair_intercell}\n"
        f"bath couplings={'complex' if args.complex_bath else 'real'}, "
        f"metric={args.bath_fit_metric}\n"
        f"source h={args.source_strength:g}, q={source_q}, sector={source_sector or 'none'}",
        flush=True,
    )
    if np.isclose(float(args.t1), float(args.t2), rtol=0.0, atol=1e-14) and args.V != 0.0:
        jn, jm, jz = reference_pair_effective_couplings(
            t=float(args.t1), V=float(args.V),
            Vprime=float(args.Vprime), Vcross=float(args.Vcross),
        )
        print(
            "[strong-coupling diagnostic] "
            f"Jn={jn:+.6e}, Jm={jm:+.6e}, Jz={jz:+.6e}",
            flush=True,
        )

    continuation_source = ""
    source_V = np.nan
    source_Vprime = np.nan
    source_Vcross = np.nan
    source_orientation = -1
    seed_mode = "standalone_scgw"

    if args.continue_from is None:
        result = solve_cluster_ed_gw_fast(
            h0, Vq, params, grid, gw_opts=gw_opts, embed_opts=embed_opts
        )
    else:
        source_projection = str(_saved_scalar(
            args.continue_from, ("cluster_projection",), "legacy_unknown"
        ))
        if source_projection != "physical_pair_no_intercell_collapse":
            raise ValueError(
                "continuation checkpoint uses a different cluster interaction partition "
                f"({source_projection!r}). Start a fresh physical-pair ED+GW run "
                "instead of reusing an impurity self-energy from an older scheme."
            )
        restart = load_cluster_ed_gw_restart(
            args.continue_from,
            Lx=int(args.Lx), Ly=int(args.Ly),
            filling=float(args.filling), T=float(args.T),
            params=params, grid=grid, nbath=int(args.nbath),
            allow_interaction_change=True,
            allow_filling_change=True,
        )
        continuation_source = str(args.continue_from)
        source_V = float(restart.source_V)
        source_Vprime, source_Vcross = _saved_interactions(args.continue_from)
        source_filling = float(_saved_scalar(
            args.continue_from, ("filling",), args.filling
        ))
        source_orientation = _saved_orientation(args.continue_from)
        print(
            "[cluster-ED+GW:orientation] seed: "
            f"ori {source_orientation} -> {args.orientation}; "
            f"(V,V',Vx,n)=({source_V:g},{source_Vprime:g},{source_Vcross:g},{source_filling:g}) -> "
            f"({args.V:g},{args.Vprime:g},{args.Vcross:g},{args.filling:g})",
            flush=True,
        )

        if int(source_orientation) == int(args.orientation):
            seed_mode = "full_parameter_continuation"
            result = solve_cluster_ed_gw_fast_continued(
                h0, Vq, params, grid,
                gw_opts=gw_opts, embed_opts=embed_opts, restart=restart,
            )
        else:
            # A different cluster cut has a different impurity Weiss problem.
            # Seed it with the actively C3-related physical partner, expressed
            # in the target orientation gauge.  The local impurity self-energy
            # and finite bath are rebuilt rather than copied across cuts.
            seed_mode = "c3_rotated_lattice_reseed"
            oriented_restart = _lattice_reseed_for_orientation(
                restart,
                source_orientation=int(source_orientation),
                target_orientation=int(args.orientation),
            )
            carrier = _continuation_background(oriented_restart, grid)
            result = solve_cluster_ed_gw_fast(
                h0, Vq, params, grid,
                gw_opts=gw_opts, embed_opts=embed_opts, background=carrier,
            )

    thermo_payload = {}
    if bool(args.evaluate_free_energy):
        thermo = evaluate_cluster_ed_gw_free_energy(
            result,
            h0,
            Vq,
            physical_pair_cluster_interactions(params, int(args.orientation)),
            grid,
            target_particles=float(args.filling),
            discard_weight_tol=float(args.discard_weight_tol),
        )
        thermo_payload = {
            "free_energy_method": np.asarray("cluster_ed_gw_luttinger_ward"),
            "grand_potential": float(thermo.grand_potential),
            "helmholtz_free_energy": float(thermo.helmholtz_free_energy),
            "omega0_lattice": float(thermo.omega0_lattice),
            "fermionic_lw_lattice": float(thermo.fermionic_lw_lattice),
            "phi_gw_lattice": float(thermo.phi_gw_lattice),
            "phi_gw_cluster": float(thermo.phi_gw_cluster),
            "phi_ed_cluster": float(thermo.phi_ed_cluster),
            "phi_cluster_correction": float(thermo.phi_cluster_correction),
            "phi_embedded_total": float(thermo.phi_embedded_total),
            "free_energy_particle_number_actual": float(thermo.particle_number_actual),
            "free_energy_particle_number_legendre": float(
                thermo.particle_number_legendre
            ),
            "impurity_grand_potential": float(thermo.impurity_grand_potential),
            "impurity_internal_energy": float(thermo.impurity_internal_energy),
            "impurity_entropy": float(thermo.impurity_entropy),
            "impurity_particle_number": float(thermo.impurity_particle_number),
            "free_energy_gimp_gc_mismatch": float(
                thermo.gimp_gc_relative_mismatch
            ),
            "free_energy_gimp_reconstruction_mismatch": float(
                thermo.gimp_reconstruction_mismatch
            ),
        }
        print(
            "[thermo] "
            f"F={thermo.helmholtz_free_energy:+.12e}, "
            f"Omega={thermo.grand_potential:+.12e}, "
            f"Phi_ED-Phi_GWc={thermo.phi_cluster_correction:+.6e}, "
            f"Gimp/Gc={thermo.gimp_gc_relative_mismatch:.3e}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    source_suffix = ""
    if source_matrix is not None:
        source_suffix = (
            f"_mode{int(args.source_mode_row)}_h{_tag(args.source_strength)}"
        )
    outfile = args.out / (
        f"cluster_ed_gw_vprime_vcross_ori{args.orientation}_L{args.Lx}x{args.Ly}_"
        f"V{_tag(args.V)}_Vp{_tag(args.Vprime)}_Vx{_tag(args.Vcross)}_"
        f"fill{_tag(args.filling)}{source_suffix}.npz"
    )
    np.savez_compressed(
        outfile,
        interaction_model=np.asarray("lattice_full_V_Vprime_Vcross__cluster_ED_physical_pair"),
        cluster_projection=np.asarray("physical_pair_no_intercell_collapse"),
        embedding_scheme=np.asarray("ED(V+one_real_pair_Vprime_Vcross)+GW(full_lattice)"),
        cluster_orientation=int(args.orientation),
        cluster_b_shift=np.asarray(shift, dtype=int),
        orientation_method=np.asarray("B_triangle_cell_gauge_three_orientation_ensemble"),
        orientation_seed_mode=np.asarray(seed_mode),
        continuation_source=np.asarray(continuation_source),
        continuation_source_V=float(source_V),
        continuation_source_Vprime=float(source_Vprime),
        continuation_source_Vcross=float(source_Vcross),
        continuation_source_orientation=int(source_orientation),
        background_kind=np.asarray(
            "orientation_parameter_continuation" if args.continue_from is not None
            else "orientation_standalone_scgw"
        ),
        Lx=int(args.Lx), Ly=int(args.Ly),
        V=float(args.V), Vprime=float(args.Vprime), Vp=float(args.Vprime),
        Vcross=float(args.Vcross), Vx=float(args.Vcross),
        filling=float(args.filling), T=float(args.T),
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        omega=np.asarray(grid.omega), Omega=np.asarray(grid.Omega),
        converged=bool(result.converged), iterations=int(result.iterations),
        final_error=float(result.final_error),
        impurity_mismatch=float(result.impurity_mismatch),
        bath_fit_error=float(result.bath_fit_error),
        bath_fit_metric=np.asarray(str(args.bath_fit_metric)),
        mixing_method=str(result.mixing_method),
        pulay_fallbacks=int(result.pulay_fallbacks),
        mixer_fallbacks=int(result.pulay_fallbacks),
        broyden_history=int(args.embed_broyden_history),
        broyden_regularization=float(args.embed_broyden_regularization),
        broyden_step_cap=float(args.embed_broyden_step_cap),
        broyden_reset_growth=float(args.embed_broyden_reset_growth),
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
        impurity_static_shift=np.asarray(result.impurity_static_shift),
        G_cluster=np.asarray(result.G_cluster), G_impurity=np.asarray(result.G_impurity),
        bath_energies=np.asarray(result.bath.energies),
        bath_couplings=np.asarray(result.bath.couplings),
        bath_coupling_kind=np.asarray("complex" if args.complex_bath else "real"),
        source_strength=float(args.source_strength),
        source_mode_file=np.asarray(
            "" if args.source_mode_file is None else str(args.source_mode_file)
        ),
        source_mode_row=int(-1 if args.source_mode_row is None else args.source_mode_row),
        source_q=np.asarray(source_q, dtype=int),
        source_sector=np.asarray(source_sector),
        source_matrix=(
            np.zeros((6, 6), dtype=complex)
            if source_matrix is None else np.asarray(source_matrix, dtype=complex)
        ),
        G_background=np.asarray(result.background.G),
        mu_background=float(result.background.mu),
        **thermo_payload,
    )
    print(
        f"saved {outfile}\n"
        f"converged={result.converged}, residual={result.final_error:.3e}, "
        f"bath={result.bath_fit_error:.3e}, Gimp/Gc={result.impurity_mismatch:.3e}, "
        f"mu={result.mu:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
