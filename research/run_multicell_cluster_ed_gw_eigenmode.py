#!/usr/bin/env python3
"""Run one commensurate eigenmode-seeded multi-impurity cluster-ED+GW state.

The primitive Lx x Ly torus of a converged physical-pair checkpoint is folded
to one 6*Lx*Ly orbital supercell.  A static q/-q source derived from a retained
JF mode is applied and each primitive cell receives its own six-site finite-bath
ED impurity.  q=0 and finite-q branches therefore share exactly the same
nonlinear approximation and can be compared with the same Luttinger-Ward free
energy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions,
    MultiCellRestart,
    solve_multicell_cluster_ed_gw,
)
from rubycgw.cluster_ed_gw_multicell_free_energy import evaluate_multicell_free_energy
from rubycgw.cluster_orientation import build_oriented_lattice_fields
from rubycgw.folded_torus import (
    commensurate_source_matrix,
    fold_dynamic_k_field,
    fold_static_field_as_grid,
    torus_cells,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import build_h0
from rubycgw.models.ruby import physical_pair_cluster_interactions
from rubycgw.pseudospin import primitive_cell_pseudospin_channels
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--mode-file", type=Path, default=None)
    p.add_argument("--mode-row", type=int, default=None)
    p.add_argument("--source-strength", type=float, default=0.0)
    p.add_argument("--continue-from", type=Path, default=None)
    p.add_argument("--max-iter", type=int, default=160)
    p.add_argument("--tol", type=float, default=2e-6)
    p.add_argument("--mixing", type=float, default=0.45)
    p.add_argument("--impurity-mixing", type=float, default=1.0)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=400)
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--real-bath", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _scalar(d, key, cast=float, default=None):
    if key not in d:
        if default is None:
            raise KeyError(key)
        return cast(default)
    return cast(np.asarray(d[key]).reshape(()))


def _tag(x):
    return f"{float(x):g}"


def _source_from_mode(path, row, Lx, Ly):
    d = _load(path)
    if "mode_static_matrix" not in d or "mode_q_index" not in d:
        raise ValueError("mode file lacks source-ready JF spectrum fields")
    row = int(row)
    M = np.asarray(d["mode_static_matrix"][row], dtype=complex)
    q = tuple(int(x) for x in np.asarray(d["mode_q_index"][row], dtype=int))
    sector = (
        str(np.asarray(d["mode_sector"][row]))
        if "mode_sector" in d else "unknown"
    )
    O = commensurate_source_matrix(M, q, Lx, Ly, normalize=True)
    return O, q, sector


def _primitive_restart(bg, ncell):
    sigma_h6 = np.asarray(bg["Sigma_H"], dtype=complex)
    sigma_h = np.kron(np.eye(ncell, dtype=complex), sigma_h6)
    sigma_emb = fold_dynamic_k_field(np.asarray(bg["Sigma_emb"], dtype=complex))
    G = fold_dynamic_k_field(np.asarray(bg["G"], dtype=complex))
    simp = np.asarray(bg["Sigma_ED_cluster"], dtype=complex)
    sigma_imp = np.repeat(simp[None, :, :, :], ncell, axis=0)
    eps = np.asarray(bg["bath_energies"], dtype=float)
    hyb = np.asarray(bg["bath_couplings"], dtype=complex)
    bath_energies = np.repeat(eps[None, :], ncell, axis=0)
    bath_couplings = np.repeat(hyb[None, :, :], ncell, axis=0)
    return MultiCellRestart(
        Sigma_H=sigma_h,
        Sigma_emb=sigma_emb,
        Sigma_imp=sigma_imp,
        G=G,
        mu=_scalar(bg, "mu", float),
        bath_energies=bath_energies,
        bath_couplings=bath_couplings,
    )


def _multicell_restart(path):
    d = _load(path)
    return MultiCellRestart(
        Sigma_H=np.asarray(d["Sigma_H"], dtype=complex),
        Sigma_emb=np.asarray(d["Sigma_emb"], dtype=complex),
        Sigma_imp=np.asarray(d["Sigma_ED_cluster"], dtype=complex),
        G=np.asarray(d["G"], dtype=complex),
        mu=_scalar(d, "mu", float),
        bath_energies=np.asarray(d["bath_energies"], dtype=float),
        bath_couplings=np.asarray(d["bath_couplings"], dtype=complex),
    )


def _order_spectrum(rho_sc: np.ndarray, Lx: int, Ly: int):
    channels = primitive_cell_pseudospin_channels()
    names = ("x_even", "y_even", "x_odd", "y_odd", "z_even", "z_odd")
    cells = torus_cells(Lx, Ly)
    local = np.zeros((len(names), len(cells)), dtype=complex)
    rho_sc = np.asarray(rho_sc, dtype=complex)
    for ic, _ in enumerate(cells):
        sl = slice(6*ic, 6*(ic+1))
        rc = rho_sc[sl, sl]
        for ia, name in enumerate(names):
            K = np.asarray(channels[name], dtype=complex)
            den = max(float(np.vdot(K, K).real), 1e-300)
            local[ia, ic] = np.vdot(K, rc) / den

    spec = np.zeros((len(names), Lx, Ly), dtype=complex)
    for ia in range(len(names)):
        for q1 in range(Lx):
            for q2 in range(Ly):
                acc = 0.0j
                for ic, (r1, r2) in enumerate(cells):
                    phase = np.exp(
                        -2j * np.pi * (
                            q1 * r1 / float(Lx)
                            + q2 * r2 / float(Ly)
                        )
                    )
                    acc += phase * local[ia, ic]
                spec[ia, q1, q2] = acc / float(len(cells))
    flat = int(np.argmax(np.abs(spec)))
    ia, q1, q2 = np.unravel_index(flat, spec.shape)
    return names, local, spec, names[ia], (int(q1), int(q2)), complex(spec[ia, q1, q2])


def main():
    args = _args()
    bg = _load(args.background)
    if not bool(_scalar(bg, "converged", bool, True)):
        raise ValueError("background checkpoint is not converged")

    Lx = _scalar(bg, "Lx", int)
    Ly = _scalar(bg, "Ly", int)
    ncell = int(Lx * Ly)
    orientation = _scalar(bg, "cluster_orientation", int)
    filling = _scalar(bg, "filling", float)
    T = _scalar(bg, "T", float)
    nw = len(np.asarray(bg["omega"])) // 2
    nOmega = (len(np.asarray(bg["Omega"])) - 1) // 2
    nbath = len(np.asarray(bg["bath_energies"]))

    params = VPrimeCrossParameters(
        ti=_scalar(bg, "ti", float),
        t1=_scalar(bg, "t1", float),
        t2=_scalar(bg, "t2", float),
        V=_scalar(bg, "V", float),
        Vprime=_scalar(bg, "Vprime", float),
        Vcross=_scalar(bg, "Vcross", float),
    )
    pgrid = MatsubaraGrid(
        nk1=Lx, nk2=Ly, nw=nw, nOmega=nOmega, T=T
    )
    h0b = build_h0(pgrid.kmesh(), params)
    Vqb = build_vprime_vcross_interaction(pgrid.qmesh(), params)
    h0p, Vqp = build_oriented_lattice_fields(h0b, Vqb, orientation)

    h0sc = fold_static_field_as_grid(h0p)
    Vqsc = fold_static_field_as_grid(Vqp)
    source = np.zeros_like(h0sc[0, 0])
    q = (-1, -1)
    sector = "none"
    if args.mode_file is not None:
        if args.mode_row is None:
            raise ValueError("--mode-file requires --mode-row")
        source, q, sector = _source_from_mode(
            args.mode_file, args.mode_row, Lx, Ly
        )
        h0sc = np.asarray(h0sc, dtype=complex) - float(args.source_strength) * source[
            None, None, :, :
        ]
    elif args.mode_row is not None:
        raise ValueError("--mode-row requires --mode-file")
    elif not np.isclose(float(args.source_strength), 0.0):
        raise ValueError("nonzero source requires --mode-file")

    scgrid = MatsubaraGrid(
        nk1=1, nk2=1, nw=nw, nOmega=nOmega, T=T
    )
    if args.continue_from is None:
        restart = _primitive_restart(bg, ncell)
        seed_kind = "folded_primitive_background"
    else:
        restart = _multicell_restart(args.continue_from)
        seed_kind = "multicell_continuation"

    interactions = physical_pair_cluster_interactions(params, orientation)
    opts = MultiCellEDGWOptions(
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        mixing=float(args.mixing),
        impurity_mixing=float(args.impurity_mixing),
        nbath=int(nbath),
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        discard_weight_tol=float(args.discard_weight_tol),
        complex_bath=not bool(args.real_bath),
        verbose=not bool(args.quiet),
    )

    print(
        "=== finite-torus multi-impurity cluster ED+GW ===\n"
        f"primitive mesh={Lx}x{Ly}, ncell={ncell}, folded norb={6*ncell}\n"
        f"V={params.V:g}, V'={params.Vprime:g}, Vx={params.Vcross:g}, "
        f"filling/pc={filling:g}, T={T:g}\n"
        f"source q={q}, sector={sector}, h={args.source_strength:g}, "
        f"seed={seed_kind}",
        flush=True,
    )
    result = solve_multicell_cluster_ed_gw(
        h0sc,
        Vqsc,
        interactions,
        scgrid,
        target_particles=float(filling * ncell),
        opts=opts,
        restart=restart,
    )

    thermo = evaluate_multicell_free_energy(
        result,
        h0sc,
        Vqsc,
        interactions,
        scgrid,
        target_particles=float(filling * ncell),
        discard_weight_tol=float(args.discard_weight_tol),
    )
    rho = one_body_density_matrix_tail(
        result.G, scgrid, h0sc, result.mu, result.Sigma_H
    )
    if np.max(np.abs(source), initial=0.0) > 0:
        source_expect = np.vdot(source, rho[0, 0])
        source_expect = complex(source_expect / float(ncell))
    else:
        source_expect = 0.0j
    (
        order_names,
        order_local,
        order_spectrum,
        dominant_order_channel,
        dominant_order_q,
        dominant_order_amplitude,
    ) = _order_spectrum(rho[0, 0], Lx, Ly)

    args.out.mkdir(parents=True, exist_ok=True)
    qtag = "normal" if q == (-1, -1) else f"q{q[0]}_{q[1]}_mode{int(args.mode_row)}"
    outfile = args.out / (
        f"multicell_ed_gw_ori{orientation}_L{Lx}x{Ly}_"
        f"V{_tag(params.V)}_fill{_tag(filling)}_{qtag}_"
        f"h{_tag(args.source_strength)}.npz"
    )
    np.savez_compressed(
        outfile,
        solver=np.asarray("multicell_physical_pair_cluster_ed_gw"),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        primitive_Lx=int(Lx),
        primitive_Ly=int(Ly),
        ncell=int(ncell),
        cluster_orientation=int(orientation),
        V=float(params.V),
        Vprime=float(params.Vprime),
        Vcross=float(params.Vcross),
        filling=float(filling),
        target_particles_supercell=float(filling*ncell),
        T=float(T),
        ti=float(params.ti),
        t1=float(params.t1),
        t2=float(params.t2),
        omega=np.asarray(scgrid.omega),
        Omega=np.asarray(scgrid.Omega),
        source_strength=float(args.source_strength),
        source_q=np.asarray(q, dtype=int),
        source_sector=np.asarray(sector),
        source_mode_file=np.asarray("" if args.mode_file is None else str(args.mode_file)),
        source_mode_row=int(-1 if args.mode_row is None else args.mode_row),
        source_matrix=np.asarray(source),
        source_expectation_per_pc=complex(source_expect),
        order_channel_names=np.asarray(order_names),
        order_local_amplitudes=np.asarray(order_local),
        order_spectrum=np.asarray(order_spectrum),
        dominant_order_channel=np.asarray(dominant_order_channel),
        dominant_order_q=np.asarray(dominant_order_q, dtype=int),
        dominant_order_amplitude=complex(dominant_order_amplitude),
        seed_kind=np.asarray(seed_kind),
        continuation_source=np.asarray(
            "" if args.continue_from is None else str(args.continue_from)
        ),
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H),
        Sigma_emb=np.asarray(result.Sigma_emb),
        Sigma_GW_lattice=np.asarray(result.Sigma_GW_lattice),
        Sigma_GW_cluster=np.asarray(result.Sigma_GW_cluster),
        Sigma_ED_cluster=np.asarray(result.Sigma_ED_cluster),
        G_cluster=np.asarray(result.G_cluster),
        G_impurity=np.asarray(result.G_impurity),
        mu=float(result.mu),
        density=np.asarray(result.density),
        bath_energies=np.asarray(result.bath_energies),
        bath_couplings=np.asarray(result.bath_couplings),
        bath_fit_error=np.asarray(result.bath_fit_error),
        impurity_static_shift=np.asarray(result.impurity_static_shift),
        impurity_mismatch=np.asarray(result.impurity_mismatch),
        residual_history=np.asarray(result.residual_history),
        elapsed_history=np.asarray(result.elapsed_history),
        free_energy_method=np.asarray("multicell_cluster_ed_gw_luttinger_ward"),
        grand_potential_supercell=float(thermo.grand_potential_supercell),
        helmholtz_free_energy_supercell=float(thermo.helmholtz_free_energy_supercell),
        helmholtz_free_energy_per_primitive_cell=float(
            thermo.helmholtz_free_energy_per_primitive_cell
        ),
        phi_gw_lattice=float(thermo.phi_gw_lattice),
        phi_gw_cluster_sum=float(thermo.phi_gw_cluster_sum),
        phi_ed_cluster_sum=float(thermo.phi_ed_cluster_sum),
        phi_cluster_correction_sum=float(thermo.phi_cluster_correction_sum),
        free_energy_particle_number_actual=float(thermo.particle_number_actual),
        free_energy_particle_number_legendre=float(thermo.particle_number_legendre),
        free_energy_gimp_gc_mismatch=np.asarray(thermo.gimp_gc_relative_mismatch),
        free_energy_gimp_reconstruction_mismatch=np.asarray(
            thermo.gimp_reconstruction_mismatch
        ),
        impurity_grand_potential=np.asarray(thermo.impurity_grand_potential),
        impurity_internal_energy=np.asarray(thermo.impurity_internal_energy),
        impurity_entropy=np.asarray(thermo.impurity_entropy),
    )
    print(
        f"saved {outfile}\n"
        f"converged={result.converged}, residual={result.final_error:.3e}, "
        f"max Gimp/Gc={np.max(result.impurity_mismatch):.3e}, "
        f"F/pc={thermo.helmholtz_free_energy_per_primitive_cell:+.12e}, "
        f"<Osrc>/pc={source_expect.real:+.6e}{source_expect.imag:+.2e}i, "
        f"dominant={dominant_order_channel}@{dominant_order_q} "
        f"|A|={abs(dominant_order_amplitude):.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
