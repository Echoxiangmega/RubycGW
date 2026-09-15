#!/usr/bin/env python3
"""Scan the 18-site exact-diagonalization spectrum and correlations versus V.

The calculation uses the same three-cell periodic torus as ``rubycgw.supercell``.
At primitive filling n the fixed particle number is Nf=3n, when this is an
integer.  The 18-site cluster contains only primitive Gamma and
+/-Q=(1/3,1/3); it cannot represent the period-two M point.

By default the script computes the ground-state spectrum and equal-time
six-channel pseudospin structure factor.  With ``--with-chi`` it additionally
computes the exact fixed-N zero-temperature static susceptibility using
correction-vector solves.  Exact ground-manifold zero modes are handled
explicitly: the finite excited-state contribution chi^reg and the coefficient
C^GS of any T->0 singular/Curie contribution are stored separately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rubycgw.ed18 import (
    ED18_CHANNELS,
    ED18Solver,
    Q_GAMMA,
    Q_PERIOD3,
    phase_fix_vector,
    translation_phase_reduced,
)
from rubycgw.ed18_chi import zero_temperature_susceptibility
from rubycgw.ed18_chi_plot import save_ed18_chi_plot
from rubycgw.ed18_plot import (
    ground_sector_label,
    leading_subspace_weights,
    save_ed18_plots,
)
from rubycgw.model import RubyParameters


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Vmin", type=float, default=0.0)
    p.add_argument("--Vmax", type=float, default=2.0)
    p.add_argument("--dV", type=float, default=0.05)
    p.add_argument("--V-list", nargs="*", type=float, default=None)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--n-eigs", type=int, default=10)
    p.add_argument("--eig-tol", type=float, default=1e-10)
    p.add_argument("--deg-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--out", default="ed18_n2_vscan.npz")
    p.add_argument("--csv", default=None)
    p.add_argument("--plot", default=None, help="Summary PNG path. Default: same stem as --out.")
    p.add_argument(
        "--modes-plot",
        default=None,
        help="Equal-time channel-resolved PNG. Default: <out_stem>_modes.png.",
    )
    p.add_argument("--print-modes", type=int, default=3)
    p.add_argument("--top-spectrum", type=int, default=6)
    p.add_argument("--top-structure", type=int, default=3)
    p.add_argument("--dpi", type=int, default=180)

    p.add_argument(
        "--with-chi",
        action="store_true",
        help="Also compute exact fixed-N T=0 static susceptibility at Gamma and Q.",
    )
    p.add_argument("--chi-tol", type=float, default=1e-10, help="Correction-vector CG relative tolerance.")
    p.add_argument("--chi-maxiter", type=int, default=20000)
    p.add_argument(
        "--chi-singular-tol",
        type=float,
        default=1e-10,
        help="Tolerance for detecting nontrivial operator covariance inside an exact GS manifold.",
    )
    p.add_argument(
        "--chi-plot",
        default=None,
        help="T=0 susceptibility PNG. Default: <out_stem>_chi.png when --with-chi.",
    )
    p.add_argument("--chi-top-modes", type=int, default=3)
    return p.parse_args()


def _v_grid(args):
    if args.V_list:
        return np.asarray(sorted(set(float(x) for x in args.V_list)), dtype=float)
    if args.dV <= 0:
        raise ValueError("--dV must be positive")
    n = int(np.floor((args.Vmax - args.Vmin) / args.dV + 0.5))
    out = args.Vmin + args.dV * np.arange(n + 1, dtype=float)
    return out[out <= args.Vmax + 1e-12]


def _mode_text(channels, vec, threshold=0.08):
    vec = phase_fix_vector(vec)
    weights = np.abs(vec) ** 2
    pieces = []
    for i in np.argsort(weights)[::-1]:
        if weights[i] < threshold:
            continue
        z = vec[i]
        pieces.append(
            f"{channels[i]}:{weights[i]:.3f}"
            f"({z.real:+.3f}{z.imag:+.3f}j)"
        )
    return " ".join(pieces) if pieces else "mixed"


def main():
    args = _args()
    Vvals = _v_grid(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    solver = ED18Solver(params, primitive_filling=args.filling)

    print("=== 18-site Ruby ED ===")
    print(f"sites=18, particles={solver.n_particles}, dimension={solver.dimension}")
    print(f"primitive filling={solver.primitive_filling:g}")
    print(f"ti={args.ti:g}, t1={args.t1:g}, t2={args.t2:g}")
    print("allowed primitive momenta: Gamma and +/-Q=(1/3,1/3); M is NOT commensurate")
    print(f"V points={len(Vvals)} from {Vvals[0]:g} to {Vvals[-1]:g}")
    if args.with_chi:
        print(
            "T=0 chi enabled: chi^reg excludes exact GS zero modes; "
            "nonzero C^GS is reported as a singular susceptibility."
        )

    nv = len(Vvals)
    nc = len(ED18_CHANNELS)
    ne = int(args.n_eigs)
    energies = np.full((nv, ne), np.nan, dtype=float)
    ground_deg = np.zeros(nv, dtype=int)
    gaps = np.full(nv, np.nan, dtype=float)
    dEdV = np.full(nv, np.nan, dtype=float)
    trans = np.full((nv, ne), np.nan + 1j * np.nan, dtype=complex)
    S = np.zeros((nv, 2, nc, nc), dtype=complex)
    Sevals = np.zeros((nv, 2, nc), dtype=float)
    Sevecs = np.zeros((nv, 2, nc, nc), dtype=complex)
    means = np.zeros((nv, 2, nc), dtype=complex)
    leading_q = np.empty(nv, dtype="U8")
    ground_sector = np.empty(nv, dtype="U32")

    if args.with_chi:
        chi_reg = np.zeros((nv, 2, nc, nc), dtype=complex)
        chi_revals = np.zeros((nv, 2, nc), dtype=float)
        chi_revecs = np.zeros((nv, 2, nc, nc), dtype=complex)
        chi_sing = np.zeros((nv, 2, nc, nc), dtype=complex)
        chi_sevals = np.zeros((nv, 2, nc), dtype=float)
        chi_sevecs = np.zeros((nv, 2, nc, nc), dtype=complex)
        chi_is_singular = np.zeros((nv, 2), dtype=bool)
        chi_residual_max = np.zeros((nv, 2), dtype=float)
        chi_solver_info_max = np.zeros((nv, 2), dtype=int)

    previous = None
    rows = []
    q_items = (("Gamma", Q_GAMMA), ("Q", Q_PERIOD3))

    for iv, V in enumerate(Vvals):
        spec = solver.solve(
            float(V),
            n_eigs=ne,
            tol=args.eig_tol,
            maxiter=args.maxiter,
            v0=previous,
            degeneracy_tol=args.deg_tol,
        )
        previous = spec.eigenvectors[:, 0].real.copy()

        energies[iv, : len(spec.energies)] = spec.energies
        ground_deg[iv] = spec.ground_multiplicity
        gaps[iv] = spec.gap_above_manifold
        dEdV[iv] = spec.interaction_expectation
        trans[iv, : len(spec.translation_eigenvalues)] = spec.translation_eigenvalues

        sfG = solver.structure_factor(spec, Q_GAMMA)
        sfQ = solver.structure_factor(spec, Q_PERIOD3)
        for iq, sf in enumerate((sfG, sfQ)):
            S[iv, iq] = sf.matrix
            Sevals[iv, iq] = sf.eigenvalues
            Sevecs[iv, iq] = sf.eigenvectors
            means[iv, iq] = sf.mean

        qlead = "Q" if sfQ.eigenvalues[0] > sfG.eigenvalues[0] else "Gamma"
        leading_q[iv] = qlead
        sector = ground_sector_label(spec.translation_eigenvalues)
        sector_ascii = {
            r"$\Gamma$": "Gamma",
            r"$\pm Q$": "+/-Q",
            r"$+Q$": "+Q",
            r"$-Q$": "-Q",
        }.get(sector, sector.replace("$", ""))
        ground_sector[iv] = sector_ascii
        phases = [translation_phase_reduced(z) for z in spec.translation_eigenvalues]
        ptext = ",".join(f"{x:+.3f}" for x in phases)

        print(
            f"V={V:7.4f}  E0={spec.energies[0]:+.10f}  "
            f"deg={spec.ground_multiplicity}  gap={spec.gap_above_manifold:.5e}  "
            f"<D>={spec.interaction_expectation:.6f}  Tphase=[{ptext}]"
        )
        print(
            f"          Smax(G)={sfG.eigenvalues[0]:.6f}  "
            f"Smax(Q)={sfQ.eigenvalues[0]:.6f}  leading={qlead}"
        )
        if args.print_modes > 0:
            for name, sf in (("G", sfG), ("Q", sfQ)):
                nm = min(int(args.print_modes), nc)
                for m in range(nm):
                    print(
                        f"          {name} mode {m+1}: S={sf.eigenvalues[m]:.6f}  "
                        + _mode_text(sf.channels, sf.eigenvectors[:, m])
                    )

        chi_results = []
        if args.with_chi:
            for iq, (qname, qvec) in enumerate(q_items):
                cres = zero_temperature_susceptibility(
                    solver,
                    spec,
                    qvec,
                    solve_tol=args.chi_tol,
                    maxiter=args.chi_maxiter,
                    singular_tol=args.chi_singular_tol,
                )
                chi_results.append(cres)
                chi_reg[iv, iq] = cres.regular_matrix
                chi_revals[iv, iq] = cres.regular_eigenvalues
                chi_revecs[iv, iq] = cres.regular_eigenvectors
                chi_sing[iv, iq] = cres.ground_singular_matrix
                chi_sevals[iv, iq] = cres.ground_singular_eigenvalues
                chi_sevecs[iv, iq] = cres.ground_singular_eigenvectors
                chi_is_singular[iv, iq] = cres.is_singular
                chi_residual_max[iv, iq] = float(np.max(cres.solver_residuals))
                chi_solver_info_max[iv, iq] = int(np.max(np.abs(cres.solver_info)))

            print(
                "          T=0 chi: "
                f"G reg={chi_results[0].regular_eigenvalues[0]:.6g} "
                f"C_GS={chi_results[0].ground_singular_eigenvalues[0]:.3e} "
                f"{'SINGULAR' if chi_results[0].is_singular else 'finite'}; "
                f"Q reg={chi_results[1].regular_eigenvalues[0]:.6g} "
                f"C_GS={chi_results[1].ground_singular_eigenvalues[0]:.3e} "
                f"{'SINGULAR' if chi_results[1].is_singular else 'finite'}"
            )
            if max(chi_residual_max[iv]) > max(1e-8, 20.0 * args.chi_tol):
                print(
                    "          WARNING: large correction-vector residual: "
                    f"{max(chi_residual_max[iv]):.3e}"
                )

        row = {
            "V": float(V),
            "E0": float(spec.energies[0]),
            "ground_multiplicity": int(spec.ground_multiplicity),
            "ground_sector": ground_sector[iv],
            "gap_above_manifold": float(spec.gap_above_manifold),
            "dE_dV_interaction_count": float(spec.interaction_expectation),
            "translation_phases_reduced": ";".join(f"{x:.12g}" for x in phases),
            "Smax_Gamma": float(sfG.eigenvalues[0]),
            "Smax_Q": float(sfQ.eigenvalues[0]),
            "leading_q": qlead,
            "leading_mode_Gamma": _mode_text(sfG.channels, sfG.eigenvectors[:, 0]),
            "leading_mode_Q": _mode_text(sfQ.channels, sfQ.eigenvectors[:, 0]),
        }
        for qname, sf in (("Gamma", sfG), ("Q", sfQ)):
            for ic, ch in enumerate(sf.channels):
                row[f"Sdiag_{qname}_{ch}"] = float(sf.matrix[ic, ic].real)
                row[f"leading_weight_{qname}_{ch}"] = float(abs(sf.eigenvectors[ic, 0]) ** 2)
            for m in range(min(3, nc)):
                row[f"Seig{m+1}_{qname}"] = float(sf.eigenvalues[m])

        if args.with_chi:
            for iq, (qname, _) in enumerate(q_items):
                cres = chi_results[iq]
                row[f"chi_reg_max_{qname}"] = float(cres.regular_eigenvalues[0])
                row[f"chi_GS_singular_coeff_max_{qname}"] = float(cres.ground_singular_eigenvalues[0])
                row[f"chi_is_singular_{qname}"] = bool(cres.is_singular)
                row[f"chi_residual_max_{qname}"] = float(chi_residual_max[iv, iq])
                row[f"chi_leading_mode_{qname}"] = _mode_text(cres.channels, cres.regular_eigenvectors[:, 0])
                row[f"chi_singular_mode_{qname}"] = _mode_text(cres.channels, cres.ground_singular_eigenvectors[:, 0])
                for ic, ch in enumerate(cres.channels):
                    row[f"chi_reg_diag_{qname}_{ch}"] = float(cres.regular_matrix[ic, ic].real)
        rows.append(row)

    Sdiag = np.real(np.diagonal(S, axis1=-2, axis2=-1))
    leading_weights = np.abs(Sevecs[..., 0]) ** 2
    leading_weights /= np.maximum(np.sum(leading_weights, axis=-1, keepdims=True), 1e-300)
    leading_subspace_w, leading_subspace_dim = leading_subspace_weights(Sevals, Sevecs)

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        V=Vvals,
        filling=float(args.filling),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        n_sites=18,
        n_particles=solver.n_particles,
        dimension=solver.dimension,
        channels=np.asarray(ED18_CHANNELS),
        q_names=np.asarray(["Gamma", "Q"]),
        q_vectors=np.asarray([Q_GAMMA, Q_PERIOD3]),
        energies=energies,
        ground_multiplicity=ground_deg,
        ground_sector=ground_sector,
        gap_above_manifold=gaps,
        interaction_expectation=dEdV,
        translation_eigenvalues=trans,
        structure_matrix=S,
        structure_eigenvalues=Sevals,
        structure_eigenvectors=Sevecs,
        structure_means=means,
        structure_diagonal=Sdiag,
        leading_mode_weights=leading_weights,
        leading_subspace_weights=leading_subspace_w,
        leading_subspace_dimension=leading_subspace_dim,
        leading_q=leading_q,
        cluster_warning=np.asarray(
            "18-site index-3 torus contains Gamma and +/-Q only; M is not commensurate"
        ),
    )
    if args.with_chi:
        payload.update(
            chi_temperature=np.asarray(0.0),
            chi_definition=np.asarray(
                "fixed-N thermodynamic/Kubo T=0 susceptibility; chi_regular excludes exact GS zero modes; nonzero C_GS means singular response"
            ),
            chi_regular_matrix=chi_reg,
            chi_regular_eigenvalues=chi_revals,
            chi_regular_eigenvectors=chi_revecs,
            chi_ground_singular_matrix=chi_sing,
            chi_ground_singular_eigenvalues=chi_sevals,
            chi_ground_singular_eigenvectors=chi_sevecs,
            chi_is_singular=chi_is_singular,
            chi_solver_residual_max=chi_residual_max,
            chi_solver_info_max=chi_solver_info_max,
        )
    np.savez_compressed(out, **payload)
    print("saved:", out)

    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", csv_path)

    summary_path = Path(args.plot) if args.plot else out.with_suffix(".png")
    modes_path = Path(args.modes_plot) if args.modes_plot else out.with_name(out.stem + "_modes.png")
    summary_path, modes_path = save_ed18_plots(
        out,
        summary_path=summary_path,
        modes_path=modes_path,
        top_spectrum=args.top_spectrum,
        top_structure=args.top_structure,
        dpi=args.dpi,
    )
    print("saved:", summary_path)
    print("saved:", modes_path)

    if args.with_chi:
        chi_path = Path(args.chi_plot) if args.chi_plot else out.with_name(out.stem + "_chi.png")
        chi_path = save_ed18_chi_plot(
            out,
            chi_path=chi_path,
            top_modes=args.chi_top_modes,
            dpi=args.dpi,
        )
        print("saved:", chi_path)

    changes = np.flatnonzero(
        (ground_deg[1:] != ground_deg[:-1]) | (ground_sector[1:] != ground_sector[:-1])
    )
    if len(changes):
        print("\n=== finite-size ground-sector changes ===")
        for i in changes:
            print(
                f"between V={Vvals[i]:.8g} (deg={ground_deg[i]}, sector={ground_sector[i]}) and "
                f"V={Vvals[i+1]:.8g} (deg={ground_deg[i+1]}, sector={ground_sector[i+1]})"
            )
    print("\nReminder: this 18-site cluster cannot test the M-point period-two state.")


if __name__ == "__main__":
    main()
