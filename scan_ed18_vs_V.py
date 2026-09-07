#!/usr/bin/env python3
"""Scan the 18-site exact-diagonalization spectrum and correlations versus V.

At primitive filling n=2 the calculation uses Nf=6 spinless fermions in the
same three-cell periodic torus as ``rubycgw.supercell``.  The Hilbert-space
dimension is C(18,6)=18564.

This cluster contains only primitive Gamma and +/-Q=(1/3,1/3).  It cannot
represent the period-two M point, so the resulting phase diagram is an exact
finite-size Gamma/Q benchmark, not a proof that an M phase is absent.

Besides the compact summary figure, this driver now writes a second
channel-resolved figure showing the full six-channel equal-time pseudospin
structure matrix: diagonal S_{mu,mu}, the first few structure eigenvalues, and
the composition |v_mu|^2 of the leading collective mode.
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
from rubycgw.ed18_plot import (
    ground_sector_label,
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
    p.add_argument(
        "--plot",
        default=None,
        help="Summary PNG path. Default: same stem as --out.",
    )
    p.add_argument(
        "--modes-plot",
        default=None,
        help="Channel-resolved modes PNG path. Default: <out_stem>_modes.png.",
    )
    p.add_argument("--print-modes", type=int, default=3)
    p.add_argument(
        "--top-spectrum",
        type=int,
        default=6,
        help="Number of low-energy ED levels shown in the summary plot.",
    )
    p.add_argument(
        "--top-structure",
        type=int,
        default=3,
        help="Number of structure-factor eigenvalues shown at each q.",
    )
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _v_grid(args):
    if args.V_list:
        return np.asarray(sorted(set(float(x) for x in args.V_list)), dtype=float)
    if args.dV <= 0:
        raise ValueError("--dV must be positive")
    n = int(np.floor((args.Vmax - args.Vmin) / args.dV + 0.5))
    out = args.Vmin + args.dV * np.arange(n + 1, dtype=float)
    out = out[out <= args.Vmax + 1e-12]
    return out


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
    print(f"sites={18}, particles={solver.n_particles}, dimension={solver.dimension}")
    print(f"primitive filling={solver.primitive_filling:g}")
    print(f"ti={args.ti:g}, t1={args.t1:g}, t2={args.t2:g}")
    print("allowed primitive momenta: Gamma and +/-Q=(1/3,1/3); M is NOT commensurate")
    print(f"V points={len(Vvals)} from {Vvals[0]:g} to {Vvals[-1]:g}")

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

    previous = None
    rows = []
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
                row[f"leading_weight_{qname}_{ch}"] = float(
                    abs(sf.eigenvectors[ic, 0]) ** 2
                )
            for m in range(min(3, nc)):
                row[f"Seig{m+1}_{qname}"] = float(sf.eigenvalues[m])
        rows.append(row)

    # Convenient derived arrays for downstream plotting/analysis.
    Sdiag = np.real(np.diagonal(S, axis1=-2, axis2=-1))
    leading_weights = np.abs(Sevecs[..., 0]) ** 2
    leading_weights /= np.maximum(
        np.sum(leading_weights, axis=-1, keepdims=True), 1e-300
    )

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
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
        leading_q=leading_q,
        cluster_warning=np.asarray(
            "18-site index-3 torus contains Gamma and +/-Q only; M is not commensurate"
        ),
    )
    print("saved:", out)

    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", csv_path)

    summary_path = Path(args.plot) if args.plot else out.with_suffix(".png")
    modes_path = (
        Path(args.modes_plot)
        if args.modes_plot
        else out.with_name(out.stem + "_modes.png")
    )
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

    # Report intervals where the exact finite-size ground-state multiplet changes.
    changes = np.flatnonzero(
        (ground_deg[1:] != ground_deg[:-1])
        | (ground_sector[1:] != ground_sector[:-1])
    )
    if len(changes):
        print("\n=== finite-size ground-sector changes ===")
        for i in changes:
            print(
                f"between V={Vvals[i]:.8g} "
                f"(deg={ground_deg[i]}, sector={ground_sector[i]}) and "
                f"V={Vvals[i+1]:.8g} "
                f"(deg={ground_deg[i+1]}, sector={ground_sector[i+1]})"
            )
    print("\nReminder: this 18-site cluster cannot test the M-point period-two state.")


if __name__ == "__main__":
    main()
