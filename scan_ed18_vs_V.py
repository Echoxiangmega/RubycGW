#!/usr/bin/env python3
"""Scan the 18-site exact-diagonalization spectrum and correlations versus V.

At primitive filling n=2 the calculation uses Nf=6 spinless fermions in the
same three-cell periodic torus as ``rubycgw.supercell``.  The Hilbert-space
dimension is C(18,6)=18564.

This cluster contains only primitive Gamma and +/-Q=(1/3,1/3).  It cannot
represent the period-two M point, so the resulting phase diagram is an exact
finite-size Gamma/Q benchmark, not a proof that an M phase is absent.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.ed18 import (
    ED18_CHANNELS,
    ED18Solver,
    Q_GAMMA,
    Q_PERIOD3,
    phase_fix_vector,
    translation_phase_reduced,
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
    p.add_argument("--plot", default=None)
    p.add_argument("--print-modes", type=int, default=3)
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

        rows.append(
            {
                "V": float(V),
                "E0": float(spec.energies[0]),
                "ground_multiplicity": int(spec.ground_multiplicity),
                "gap_above_manifold": float(spec.gap_above_manifold),
                "dE_dV_interaction_count": float(spec.interaction_expectation),
                "translation_phases_reduced": ";".join(f"{x:.12g}" for x in phases),
                "Smax_Gamma": float(sfG.eigenvalues[0]),
                "Smax_Q": float(sfQ.eigenvalues[0]),
                "leading_q": qlead,
                "leading_mode_Gamma": _mode_text(sfG.channels, sfG.eigenvectors[:, 0]),
                "leading_mode_Q": _mode_text(sfQ.channels, sfQ.eigenvectors[:, 0]),
            }
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
        gap_above_manifold=gaps,
        interaction_expectation=dEdV,
        translation_eigenvalues=trans,
        structure_matrix=S,
        structure_eigenvalues=Sevals,
        structure_eigenvectors=Sevecs,
        structure_means=means,
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

    plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.8), sharex=True)
    axes[0].plot(Vvals, energies[:, 0], marker="o", ms=3)
    axes[0].set_ylabel(r"$E_0$")
    axes[0].set_title("18-site Ruby ED, n=2 (Gamma/Q benchmark; M not commensurate)")

    axes[1].plot(Vvals, gaps, marker="o", ms=3, label="gap above GS manifold")
    axes[1].plot(Vvals, dEdV, marker="s", ms=3, label=r"$\langle H_V/V\rangle$")
    axes[1].set_ylabel("gap / interaction count")
    axes[1].legend()

    axes[2].plot(Vvals, Sevals[:, 0, 0], marker="o", ms=3, label=r"$S_{max}(\Gamma)$")
    axes[2].plot(Vvals, Sevals[:, 1, 0], marker="s", ms=3, label=r"$S_{max}(Q)$")
    axes[2].set_xlabel(r"$V$")
    axes[2].set_ylabel("leading structure eigenvalue")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print("saved:", plot_path)

    # Report intervals where the exact finite-size ground-state multiplet changes.
    changes = np.flatnonzero(ground_deg[1:] != ground_deg[:-1])
    if len(changes):
        print("\n=== finite-size ground-sector changes ===")
        for i in changes:
            print(
                f"between V={Vvals[i]:.8g} (deg={ground_deg[i]}) and "
                f"V={Vvals[i+1]:.8g} (deg={ground_deg[i+1]})"
            )
    print("\nReminder: this 18-site cluster cannot test the M-point period-two state.")


if __name__ == "__main__":
    main()
