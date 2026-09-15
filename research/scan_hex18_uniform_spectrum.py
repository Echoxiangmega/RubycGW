#!/usr/bin/env python3
"""Scan low-energy states and uniform loop-current spectral weight on hex18.

For each V this script independently diagonalizes the six-triangle 18-site OBC
cluster, then resolves the uniform physical loop-current operator

    J_u = (J_0 + ... + J_5) / sqrt(6)

against the low-energy eigenstates.  The most useful diagnostics are

    delta_n       = E_n - E_0,
    weight_n      = |<n|J_u|0>|^2,
    delta_current = delta_n of the excited state carrying the largest weight,
    chi_low       = 2 sum_n weight_n / delta_n,
    S_low         = sum_n weight_n,
    delta_eff_low = 2 S_low / chi_low.

``chi_low`` and ``S_low`` contain only the requested ``n_eigs`` low-energy
states, so increase ``--n-eigs`` to check spectral-weight convergence.  In
contrast, ``delta_current`` is already very informative if the dominant weight
moves onto a state whose gap collapses with increasing V.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.hex18_ed import Hex18Solver, canonical_ring_mode
from rubycgw.model import RubyParameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--n-eigs", type=int, default=24,
                   help="number of lowest many-body eigenstates kept at each V")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=10000)
    p.add_argument("--degeneracy-tol", type=float, default=1e-8)
    p.add_argument("--out", type=Path,
                   default=Path("results/hex18_uniform_spectrum.npz"))
    return p.parse_args()


def main():
    args = parse_args()
    solver = Hex18Solver(
        RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0),
        primitive_filling=args.filling,
    )

    u = canonical_ring_mode(0).real
    Ju = sum(float(u[m]) * solver.current_ops[m] for m in range(6))

    Vs = np.asarray(args.V, dtype=float)
    nV = len(Vs)
    ne = int(args.n_eigs)

    energies = np.full((nV, ne), np.nan)
    gaps = np.full((nV, ne), np.nan)
    weights = np.full((nV, ne), np.nan)
    gs_mult = np.zeros(nV, dtype=int)
    first_gap = np.full(nV, np.nan)
    current_state = np.full(nV, -1, dtype=int)
    current_gap = np.full(nV, np.nan)
    current_weight = np.full(nV, np.nan)
    chi_low = np.full(nV, np.nan)
    S_low = np.full(nV, np.nan)
    delta_eff_low = np.full(nV, np.nan)
    ground_J_eigs = []

    print("=== hex18 uniform-current spectral scan ===")
    print(f"sites=18 particles={solver.n_particles} dim={solver.dimension}")
    print(f"ti={args.ti} t1={args.t1} t2={args.t2} filling={args.filling}")
    print(f"n_eigs={ne}")

    for iv, V in enumerate(Vs):
        # Important: do NOT warm-start from the previous V.  A symmetry-pure v0
        # can lock eigsh into one C6 sector and hide the actual first excited state.
        spec = solver.solve(
            float(V), n_eigs=ne, tol=args.tol, maxiter=args.maxiter,
            v0=None, degeneracy_tol=args.degeneracy_tol,
        )
        e = np.asarray(spec.energies, dtype=float)
        vec = np.asarray(spec.eigenvectors, dtype=complex)
        nfound = len(e)
        energies[iv, :nfound] = e
        gaps[iv, :nfound] = e - e[0]
        gs_mult[iv] = int(spec.ground_multiplicity)
        first_gap[iv] = float(spec.gap_above_manifold)

        ng = int(spec.ground_multiplicity)
        G = vec[:, :ng]

        # Diagnose whether an exactly/nearly degenerate ground manifold itself
        # can be rotated into current-carrying states.
        Jgg = G.conj().T @ (Ju @ G)
        jge = np.linalg.eigvalsh(0.5 * (Jgg + Jgg.conj().T)).real
        ground_J_eigs.append(jge)

        # Ground-manifold-averaged spectral weight.  For ng=1 this is exactly
        # |<n|Ju|0>|^2.  States inside the ground manifold are excluded below.
        for n in range(nfound):
            psi_n = vec[:, n]
            amp2 = 0.0
            for g in range(ng):
                amp = np.vdot(psi_n, Ju @ G[:, g])
                amp2 += float(abs(amp) ** 2) / float(ng)
            weights[iv, n] = amp2

        exc = np.arange(ng, nfound, dtype=int)
        if exc.size:
            dex = gaps[iv, exc]
            wex = weights[iv, exc]
            good = dex > max(args.degeneracy_tol, 1e-14)
            exc = exc[good]
            dex = dex[good]
            wex = wex[good]

            if exc.size:
                imax = int(np.argmax(wex))
                nstar = int(exc[imax])
                current_state[iv] = nstar
                current_gap[iv] = float(gaps[iv, nstar])
                current_weight[iv] = float(weights[iv, nstar])
                S_low[iv] = float(np.sum(wex))
                chi_low[iv] = float(2.0 * np.sum(wex / dex))
                if chi_low[iv] > 0:
                    delta_eff_low[iv] = float(2.0 * S_low[iv] / chi_low[iv])

        print(f"\nV={V:g}")
        print(f"  E0={e[0]:+.12f}  gs_mult={ng}  first_gap={first_gap[iv]:.8e}")
        print(f"  ground-manifold eig(P J_u P) = {np.array2string(jge, precision=7)}")
        print(f"  dominant-current state n*={current_state[iv]}  "
              f"gap={current_gap[iv]:.8e}  weight={current_weight[iv]:.8e}")
        print(f"  low-spectrum S={S_low[iv]:.8e}  chi={chi_low[iv]:.8e}  "
              f"delta_eff={delta_eff_low[iv]:.8e}")
        print("  lowest states:   n        delta_n          weight_n")
        for n in range(min(nfound, 12)):
            print(f"                 {n:2d}   {gaps[iv,n]: .8e}   {weights[iv,n]: .8e}")

    # Ragged ground-manifold spectra -> padded rectangular array.
    max_ng = max(len(x) for x in ground_J_eigs)
    gj = np.full((nV, max_ng), np.nan)
    for i, x in enumerate(ground_J_eigs):
        gj[i, :len(x)] = x

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        V=Vs,
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        filling=float(args.filling), n_particles=int(solver.n_particles),
        dimension=int(solver.dimension), n_eigs=int(ne),
        energies=energies, gaps=gaps, weights=weights,
        ground_multiplicity=gs_mult,
        first_gap=first_gap,
        current_state=current_state,
        current_gap=current_gap,
        current_weight=current_weight,
        S_low=S_low,
        chi_low=chi_low,
        delta_eff_low=delta_eff_low,
        ground_J_eigenvalues=gj,
    )
    print(f"\nsaved: {args.out}")
    print("Plot with:")
    print(f"  python plot_hex18_uniform_spectrum.py {args.out}")


if __name__ == "__main__":
    main()
