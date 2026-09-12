#!/usr/bin/env python3
"""Scan loop-current correlations on the six-triangle 18-site Ruby hexagon."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.hex18_ed import Hex18Solver, canonical_ring_mode
from rubycgw.model import RubyParameters


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--n-eigs", type=int, default=8)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--source-h", type=float, default=1e-3)
    p.add_argument("--source", choices=("none", "uniform", "alternating", "both"), default="both")
    p.add_argument("--out", type=Path, default=Path("results/hex18_loop_current.npz"))
    return p.parse_args()


def main():
    args = _args()
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    solver = Hex18Solver(params, primitive_filling=args.filling)
    print("=== six-triangle Ruby hexagon ED ===")
    print(f"sites=18, particles={solver.n_particles}, dim={solver.dimension}, ti={args.ti}, t1={args.t1}, t2={args.t2}")
    print(f"interaction bonds={len(solver.interaction_pairs)}")

    Vs = np.asarray(args.V, dtype=float)
    energies = []
    gaps = []
    structure_eigs = []
    fourier_structure = []
    principal_modes = []
    circulant_error = []
    current_means = []
    chi_uniform = []
    chi_alternating = []
    prev = None

    u0 = canonical_ring_mode(0).real
    u3 = canonical_ring_mode(3).real
    for V in Vs:
        spec = solver.solve(V, n_eigs=args.n_eigs, tol=args.tol, maxiter=args.maxiter, v0=prev)
        prev = spec.eigenvectors[:, 0]
        cur = solver.current_structure(spec)
        Sell = np.asarray([
            np.vdot(canonical_ring_mode(ell), cur.matrix @ canonical_ring_mode(ell)).real
            for ell in range(6)
        ])
        principal = np.asarray(cur.eigenvectors[:, 0], dtype=complex)
        imax = int(np.argmax(np.abs(principal)))
        if abs(principal[imax]) > 1e-14:
            principal *= np.exp(-1j * np.angle(principal[imax]))
        print(f"\nV={V:g}: E0={spec.energies[0]:+.10f}, gap={spec.gap_above_manifold:.6e}, gs_mult={spec.ground_multiplicity}")
        print("  S_ell[0..5] = " + np.array2string(Sell, precision=7, suppress_small=True))
        print(f"  Smax={cur.eigenvalues[0]:.8f}, circulant_err={cur.circulant_error:.3e}")
        print("  principal mode = " + np.array2string(principal, precision=5, suppress_small=True))
        print("  <J_m> = " + np.array2string(cur.means.real, precision=4, suppress_small=True))

        cu = np.nan
        ca = np.nan
        if args.source in {"uniform", "both"}:
            ru = solver.source_response(V, u0, h=args.source_h, tol=args.tol, maxiter=args.maxiter, v0=prev)
            cu = float(ru["chi"])
            print(f"  chi_uniform(h={args.source_h:g})={cu:.8f}, M+={ru['M_plus']:+.6e}")
        if args.source in {"alternating", "both"}:
            ra = solver.source_response(V, u3, h=args.source_h, tol=args.tol, maxiter=args.maxiter, v0=prev)
            ca = float(ra["chi"])
            print(f"  chi_alternating(h={args.source_h:g})={ca:.8f}, M+={ra['M_plus']:+.6e}")

        energies.append(spec.energies)
        gaps.append(spec.gap_above_manifold)
        structure_eigs.append(cur.eigenvalues)
        fourier_structure.append(Sell)
        principal_modes.append(principal)
        circulant_error.append(cur.circulant_error)
        current_means.append(cur.means)
        chi_uniform.append(cu)
        chi_alternating.append(ca)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        V=Vs,
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2), filling=float(args.filling),
        n_particles=int(solver.n_particles), dimension=int(solver.dimension),
        energies=np.asarray(energies), gaps=np.asarray(gaps),
        structure_eigenvalues=np.asarray(structure_eigs),
        fourier_structure=np.asarray(fourier_structure),
        principal_modes=np.asarray(principal_modes),
        circulant_error=np.asarray(circulant_error),
        current_means=np.asarray(current_means),
        source_h=float(args.source_h),
        chi_uniform=np.asarray(chi_uniform), chi_alternating=np.asarray(chi_alternating),
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
