#!/usr/bin/env python3
"""Diagnose the low-energy manifold of the six-triangle Hex18 ED cluster.

For each requested V this script

1. solves the lowest many-body eigenstates independently (no V-to-V eigsh v0),
2. projects the uniform physical loop-current operator J_u into low-energy
   subspaces and diagonalizes P J_u P,
3. reports the energy mean/variance and local triangle currents of the most
   positive/negative projected-chirality states,
4. fingerprints each low-lying excitation with spectral weights in ring
   harmonics ell=0..5 of four local operator families:
       x, y : physical triangle pseudospin (TR even orbital/charge polarization)
       z    : physical loop-current pseudospin (TR odd)
       rho  : triangle total density.

The raw mode weights are
    w[n,alpha,ell] = |<n| O_{alpha,ell} |0>|^2,
where O_{alpha,ell} = sum_m exp(i 2 pi ell m/6) O_{alpha,m}/sqrt(6).
For ell != 0,3 the harmonic operator is complex/non-Hermitian; this is fine for
spectral classification.  Conjugate ell and 6-ell sectors should pair by C6/TR.

Because x/y span one E doublet and ell=1/5,2/4 are conjugate ring harmonics,
the script additionally reports basis-insensitive grouped fingerprints such as
"E_|ell|=2", obtained by summing x/y and the conjugate ell pair.  These grouped
labels are the preferred way to identify a physical mode across parameter scans.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.ed18 import _manybody_onebody
from rubycgw.hex18_ed import HEX_TRIANGLES, NTRI, Hex18Solver, canonical_ring_mode
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import primitive_triangle_pseudospin_vertices


COMPONENTS = np.asarray(["x", "y", "z", "rho"])

# Group raw (component index, ell) channels into labels that do not depend on
# choosing x versus y inside the E doublet or +ell versus -ell for conjugate
# ring harmonics.  For ell=0 and 3 the harmonic is self-conjugate.
GROUP_DEFINITIONS = (
    ("E_ell=0", ((0, 0), (1, 0))),
    ("E_|ell|=1", ((0, 1), (0, 5), (1, 1), (1, 5))),
    ("E_|ell|=2", ((0, 2), (0, 4), (1, 2), (1, 4))),
    ("E_ell=3", ((0, 3), (1, 3))),
    ("z_ell=0", ((2, 0),)),
    ("z_|ell|=1", ((2, 1), (2, 5))),
    ("z_|ell|=2", ((2, 2), (2, 4))),
    ("z_ell=3", ((2, 3),)),
    ("rho_ell=0", ((3, 0),)),
    ("rho_|ell|=1", ((3, 1), (3, 5))),
    ("rho_|ell|=2", ((3, 2), (3, 4))),
    ("rho_ell=3", ((3, 3),)),
)
GROUP_LABELS = np.asarray([x[0] for x in GROUP_DEFINITIONS])


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--n-eigs", type=int, default=16)
    p.add_argument("--subspace-sizes", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--print-states", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/hex18_low_energy_modes.npz"))
    return p.parse_args()


def _triangle_onebody_family(component: str) -> np.ndarray:
    """Return six 18x18 local one-body matrices for x/y/z/rho."""
    comp = component.lower()
    pv = primitive_triangle_pseudospin_vertices()
    mats = np.zeros((NTRI, 18, 18), dtype=complex)
    for m, (kind, _) in enumerate(HEX_TRIANGLES):
        sl = slice(3 * m, 3 * m + 3)
        if comp == "rho":
            block = np.eye(3, dtype=complex)
        else:
            key = kind + comp
            src = np.asarray(pv[key], dtype=complex)
            subs = (0, 1, 2) if kind == "A" else (3, 4, 5)
            block = src[np.ix_(subs, subs)]
        mats[m, sl, sl] = block
    return mats


def _manybody_local_families(solver: Hex18Solver):
    out = {}
    for comp in ("x", "y", "z", "rho"):
        one = _triangle_onebody_family(comp)
        out[comp] = tuple(
            _manybody_onebody(one[m], solver.basis, solver.index)
            for m in range(NTRI)
        )
    return out


def _harmonic_op(local_ops, ell: int):
    coeff = canonical_ring_mode(ell)
    op = local_ops[0] * coeff[0]
    for m in range(1, NTRI):
        op = op + coeff[m] * local_ops[m]
    return op


def _fingerprint(vecs: np.ndarray, families):
    """Return amplitudes/weights with shape (nstate,4,6), from ground state."""
    comps = ("x", "y", "z", "rho")
    psi0 = vecs[:, 0]
    nstate = vecs.shape[1]
    weights = np.zeros((nstate, len(comps), NTRI), dtype=float)
    amplitudes = np.zeros((nstate, len(comps), NTRI), dtype=complex)
    for ia, comp in enumerate(comps):
        for ell in range(NTRI):
            O = _harmonic_op(families[comp], ell)
            phi = O @ psi0
            amp = vecs.conj().T @ phi
            amplitudes[:, ia, ell] = amp
            weights[:, ia, ell] = np.abs(amp) ** 2
    return amplitudes, weights


def _grouped_fingerprint(weights: np.ndarray) -> np.ndarray:
    """Sum raw weights into basis-insensitive grouped channels.

    Parameters
    ----------
    weights : ndarray, shape (..., 4, 6)
        Raw spectral weights.

    Returns
    -------
    grouped : ndarray, shape (..., n_groups)
    """
    weights = np.asarray(weights, dtype=float)
    out = np.zeros(weights.shape[:-2] + (len(GROUP_DEFINITIONS),), dtype=float)
    for ig, (_, members) in enumerate(GROUP_DEFINITIONS):
        for ia, ell in members:
            out[..., ig] += weights[..., ia, ell]
    return out


def _projected_chiral_states(energies, vecs, current_ops, M: int):
    M = int(min(max(2, M), len(energies)))
    U = vecs[:, :M]
    u0 = canonical_ring_mode(0).real
    Ju = current_ops[0] * float(u0[0])
    for m in range(1, NTRI):
        Ju = Ju + float(u0[m]) * current_ops[m]
    Jlow = U.conj().T @ (Ju @ U)
    Jlow = 0.5 * (Jlow + Jlow.conj().T)
    jvals, coeff = np.linalg.eigh(Jlow)
    order = np.argsort(jvals.real)
    jvals = np.asarray(jvals[order].real, dtype=float)
    coeff = np.asarray(coeff[:, order], dtype=complex)

    E = np.asarray(energies[:M], dtype=float)
    rows = []
    for idx in (0, -1):
        c = coeff[:, idx]
        prob = np.abs(c) ** 2
        Emean = float(np.dot(prob, E))
        E2 = float(np.dot(prob, E * E))
        Evar = max(0.0, E2 - Emean * Emean)
        psi = U @ c
        localJ = np.asarray([np.vdot(psi, op @ psi).real for op in current_ops], dtype=float)
        rows.append((float(jvals[idx]), Emean, float(np.sqrt(Evar)), localJ, prob))
    return jvals, Jlow, rows


def main():
    args = _args()
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    solver = Hex18Solver(params, primitive_filling=args.filling)
    families = _manybody_local_families(solver)

    Vs = np.asarray(args.V, dtype=float)
    ne = int(args.n_eigs)
    subspaces = np.asarray(sorted(set(int(x) for x in args.subspace_sizes if int(x) >= 2)), dtype=int)
    if len(subspaces) == 0:
        raise ValueError("at least one subspace size >=2 is required")

    all_E = []
    all_weights = []
    all_grouped_weights = []
    all_amp = []
    first_best_comp = []
    first_best_ell = []
    first_best_weight = []
    first_best_group = []
    first_best_group_weight = []
    proj_jmin = []
    proj_jmax = []
    proj_Emin = []
    proj_Emax = []
    proj_sigEmin = []
    proj_sigEmax = []
    proj_localJ_min = []
    proj_localJ_max = []
    proj_prob_min = []
    proj_prob_max = []

    print("=== Hex18 low-energy mode diagnostic ===")
    print(f"sites=18 N={solver.n_particles} dim={solver.dimension} n_eigs={ne}")
    print("operator families: x,y,z(loop current),rho(triangle density)")
    print("preferred mode labels: grouped x/y E-doublet and conjugate +/-ell sectors")

    for V in Vs:
        # Intentionally no v0 from neighboring V: preserve access to all C6 sectors.
        spec = solver.solve(float(V), n_eigs=ne, tol=args.tol, maxiter=args.maxiter, v0=None)
        E = np.asarray(spec.energies, dtype=float)
        vecs = np.asarray(spec.eigenvectors, dtype=complex)
        amp, weights = _fingerprint(vecs, families)
        grouped = _grouped_fingerprint(weights)

        # Raw classification is useful diagnostically, but the grouped label is
        # the preferred invariant descriptor across V.
        flat = weights[1].reshape(-1)
        ib = int(np.argmax(flat))
        ia, ell = np.unravel_index(ib, weights[1].shape)
        bcomp = str(COMPONENTS[ia])
        bw = float(weights[1, ia, ell])
        igb = int(np.argmax(grouped[1]))
        bgroup = str(GROUP_LABELS[igb])
        bgw = float(grouped[1, igb])

        print(f"\nV={V:g}  E0={E[0]:+.10f}  gap1={E[1]-E[0]:.8e}")
        print(f"  n=1 grouped fingerprint: {bgroup}, weight={bgw:.8e}")
        print(f"      raw largest component: {bcomp}, ell={ell}, weight={bw:.8e}")
        nprint = min(int(args.print_states), len(E) - 1)
        for n in range(1, nprint + 1):
            flatn = weights[n].reshape(-1)
            ibn = int(np.argmax(flatn))
            ian, elln = np.unravel_index(ibn, weights[n].shape)
            ign = int(np.argmax(grouped[n]))
            print(
                f"  n={n:2d} gap={E[n]-E[0]:.8e}  "
                f"group={GROUP_LABELS[ign]} W={grouped[n,ign]:.6e}  "
                f"raw={COMPONENTS[ian]} ell={elln} w={weights[n,ian,elln]:.6e}  "
                f"w_z0={weights[n,2,0]:.6e}"
            )

        jmns=[]; jmxs=[]; emns=[]; emxs=[]; smns=[]; smxs=[]; ljmns=[]; ljmxs=[]; pmns=[]; pmxs=[]
        for M in subspaces:
            jvals, _, rows = _projected_chiral_states(E, vecs, solver.current_ops, int(M))
            rmin, rmax = rows
            jmns.append(rmin[0]); jmxs.append(rmax[0])
            emns.append(rmin[1]); emxs.append(rmax[1])
            smns.append(rmin[2]); smxs.append(rmax[2])
            ljmns.append(rmin[3]); ljmxs.append(rmax[3])
            # pad probabilities to max subspace size for a rectangular saved array
            pmin = np.zeros(int(subspaces[-1]), dtype=float)
            pmax = np.zeros(int(subspaces[-1]), dtype=float)
            pmin[:len(rmin[4])] = rmin[4]
            pmax[:len(rmax[4])] = rmax[4]
            pmns.append(pmin); pmxs.append(pmax)
            print(
                f"  M={M:2d}: Jmin={rmin[0]:+.8f}, Jmax={rmax[0]:+.8f}, "
                f"<H>-E0=({rmin[1]-E[0]:.6e},{rmax[1]-E[0]:.6e}), "
                f"sigmaE=({rmin[2]:.3e},{rmax[2]:.3e})"
            )

        all_E.append(E)
        all_weights.append(weights)
        all_grouped_weights.append(grouped)
        all_amp.append(amp)
        first_best_comp.append(bcomp)
        first_best_ell.append(ell)
        first_best_weight.append(bw)
        first_best_group.append(bgroup)
        first_best_group_weight.append(bgw)
        proj_jmin.append(jmns); proj_jmax.append(jmxs)
        proj_Emin.append(emns); proj_Emax.append(emxs)
        proj_sigEmin.append(smns); proj_sigEmax.append(smxs)
        proj_localJ_min.append(ljmns); proj_localJ_max.append(ljmxs)
        proj_prob_min.append(pmns); proj_prob_max.append(pmxs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        V=Vs,
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2), filling=float(args.filling),
        energies=np.asarray(all_E),
        gaps=np.asarray(all_E) - np.asarray(all_E)[:, :1],
        components=COMPONENTS,
        spectral_amplitudes=np.asarray(all_amp),
        spectral_weights=np.asarray(all_weights),
        group_labels=GROUP_LABELS,
        grouped_spectral_weights=np.asarray(all_grouped_weights),
        first_best_component=np.asarray(first_best_comp),
        first_best_ell=np.asarray(first_best_ell, dtype=int),
        first_best_weight=np.asarray(first_best_weight, dtype=float),
        first_best_group=np.asarray(first_best_group),
        first_best_group_weight=np.asarray(first_best_group_weight, dtype=float),
        subspace_sizes=subspaces,
        projected_jmin=np.asarray(proj_jmin), projected_jmax=np.asarray(proj_jmax),
        projected_Emin=np.asarray(proj_Emin), projected_Emax=np.asarray(proj_Emax),
        projected_sigmaE_min=np.asarray(proj_sigEmin), projected_sigmaE_max=np.asarray(proj_sigEmax),
        projected_localJ_min=np.asarray(proj_localJ_min), projected_localJ_max=np.asarray(proj_localJ_max),
        projected_prob_min=np.asarray(proj_prob_min), projected_prob_max=np.asarray(proj_prob_max),
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
