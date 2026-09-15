#!/usr/bin/env python3
"""Low-energy mode diagnostic for the existing 18-site index-three Ruby PBC torus.

This is the periodic counterpart of ``analyze_hex18_low_energy_modes.py``.
The natural labels are the three allowed primitive-cell momenta

    Gamma, +Q, -Q,  Q=(1/3,1/3),

and A/B-triangle even/odd combinations in the *physical* pseudospin frame.
For each V the script

1. solves the lowest many-body eigenstates independently (no V-to-V v0),
2. fingerprints low excitations with x/y/z/rho operators resolved by
   A/B parity and momentum,
3. identifies the dominant uniform physical loop-current excitation
   z-even at Gamma,
4. diagonalizes the uniform-current operator in low-energy subspaces and
   reports the most positive/negative chiral combinations.

The uniform current is exactly

    J_u = z_same(q=0) = (1/sqrt(6)) sum_{six triangles} J_m.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.ed18 import ED18Solver, _manybody_onebody
from rubycgw.model import RubyParameters, NSUB
from rubycgw.pseudospin import primitive_triangle_pseudospin_vertices
from rubycgw.supercell import NSECTOR, NSUP, SUPERCELL_REPRESENTATIVES


Q_G = np.array([0.0, 0.0], dtype=float)
Q_P = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
Q_M = -Q_P
Q_LIST = (Q_G, Q_P, Q_M)
Q_NAMES = ("G", "+Q", "-Q")
COMP_NAMES = ("x", "y", "z", "rho")
PARITY_NAMES = ("even", "odd")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", nargs="+", type=float,
                   default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--n-eigs", type=int, default=16)
    p.add_argument("--subspace-sizes", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--print-states", type=int, default=8)
    p.add_argument("--degeneracy-tol", type=float, default=1e-7)
    p.add_argument("--out", type=Path, default=Path("results/ed18_pbc_low_energy_modes.npz"))
    return p.parse_args()


def _primitive_ab_matrices(component: str):
    """Return physical A/B 6x6 matrices for x,y,z or triangle density rho."""
    comp = str(component).lower()
    if comp == "rho":
        A = np.zeros((NSUB, NSUB), dtype=complex)
        B = np.zeros_like(A)
        A[0, 0] = A[1, 1] = A[2, 2] = 1.0
        B[3, 3] = B[4, 4] = B[5, 5] = 1.0
        return A, B
    pv = primitive_triangle_pseudospin_vertices()
    return np.asarray(pv["A" + comp], dtype=complex), np.asarray(pv["B" + comp], dtype=complex)


def _onebody_harmonic(component: str, parity: str, q: np.ndarray):
    A, B = _primitive_ab_matrices(component)
    sign = +1.0 if parity == "even" else -1.0
    k6 = (A + sign * B) / np.sqrt(2.0)
    one = np.zeros((NSUP, NSUP), dtype=complex)
    q = np.asarray(q, dtype=float).reshape(2)
    for s, R in enumerate(SUPERCELL_REPRESENTATIVES):
        phase = np.exp(-2j * np.pi * np.dot(q, R)) / np.sqrt(float(NSECTOR))
        sl = slice(NSUB * s, NSUB * (s + 1))
        one[sl, sl] = phase * k6
    return one


def _operator_dictionary(solver: ED18Solver):
    """Return ops[component][parity][iq] and the six local physical J_m operators."""
    ops = {}
    for comp in COMP_NAMES:
        ops[comp] = {}
        for parity in PARITY_NAMES:
            ops[comp][parity] = tuple(
                _manybody_onebody(_onebody_harmonic(comp, parity, q), solver.basis, solver.index)
                for q in Q_LIST
            )

    pv = primitive_triangle_pseudospin_vertices()
    local_j = []
    for s in range(NSECTOR):
        for kind in ("A", "B"):
            one = np.zeros((NSUP, NSUP), dtype=complex)
            sl = slice(NSUB * s, NSUB * (s + 1))
            one[sl, sl] = np.asarray(pv[kind + "z"], dtype=complex)
            local_j.append(_manybody_onebody(one, solver.basis, solver.index))
    return ops, tuple(local_j)


def _fingerprint(vecs: np.ndarray, ops):
    """weights[n,comp,parity,q] = |<n|O|0>|^2."""
    psi0 = vecs[:, 0]
    ns = vecs.shape[1]
    amp = np.zeros((ns, len(COMP_NAMES), len(PARITY_NAMES), len(Q_LIST)), dtype=complex)
    w = np.zeros_like(amp.real)
    for ic, comp in enumerate(COMP_NAMES):
        for ip, parity in enumerate(PARITY_NAMES):
            for iq in range(len(Q_LIST)):
                phi = ops[comp][parity][iq] @ psi0
                a = vecs.conj().T @ phi
                amp[:, ic, ip, iq] = a
                w[:, ic, ip, iq] = np.abs(a) ** 2
    return amp, w


def _grouped_weights(weights: np.ndarray):
    """Basis-robust groups: E=(x+y), z, rho; momentum G or +/-Q pair; A/B parity."""
    # output group labels and weights[n,group]
    labels = []
    cols = []
    for family in ("E", "z", "rho"):
        for qg in ("G", "Q"):
            for parity in PARITY_NAMES:
                ip = PARITY_NAMES.index(parity)
                iq = [0] if qg == "G" else [1, 2]
                if family == "E":
                    val = np.sum(weights[:, 0:2, ip, :][:, :, iq], axis=(1, 2))
                elif family == "z":
                    val = np.sum(weights[:, 2, ip, :][:, iq], axis=1)
                else:
                    val = np.sum(weights[:, 3, ip, :][:, iq], axis=1)
                labels.append(f"{family}_{qg}_{parity}")
                cols.append(val)
    return np.asarray(labels), np.stack(cols, axis=1)


def _projected_chiral_states(energies, vecs, local_j, M: int):
    M = int(min(max(2, M), len(energies)))
    U = np.asarray(vecs[:, :M], dtype=complex)
    Ju = local_j[0] / np.sqrt(6.0)
    for op in local_j[1:]:
        Ju = Ju + op / np.sqrt(6.0)
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
        sigE = float(np.sqrt(max(0.0, E2 - Emean * Emean)))
        psi = U @ c
        local = np.asarray([np.vdot(psi, op @ psi).real for op in local_j], dtype=float)
        rows.append((float(jvals[idx]), Emean, sigE, local, prob))
    return jvals, rows


def _first_excited_multiplet(E: np.ndarray, tol: float):
    target = E[1]
    return np.flatnonzero(np.abs(E - target) <= float(tol))


def main():
    args = _args()
    solver = ED18Solver(
        RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0),
        primitive_filling=args.filling,
    )
    ops, local_j = _operator_dictionary(solver)
    Vs = np.asarray(args.V, dtype=float)
    subspaces = np.asarray(sorted(set(int(x) for x in args.subspace_sizes if int(x) >= 2)), dtype=int)
    if len(subspaces) == 0:
        raise ValueError("at least one subspace size >=2 is required")

    all_E=[]; all_amp=[]; all_w=[]; all_gw=[]
    first_labels=[]; first_group_weight=[]; first_mult=[]
    current_idx=[]; current_gap=[]; current_weight=[]; current_S=[]; current_chi=[]; current_deff=[]
    pjmin=[]; pjmax=[]; pEmin=[]; pEmax=[]; psmin=[]; psmax=[]; plmin=[]; plmax=[]; ppmin=[]; ppmax=[]
    group_labels = None

    print("=== ED18 PBC low-energy mode diagnostic ===")
    print(f"sites=18 N={solver.n_particles} dim={solver.dimension} n_eigs={args.n_eigs}")
    print("momenta: Gamma, +/-Q with Q=(1/3,1/3)")
    print("uniform current = z-even @ Gamma")

    for V in Vs:
        spec = solver.solve(float(V), n_eigs=args.n_eigs, tol=args.tol, maxiter=args.maxiter, v0=None)
        E = np.asarray(spec.energies, dtype=float)
        vecs = np.asarray(spec.eigenvectors, dtype=complex)
        amp, w = _fingerprint(vecs, ops)
        labels, gw = _grouped_weights(w)
        if group_labels is None:
            group_labels = labels

        mult = _first_excited_multiplet(E, args.degeneracy_tol)
        # Aggregate the first multiplet before classification, so arbitrary basis
        # rotations inside +/-Q degeneracies do not change the label.
        agg = np.sum(gw[mult], axis=0)
        ig = int(np.argmax(agg))
        flabel = str(labels[ig])
        fweight = float(agg[ig])

        wz0 = np.asarray(w[:, 2, 0, 0], dtype=float)  # z, even, Gamma
        if len(wz0) > 1:
            icur = 1 + int(np.argmax(wz0[1:]))
            cgap = float(E[icur] - E[0])
            cw = float(wz0[icur])
        else:
            icur = -1; cgap=np.nan; cw=np.nan
        gaps = E - E[0]
        good = (np.arange(len(E)) > 0) & (gaps > 1e-12)
        Slow = float(np.sum(wz0[good]))
        chilow = float(2.0 * np.sum(wz0[good] / gaps[good]))
        deff = float(2.0 * Slow / chilow) if chilow > 0 else np.nan

        print(f"\nV={V:g}  E0={E[0]:+.10f}  gap1={E[1]-E[0]:.8e}")
        print(f"  first multiplet n={mult.tolist()} strongest grouped fingerprint: {flabel}, W={fweight:.8e}")
        print(f"  dominant uniform-current state n*={icur}, gap={cgap:.8e}, w={cw:.8e}")
        print(f"  current low-spectrum S={Slow:.8e}, chi={chilow:.8e}, delta_eff={deff:.8e}")
        nprint = min(int(args.print_states), len(E)-1)
        for n in range(1, nprint+1):
            ign = int(np.argmax(gw[n]))
            print(f"  n={n:2d} gap={gaps[n]:.8e} best={labels[ign]} W={gw[n,ign]:.6e} w_zG={wz0[n]:.6e}")

        jmns=[]; jmxs=[]; emns=[]; emxs=[]; smns=[]; smxs=[]; lmns=[]; lmxs=[]; pmns=[]; pmxs=[]
        for M in subspaces:
            _, rows = _projected_chiral_states(E, vecs, local_j, int(M))
            rmin, rmax = rows
            jmns.append(rmin[0]); jmxs.append(rmax[0]); emns.append(rmin[1]); emxs.append(rmax[1]); smns.append(rmin[2]); smxs.append(rmax[2])
            lmns.append(rmin[3]); lmxs.append(rmax[3])
            pmin=np.zeros(int(subspaces[-1])); pmax=np.zeros(int(subspaces[-1])); pmin[:len(rmin[4])]=rmin[4]; pmax[:len(rmax[4])]=rmax[4]
            pmns.append(pmin); pmxs.append(pmax)
            print(f"  M={M:2d}: Jmin={rmin[0]:+.8f}, Jmax={rmax[0]:+.8f}, <H>-E0={rmax[1]-E[0]:.6e}, sigmaE={rmax[2]:.3e}")

        all_E.append(E); all_amp.append(amp); all_w.append(w); all_gw.append(gw)
        first_labels.append(flabel); first_group_weight.append(fweight); first_mult.append(len(mult))
        current_idx.append(icur); current_gap.append(cgap); current_weight.append(cw); current_S.append(Slow); current_chi.append(chilow); current_deff.append(deff)
        pjmin.append(jmns); pjmax.append(jmxs); pEmin.append(emns); pEmax.append(emxs); psmin.append(smns); psmax.append(smxs); plmin.append(lmns); plmax.append(lmxs); ppmin.append(pmns); ppmax.append(pmxs)

    arrE = np.asarray(all_E)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        V=Vs, ti=float(args.ti), t1=float(args.t1), t2=float(args.t2), filling=float(args.filling),
        energies=arrE, gaps=arrE-arrE[:, :1],
        components=np.asarray(COMP_NAMES), parities=np.asarray(PARITY_NAMES), q_names=np.asarray(Q_NAMES), q_vectors=np.asarray(Q_LIST),
        spectral_amplitudes=np.asarray(all_amp), spectral_weights=np.asarray(all_w),
        grouped_labels=np.asarray(group_labels), grouped_weights=np.asarray(all_gw),
        first_multiplet_label=np.asarray(first_labels), first_multiplet_weight=np.asarray(first_group_weight), first_multiplicity=np.asarray(first_mult, dtype=int),
        uniform_current_index=np.asarray(current_idx, dtype=int), uniform_current_gap=np.asarray(current_gap), uniform_current_weight=np.asarray(current_weight),
        uniform_current_S_low=np.asarray(current_S), uniform_current_chi_low=np.asarray(current_chi), uniform_current_delta_eff=np.asarray(current_deff),
        subspace_sizes=subspaces,
        projected_jmin=np.asarray(pjmin), projected_jmax=np.asarray(pjmax), projected_Emin=np.asarray(pEmin), projected_Emax=np.asarray(pEmax),
        projected_sigmaE_min=np.asarray(psmin), projected_sigmaE_max=np.asarray(psmax),
        projected_localJ_min=np.asarray(plmin), projected_localJ_max=np.asarray(plmax),
        projected_prob_min=np.asarray(ppmin), projected_prob_max=np.asarray(ppmax),
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
