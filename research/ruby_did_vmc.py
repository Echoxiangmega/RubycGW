#!/usr/bin/env python3
"""Fixed-number projected BCS/Pfaffian VMC for spinless Ruby fermions.

This is a small, independent reference implementation of the pairing ansatz.
It uses the native Ruby geometry, the full physical hopping and interaction,
and a projected E-orbital A--B pairing source.  It has no NetKet dependency.

Example (from the repository root):
  OPENBLAS_NUM_THREADS=1 python research/ruby_did_vmc.py \
      --cluster 2x2 --nf 6 --t1 -0.5 --t2 0.5 --V 4 \
      --form d+ --delta 0.4 --mu -1 --vtri 0.5 \
      --burn 200 --samples 500 --interval 3

Small parameter scan (increase samples and refine near the minimum):
  OPENBLAS_NUM_THREADS=1 python research/ruby_did_vmc.py \
      --grid-form s,d+,d- --grid-delta 0.2,0.4,0.8 \
      --grid-mu=-1.4,-1.0 --grid-vtri 0.3,0.7

Run independent parameter values and use the VMC energy to optimize the
ansatz.  The code deliberately does not claim a superconducting phase merely
because a paired variational state lowers the energy on a finite torus.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ed_benchmark" / "native_ed"))
from geometry import RubyPBC  # noqa: E402


def form_factors(form: str) -> np.ndarray:
    """Three A-to-B directions: (0,0), (0,1), (-1,0)."""
    omega = np.exp(2j * np.pi / 3)
    forms = {
        "s": [1, 1, 1],
        "d+": [1, omega, omega**2],
        "d-": [1, omega**2, omega],
        "d1": [2, -1, -1],
        "d2": [0, 1, -1],
    }
    if form not in forms:
        raise ValueError(f"form must be one of {tuple(forms)}")
    x = np.asarray(forms[form], dtype=np.complex128)
    return x / np.max(np.abs(x))


def orbital_pair_matrix() -> np.ndarray:
    """Physical site matrix of the effective honeycomb singlet.

    W columns are the triangle E doublet in site order (0,1,2) and (3,4,5).
    In the unrotated A/B chirality basis the pair is a_+ b_+ + a_- b_-.
    """
    omega = np.exp(2j * np.pi / 3)
    W = np.array([[1, 1], [omega, omega.conjugate()],
                  [omega**2, omega]], complex) / np.sqrt(3)
    M = W.conj() @ W.conj().T
    if not np.allclose(M.imag, 0, atol=1e-12):
        raise RuntimeError("The physical pair matrix should be real")
    return M.real


def pairing_source(g: RubyPBC, delta: float, form: str) -> np.ndarray:
    """Antisymmetric 6Nc-by-6Nc source on honeycomb A--B bonds."""
    M = orbital_pair_matrix()
    phase = form_factors(form)
    directions = ((0, 0), (0, 1), (-1, 0))
    D = np.zeros((g.Ns, g.Ns), complex)
    for ci, R in enumerate(g.reps):
        for direction, amp in zip(directions, phase):
            cj, _ = g.reduce((R[0] + direction[0], R[1] + direction[1]))
            ai = slice(6 * ci, 6 * ci + 3)
            bj = slice(6 * cj + 3, 6 * cj + 6)
            D[ai, bj] += delta * amp * M
            D[bj, ai] -= delta * amp * M.T
    if not np.allclose(D + D.T, 0, atol=1e-12):
        raise RuntimeError("Fermionic pairing source is not antisymmetric")
    return D


def hopping_matrix(g: RubyPBC, mu: float) -> np.ndarray:
    h = -float(mu) * np.eye(g.Ns, dtype=complex)
    for e in g.edges:
        h[e.i, e.j] += g.t[e.kind]
        h[e.j, e.i] += g.t[e.kind]
    return h


def bcs_pair_function(h: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve BdG and obtain exp(c^dag F c^dag/2) times blocked orbitals.

    Some strictly d-wave finite tori have unpaired occupied Gamma orbitals
    because the d-wave gap vanishes there.  Their columns are kept as Q,
    rather than making an ill-conditioned inverse U.  The number-projected
    amplitude is a bordered Pfaffian on [F_occ,Q_occ;-Q_occ.T,0].
    """
    n = h.shape[0]
    B = np.block([[h, D], [D.conj().T, -h.T]])
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    if eigenvalues[n - 1] > -1e-9 or eigenvalues[n] < 1e-9:
        raise ValueError("Gapless BdG seed; change mu or delta/boundary conditions")
    positive = eigenvectors[:, n:]
    U, V = positive[:n], positive[n:]
    _, singular, vh = np.linalg.svd(U)
    nb = int(np.count_nonzero(singular < 1e-9))
    C = vh.conj().T
    Q = np.conj(V @ C[:, n - nb:]) if nb else np.empty((n, 0), complex)
    P = null_space(Q.conj().T) if nb else np.eye(n, dtype=complex)
    Un, Vn = U @ C[:, :n - nb], V @ C[:, :n - nb]
    if np.linalg.cond(Un.conj().T @ P) > 1e11:
        raise ValueError("Unstable BdG seed; change mu or pairing amplitude")
    Fc = -np.linalg.solve(Un.conj().T @ P, Vn.conj().T @ P.conj())
    F = P @ Fc @ P.T
    defect = np.max(np.abs(F + F.T))
    if defect > 1e-6 * max(1, np.max(np.abs(F))):
        raise RuntimeError(f"BdG pair function not antisymmetric: {defect}")
    return (F - F.T) / 2, Q


def pfaffian(A: np.ndarray) -> complex:
    """Pivoted skew elimination.  Occupied indices must be in ascending order."""
    A = np.asarray(A, complex).copy()
    n = len(A)
    if n % 2:
        return 0j
    out = 1 + 0j
    for k in range(0, n, 2):
        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))
        if abs(A[k, p]) < 1e-14:
            return 0j
        if p != k + 1:
            A[[k + 1, p], :] = A[[p, k + 1], :]
            A[:, [k + 1, p]] = A[:, [p, k + 1]]
            out = -out
        pivot = A[k, k + 1]
        out *= pivot
        if k + 2 < n:
            u = A[k, k + 2:].copy()
            v = A[k + 1, k + 2:].copy()
            A[k + 2:, k + 2:] += (np.outer(v, u) - np.outer(u, v)) / pivot
    return out


def amplitude(occ: np.ndarray, F: np.ndarray, Q: np.ndarray, vtri: float) -> complex:
    """P_N |BCS> times soft intra-triangle density Jastrow."""
    if (len(occ) + Q.shape[1]) % 2 or len(occ) < Q.shape[1]:
        raise ValueError("Nf must support the number and parity of blocked orbitals")
    tri = np.bincount(occ // 3, minlength=F.shape[0] // 3)
    doubles = int(np.sum(tri * (tri - 1) // 2))
    if Q.shape[1]:
        A = np.block([[F[np.ix_(occ, occ)], Q[occ]],
                      [-Q[occ].T, np.zeros((Q.shape[1], Q.shape[1]))]])
    else:
        A = F[np.ix_(occ, occ)]
    return pfaffian(A) * np.exp(-vtri * doubles)


def local_energy(occ: np.ndarray, psi: complex, F: np.ndarray, Q: np.ndarray,
                 vtri: float, g: RubyPBC, V: float) -> complex:
    """Exact Hamiltonian matrix elements evaluated on one VMC configuration."""
    occupied = np.zeros(g.Ns, bool)
    occupied[occ] = True
    tri = np.bincount(occ // 3, minlength=g.Ns // 3)
    energy = complex(V * np.sum(tri * (tri - 1) // 2))
    for edge in g.edges:
        i, j = edge.i, edge.j
        if occupied[i] == occupied[j]:
            continue
        a = i if occupied[i] else j
        b = j if occupied[i] else i
        target = np.sort(np.r_[occ[occ != a], b])
        sign = -1 if np.count_nonzero((occ > min(i, j)) & (occ < max(i, j))) % 2 else 1
        energy += g.t[edge.kind] * sign * amplitude(target, F, Q, vtri) / psi
    return energy


def run(g: RubyPBC, nf: int, V: float, form: str, delta: float, mu: float,
        vtri: float, burn: int, samples: int, interval: int, seed: int) -> dict:
    if not np.isclose(g.t["t1"], -g.t["t2"]):
        raise ValueError("This projected E-orbital singlet is defined for t1=-t2")
    if nf % 2 or nf <= 0 or nf >= g.Ns:
        raise ValueError("Nf must be even and satisfy 0<Nf<Ns")
    if delta <= 0 or samples < 20 or burn < 0 or interval < 1:
        raise ValueError("Require delta>0, samples>=20, burn>=0, interval>=1")
    F, Q = bcs_pair_function(hopping_matrix(g, mu), pairing_source(g, delta, form))
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        occ = np.sort(rng.choice(g.Ns, nf, replace=False))
        psi = amplitude(occ, F, Q, vtri)
        if abs(psi) > 1e-14:
            break
    else:
        raise RuntimeError("Could not find a nonzero sampled Pfaffian")

    values = []
    accepted = attempted = 0
    for sweep in range(burn + samples * interval):
        for _ in range(nf):
            occupied = np.zeros(g.Ns, bool)
            occupied[occ] = True
            a = int(rng.choice(occ))
            b = int(rng.choice(np.flatnonzero(~occupied)))
            proposed = np.sort(np.r_[occ[occ != a], b])
            newpsi = amplitude(proposed, F, Q, vtri)
            attempted += 1
            if rng.random() < min(1., abs(newpsi / psi)**2):
                occ, psi = proposed, newpsi
                accepted += 1
        if sweep >= burn and (sweep - burn) % interval == 0:
            values.append(local_energy(occ, psi, F, Q, vtri, g, V))
    values = np.asarray(values)
    nb = min(20, len(values) // 10)
    blocks = np.array_split(values.real, nb)
    means = np.array([x.mean() for x in blocks])
    return {
        "E": float(values.real.mean()),
        "E_per_site": float(values.real.mean() / g.Ns),
        "error_block": float(means.std(ddof=1) / np.sqrt(nb)),
        "imag_E_check": float(abs(values.imag.mean())),
        "acceptance": accepted / attempted,
        "blocked_orbitals": Q.shape[1],
        "samples": len(values),
        "cluster": g.name, "Ns": g.Ns, "Nf": nf, "V": V,
        "t": g.t["t"], "t1": g.t["t1"], "t2": g.t["t2"],
        "form": form, "delta": delta, "mu": mu, "vtri": vtri,
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", default="2x2")
    p.add_argument("--nf", type=int, default=6)
    p.add_argument("--t", type=float, default=1.)
    p.add_argument("--t1", type=float, default=-.5)
    p.add_argument("--t2", type=float, default=.5)
    p.add_argument("--V", type=float, default=4.)
    p.add_argument("--form", choices=["s", "d+", "d-", "d1", "d2"], default="d+")
    p.add_argument("--delta", type=float, default=.4)
    p.add_argument("--mu", type=float, default=-1.)
    p.add_argument("--vtri", type=float, default=.5)
    p.add_argument("--grid-delta", default="", help="Comma-separated delta scan; overrides --delta")
    p.add_argument("--grid-mu", default="", help="Comma-separated mu scan; overrides --mu")
    p.add_argument("--grid-vtri", default="", help="Comma-separated vtri scan; overrides --vtri")
    p.add_argument("--grid-form", default="", help="Comma-separated form scan; overrides --form")
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--interval", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    g = RubyPBC(args.cluster, t=args.t, t1=args.t1, t2=args.t2)
    deltas = [float(x) for x in args.grid_delta.split(",") if x] or [args.delta]
    mus = [float(x) for x in args.grid_mu.split(",") if x] or [args.mu]
    vtris = [float(x) for x in args.grid_vtri.split(",") if x] or [args.vtri]
    forms = [x for x in args.grid_form.split(",") if x] or [args.form]
    results = []
    for form, delta, mu, vtri in product(forms, deltas, mus, vtris):
        try:
            result = run(g, args.nf, args.V, form, delta, mu, vtri,
                         args.burn, args.samples, args.interval, args.seed)
            results.append(result)
        except ValueError as error:
            results.append({"form": form, "delta": delta, "mu": mu,
                            "vtri": vtri, "error": str(error)})
    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        # The winning grid point must be remeasured with more samples:
        # noisy short VMC runs are not definitive variational comparisons.
        results.sort(key=lambda row: row.get("E", float("inf")))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
