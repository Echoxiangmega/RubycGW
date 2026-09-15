#!/usr/bin/env python3
"""Finite-size quantum ED of the Ruby strong-coupling pseudospin Hamiltonian."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vprime_study.effective_pseudospin_ed import (
    CHANNELS,
    effective_couplings,
    leading_mode_label,
    solve_effective_pseudospin_ed,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--t", type=float, default=0.2, help="t1=t2=t in the strong-coupling projection")
    p.add_argument("--V", type=float, default=1.8)
    p.add_argument("--Vprime", "--Vp", dest="Vprime", type=float, default=-0.10)
    p.add_argument("--Vcross", "--Vx", dest="Vcross", type=float, default=-0.05)
    p.add_argument("--nev", type=int, default=4, help="lowest eigenstates retained per parity sector")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--degeneracy-tol", type=float, default=1e-7)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def main():
    a = _args()
    nspin = 2 * int(a.Lx) * int(a.Ly)
    if nspin > 20:
        raise ValueError(
            f"requested {nspin} pseudospins; direct parity-resolved ED becomes expensive. "
            "Use <=18 for routine runs (e.g. 3x3), or <=20 with sufficient memory."
        )
    Jn, Jm, Jz = effective_couplings(
        t=a.t, V=a.V, Vprime=a.Vprime, Vcross=a.Vcross
    )
    print(
        "=== Ruby effective-pseudospin quantum ED ===\n"
        f"torus={a.Lx}x{a.Ly} cells, Nspin={nspin}, "
        f"t={a.t:g}, V={a.V:g}, V'={a.Vprime:g}, Vx={a.Vcross:g}\n"
        f"Jn={Jn:+.10f}, Jm={Jm:+.10f}, Jz={Jz:+.10f}",
        flush=True,
    )
    r = solve_effective_pseudospin_ed(
        Lx=a.Lx, Ly=a.Ly, Jn=Jn, Jm=Jm, Jz=Jz,
        nev=a.nev, tol=a.tol, maxiter=a.maxiter,
        degeneracy_tol=a.degeneracy_tol,
    )
    ig = int(np.argmax(r.lambda_max))
    comp, parity, weight = leading_mode_label(r.sf_eigenvectors[ig, :, 0])
    gamma = int(np.argmin(np.linalg.norm(r.q_centered, axis=1)))
    ixy = int(np.argmax(r.xy_max))
    izs = int(np.argmax(r.z_same))
    print(
        f"E0={r.ground_energy:+.12f}, E0/N={r.ground_energy/r.nspin:+.12f}, "
        f"gap={r.gap:.6e}, GS degeneracy={r.ground_degeneracy}\n"
        f"global S(q): lambda={r.lambda_max[ig]:.8f} "
        f"(lambda/N={r.lambda_max[ig]/r.nspin:.8f}) at "
        f"q=({r.q_centered[ig,0]:+.6f},{r.q_centered[ig,1]:+.6f}), "
        f"mode={comp}_{parity}, weight={weight:.6f}\n"
        f"z_same@Gamma={r.z_same[gamma]:.8f} "
        f"({r.z_same[gamma]/r.nspin:.8f}/spin), "
        f"z_same(max)={r.z_same[izs]:.8f} at "
        f"({r.q_centered[izs,0]:+.6f},{r.q_centered[izs,1]:+.6f})\n"
        f"xy(max)={r.xy_max[ixy]:.8f} at "
        f"({r.q_centered[ixy,0]:+.6f},{r.q_centered[ixy,1]:+.6f})",
        flush=True,
    )

    out = a.out
    if out is None:
        out = Path("results/effective_pseudospin_ed") / (
            f"effective_ed_L{a.Lx}x{a.Ly}_V{_tag(a.V)}_Vp{_tag(a.Vprime)}_"
            f"Vx{_tag(a.Vcross)}.npz"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        model=np.asarray("Ruby_120deg_compass_plus_current"),
        channels=CHANNELS,
        Lx=int(a.Lx), Ly=int(a.Ly), nspin=int(r.nspin),
        t=float(a.t), V=float(a.V), Vprime=float(a.Vprime), Vp=float(a.Vprime),
        Vcross=float(a.Vcross), Vx=float(a.Vcross),
        Jn=float(Jn), Jm=float(Jm), Jz=float(Jz),
        ground_energy=float(r.ground_energy), gap=float(r.gap),
        ground_degeneracy=int(r.ground_degeneracy),
        ground_parities=np.asarray(r.ground_parities),
        energies_even=np.asarray(r.energies_even), energies_odd=np.asarray(r.energies_odd),
        q_raw=np.asarray(r.q_raw), q_centered=np.asarray(r.q_centered),
        structure_factor=np.asarray(r.structure_factor),
        sf_eigenvalues=np.asarray(r.sf_eigenvalues),
        sf_eigenvectors=np.asarray(r.sf_eigenvectors),
        lambda_max=np.asarray(r.lambda_max),
        z_same=np.asarray(r.z_same), z_opposite=np.asarray(r.z_opposite),
        xy_max=np.asarray(r.xy_max),
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
