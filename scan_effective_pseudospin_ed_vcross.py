#!/usr/bin/env python3
"""Scan Vx in the finite-size quantum ED effective pseudospin model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from vprime_study.effective_pseudospin_ed import (
    effective_couplings,
    leading_mode_label,
    solve_effective_pseudospin_ed,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--t", type=float, default=0.2)
    p.add_argument("--V", type=float, default=1.8)
    p.add_argument("--Vprime", "--Vp", dest="Vprime", type=float, default=-0.10)
    p.add_argument(
        "--Vx-values", type=str,
        default="0,-0.03,-0.04,-0.05,-0.0555556,-0.06,-0.07",
        help="comma-separated crossed interactions",
    )
    p.add_argument("--nev", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--degeneracy-tol", type=float, default=1e-7)
    p.add_argument("--out", type=Path, default=Path("results/effective_pseudospin_ed/vcross_scan.csv"))
    return p.parse_args()


def main():
    a = _args()
    nspin = 2 * a.Lx * a.Ly
    if nspin > 20:
        raise ValueError("routine direct ED is limited to <=20 pseudospins")
    values = [float(x.strip()) for x in a.Vx_values.split(",") if x.strip()]
    rows = []
    for vx in values:
        Jn, Jm, Jz = effective_couplings(t=a.t, V=a.V, Vprime=a.Vprime, Vcross=vx)
        r = solve_effective_pseudospin_ed(
            Lx=a.Lx, Ly=a.Ly, Jn=Jn, Jm=Jm, Jz=Jz,
            nev=a.nev, tol=a.tol, maxiter=a.maxiter,
            degeneracy_tol=a.degeneracy_tol,
        )
        ig = int(np.argmax(r.lambda_max))
        comp, parity, weight = leading_mode_label(r.sf_eigenvectors[ig, :, 0])
        gamma = int(np.argmin(np.linalg.norm(r.q_centered, axis=1)))
        ixy = int(np.argmax(r.xy_max))
        row = {
            "Vcross": vx,
            "Jn": Jn, "Jm": Jm, "Jz": Jz,
            "E0": r.ground_energy, "E0_per_spin": r.ground_energy / r.nspin,
            "gap": r.gap, "ground_degeneracy": r.ground_degeneracy,
            "global_lambda": r.lambda_max[ig],
            "global_lambda_per_spin": r.lambda_max[ig] / r.nspin,
            "global_q1": r.q_centered[ig, 0], "global_q2": r.q_centered[ig, 1],
            "global_component": comp, "global_parity": parity,
            "global_component_weight": weight,
            "z_same_gamma": r.z_same[gamma],
            "z_same_gamma_per_spin": r.z_same[gamma] / r.nspin,
            "xy_max": r.xy_max[ixy], "xy_max_per_spin": r.xy_max[ixy] / r.nspin,
            "xy_q1": r.q_centered[ixy, 0], "xy_q2": r.q_centered[ixy, 1],
        }
        rows.append(row)
        print(
            f"Vx={vx:+.6f}  E0/N={row['E0_per_spin']:+.8f}  gap={row['gap']:.3e}  "
            f"Smax/N={row['global_lambda_per_spin']:.6f} "
            f"q=({row['global_q1']:+.3f},{row['global_q2']:+.3f}) {comp}_{parity}  "
            f"zSameG/N={row['z_same_gamma_per_spin']:.6f}  "
            f"xyMax/N={row['xy_max_per_spin']:.6f}",
            flush=True,
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
