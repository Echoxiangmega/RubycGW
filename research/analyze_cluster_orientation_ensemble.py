#!/usr/bin/env python3
"""Compare three C3-related cluster-orientation ED+GW backgrounds.

The individual orientation runs are ordinary converged cluster-ED+GW fixed
points in three gauge-equivalent primitive-cell choices.  This script reports
how strongly the six-site cluster approximation depends on that choice and
forms gauge-restored lattice averages for diagnostic purposes.

The averaged lattice fields are *not* themselves claimed to be a fixed point of
one impurity problem.  For response calculations one should average the three
orientation-resolved susceptibilities rather than run an ordinary JF solve on
the averaged checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.cluster_orientation import transform_between_orientations


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs=3, type=Path, help="orientation 0/1/2 NPZ files, in any order")
    p.add_argument("--save", type=Path, default=None, help="optional diagnostic ensemble NPZ")
    return p.parse_args()


def _scalar(z, key):
    return np.asarray(z[key]).reshape(()).item()


def _load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]).copy() for k in z.files}
    if "cluster_orientation" not in d:
        raise ValueError(f"{path} has no cluster_orientation metadata")
    d["_path"] = str(path)
    return d


def _check_same(ds, key, *, atol=1e-12):
    vals = [float(np.asarray(d[key]).reshape(())) for d in ds]
    if not np.allclose(vals, vals[0], rtol=1e-11, atol=atol):
        raise ValueError(f"orientation files disagree in {key}: {vals}")


def _density_E(density):
    n = np.asarray(density, dtype=float).reshape(6)
    xA = 2.0 * n[0] - n[1] - n[2]
    yA = np.sqrt(3.0) * (n[2] - n[1])
    xB = 2.0 * n[3] - n[4] - n[5]
    # Physical B-frame convention has the opposite local y handedness.
    yB = np.sqrt(3.0) * (n[4] - n[5])
    return np.asarray([xA, yA, xB, yB], dtype=float)


def _spread(n):
    n = np.asarray(n, dtype=float).reshape(6)
    return max(float(np.ptp(n[:3])), float(np.ptp(n[3:])))


def main():
    args = _args()
    ds = [_load(p) for p in args.inputs]
    by_ori = {int(np.asarray(d["cluster_orientation"]).reshape(())): d for d in ds}
    if set(by_ori) != {0, 1, 2}:
        raise ValueError(f"need exactly orientations 0,1,2; got {sorted(by_ori)}")
    ds = [by_ori[r] for r in (0, 1, 2)]

    for key in ("Lx", "Ly", "V", "Vprime", "Vcross", "filling", "T", "ti", "t1", "t2"):
        _check_same(ds, key)

    print("=== three-orientation cluster ED+GW diagnostic ===")
    print(
        f"L={int(_scalar(ds[0],'Lx'))}x{int(_scalar(ds[0],'Ly'))}, "
        f"V={float(_scalar(ds[0],'V')):g}, V'={float(_scalar(ds[0],'Vprime')):g}, "
        f"Vx={float(_scalar(ds[0],'Vcross')):g}, T={float(_scalar(ds[0],'T')):g}"
    )

    densities = []
    mus = []
    for r, d in enumerate(ds):
        conv = bool(_scalar(d, "converged"))
        err = float(_scalar(d, "final_error"))
        mu = float(_scalar(d, "mu"))
        n = np.asarray(d["density"], dtype=float)
        e = _density_E(n)
        densities.append(n)
        mus.append(mu)
        print(
            f"ori {r}: converged={conv}, residual={err:.3e}, mu={mu:+.9f}, "
            f"density={np.array2string(n, precision=7)}, "
            f"(xA,yA,xB,yB)={np.array2string(e, precision=6)}"
        )

    navg = np.mean(np.asarray(densities), axis=0)
    print(
        "ensemble density average = "
        f"{np.array2string(navg, precision=8)}, C3-spread={_spread(navg):.3e}, "
        f"E={np.array2string(_density_E(navg), precision=6)}"
    )
    print(
        f"mu mean={np.mean(mus):+.9f}, orientation spread="
        f"{np.ptp(np.asarray(mus)):.3e}"
    )

    # Put all lattice matrix fields back in orientation-0 gauge before averaging.
    avg_payload = {}
    for key in ("G", "Sigma_emb", "Sigma_GW_lattice"):
        fields = [
            transform_between_orientations(np.asarray(d[key]), r, 0)
            for r, d in enumerate(ds)
        ]
        avg_payload[key] = np.mean(np.asarray(fields), axis=0)
        pairspread = max(
            float(np.max(np.abs(fields[a] - fields[b]), initial=0.0))
            for a in range(3) for b in range(a + 1, 3)
        )
        print(f"{key}: max pairwise orientation difference in common gauge = {pairspread:.3e}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            ensemble_kind=np.asarray("three_orientation_gauge_average_diagnostic"),
            not_single_impurity_fixed_point=np.asarray(True),
            cluster_orientations=np.asarray([0, 1, 2], dtype=int),
            source_files=np.asarray([d["_path"] for d in ds]),
            density=np.asarray(navg),
            mu=np.asarray(float(np.mean(mus))),
            G=np.asarray(avg_payload["G"]),
            Sigma_emb=np.asarray(avg_payload["Sigma_emb"]),
            Sigma_GW_lattice=np.asarray(avg_payload["Sigma_GW_lattice"]),
            Lx=np.asarray(ds[0]["Lx"]), Ly=np.asarray(ds[0]["Ly"]),
            V=np.asarray(ds[0]["V"]), Vprime=np.asarray(ds[0]["Vprime"]),
            Vcross=np.asarray(ds[0]["Vcross"]), filling=np.asarray(ds[0]["filling"]),
            T=np.asarray(ds[0]["T"]), ti=np.asarray(ds[0]["ti"]),
            t1=np.asarray(ds[0]["t1"]), t2=np.asarray(ds[0]["t2"]),
            omega=np.asarray(ds[0]["omega"]), Omega=np.asarray(ds[0]["Omega"]),
        )
        print(f"saved diagnostic ensemble {args.save}")


if __name__ == "__main__":
    main()
