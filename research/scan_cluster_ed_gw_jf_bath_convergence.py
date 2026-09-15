#!/usr/bin/env python3
"""Compare the cluster-ED+GW JF response across finite-bath backgrounds.

Generate backgrounds with different ``nbath`` (for example with
``scan_cluster_ed_gw_bath.py``), then pass the saved NPZ files here.  This
script runs the same JF response at the same q for every background and records
both physical convergence (leading susceptibility/eigenvector) and numerical
cost (bath-tangent build time, Krylov iterations and wall time).

Example
-------

    python scan_cluster_ed_gw_jf_bath_convergence.py \
        results/bath_scan/*_nbath4.npz \
        results/bath_scan/*_nbath5.npz \
        results/bath_scan/*_nbath6.npz \
        --q-index 0 0 --bath-rank 24
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


DEFAULT_CHANNELS = ("Ax", "Ay", "Az", "Bx", "By", "Bz")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    q = p.add_mutually_exclusive_group()
    q.add_argument("--q-index", nargs=2, type=int, default=[0, 0])
    q.add_argument("--q", nargs=2, type=float)
    p.add_argument("--bath-metric", choices=("delta", "g0"), default="delta")
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--bath-fd-scheme", choices=("centered", "forward"), default="centered")
    p.add_argument("--solver", choices=("gcrotmk", "gmres"), default="gcrotmk")
    p.add_argument("--jf-tol", type=float, default=1e-8)
    p.add_argument("--jf-maxiter", type=int, default=80)
    p.add_argument("--krylov-m", type=int, default=24)
    p.add_argument("--recycle-k", type=int, default=12)
    p.add_argument("--stage", choices=("split-mt", "full"), default="full")
    p.add_argument("--out", type=Path, default=Path("results/jf_bath_convergence"))
    return p.parse_args()


def _scalar(z, key, default=np.nan):
    if key not in z:
        return default
    a = np.asarray(z[key])
    if a.size != 1:
        return default
    return a.reshape(()).item()


def _run_one(args, source: Path, output: Path):
    cmd = [
        sys.executable,
        "scan_cluster_ed_gw_jf_q.py",
        str(source),
        "--channels",
        *[str(x) for x in args.channels],
        "--bath-metric", str(args.bath_metric),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-rank", str(args.bath_rank),
        "--bath-svd-rcond", str(args.bath_svd_rcond),
        "--bath-fd-step", str(args.bath_fd_step),
        "--bath-fd-scheme", str(args.bath_fd_scheme),
        "--solver", str(args.solver),
        "--jf-tol", str(args.jf_tol),
        "--jf-maxiter", str(args.jf_maxiter),
        "--krylov-m", str(args.krylov_m),
        "--recycle-k", str(args.recycle_k),
        "--stage", str(args.stage),
        "--out", str(output),
        "--quiet",
    ]
    if args.q is not None:
        cmd += ["--q", str(args.q[0]), str(args.q[1])]
    else:
        cmd += ["--q-index", str(args.q_index[0]), str(args.q_index[1])]
    t0 = perf_counter()
    subprocess.run(cmd, check=True)
    return perf_counter() - t0


def _phase_overlap(v, ref):
    v = np.asarray(v, dtype=complex).reshape(-1)
    ref = np.asarray(ref, dtype=complex).reshape(-1)
    nv = np.linalg.norm(v)
    nr = np.linalg.norm(ref)
    if nv == 0.0 or nr == 0.0:
        return np.nan
    return float(abs(np.vdot(ref, v)) / (nv * nr))


def main():
    args = _args()
    for p in args.inputs:
        if not p.exists():
            raise FileNotFoundError(p)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    vectors = []
    output_files = []
    for i, source in enumerate(args.inputs):
        with np.load(source, allow_pickle=False) as z:
            nbath = len(np.asarray(z["bath_energies"]).reshape(-1))
            background_residual = float(_scalar(z, "final_error"))
            background_bath_error = float(_scalar(z, "bath_fit_error"))
            background_iterations = int(_scalar(z, "iterations", -1))
            background_elapsed = float(np.sum(np.asarray(z.get("elapsed_history", []), dtype=float)))
            background_mismatch = float(_scalar(z, "impurity_mismatch"))

        output = args.out / f"jf_nbath{nbath}_{i}.npz"
        print(f"\n=== JF bath convergence: nbath={nbath}, source={source} ===", flush=True)
        wall = _run_one(args, source, output)
        output_files.append(str(output))

        with np.load(output, allow_pickle=False) as z:
            lam = np.asarray(z["lambda_max"], dtype=float)
            pos = int(np.argmax(lam))
            vals = np.asarray(z["eigenvalues"], dtype=float)
            vecs = np.asarray(z["eigenvectors"], dtype=complex)
            leading_vec = vecs[pos, :, 0]
            vectors.append(leading_vec)
            row = {
                "nbath": int(nbath),
                "background_iterations": background_iterations,
                "background_residual": background_residual,
                "background_bath_fit": background_bath_error,
                "background_Gimp_Gc": background_mismatch,
                "background_elapsed_s": background_elapsed,
                "jf_wall_s": float(wall),
                "tangent_build_s": float(_scalar(z, "bath_tangent_build_seconds")),
                "tangent_rank": int(_scalar(z, "bath_tangent_rank", -1)),
                "tangent_condition": float(_scalar(z, "bath_tangent_condition")),
                "lambda_max": float(lam[pos]),
                "lambda_2": float(vals[pos, 1]) if vals.shape[1] > 1 else np.nan,
                "mean_krylov_iterations": float(np.mean(np.asarray(z["iterations"])[pos])),
                "max_response_residual": float(np.max(np.asarray(z["response_residuals"])[pos])),
                "result_file": str(output),
                "source_file": str(source),
            }
        rows.append(row)
        print(
            f"nbath={nbath}: lambda={row['lambda_max']:+.8e}, "
            f"rank={row['tangent_rank']}, tangent={row['tangent_build_s']:.1f}s, "
            f"JF wall={row['jf_wall_s']:.1f}s, mean Krylov={row['mean_krylov_iterations']:.1f}",
            flush=True,
        )

    # Use the largest nbath as the reference mode and report phase-insensitive
    # eigenvector overlaps.  This is more informative than lambda alone when two
    # channels are close.
    ref_index = int(np.argmax([r["nbath"] for r in rows]))
    ref = vectors[ref_index]
    for row, vec in zip(rows, vectors):
        row["leading_mode_overlap_with_largest_nbath"] = _phase_overlap(vec, ref)

    csv_path = args.out / "jf_bath_convergence.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.out / "jf_bath_convergence.npz"
    np.savez_compressed(
        npz_path,
        nbath=np.asarray([r["nbath"] for r in rows], dtype=int),
        background_residual=np.asarray([r["background_residual"] for r in rows]),
        background_bath_fit=np.asarray([r["background_bath_fit"] for r in rows]),
        background_Gimp_Gc=np.asarray([r["background_Gimp_Gc"] for r in rows]),
        background_elapsed_s=np.asarray([r["background_elapsed_s"] for r in rows]),
        jf_wall_s=np.asarray([r["jf_wall_s"] for r in rows]),
        tangent_build_s=np.asarray([r["tangent_build_s"] for r in rows]),
        tangent_rank=np.asarray([r["tangent_rank"] for r in rows], dtype=int),
        tangent_condition=np.asarray([r["tangent_condition"] for r in rows]),
        lambda_max=np.asarray([r["lambda_max"] for r in rows]),
        lambda_2=np.asarray([r["lambda_2"] for r in rows]),
        mean_krylov_iterations=np.asarray([r["mean_krylov_iterations"] for r in rows]),
        max_response_residual=np.asarray([r["max_response_residual"] for r in rows]),
        mode_overlap=np.asarray([r["leading_mode_overlap_with_largest_nbath"] for r in rows]),
        leading_eigenvectors=np.asarray(vectors),
        source_files=np.asarray([str(x) for x in args.inputs]),
        result_files=np.asarray(output_files),
    )

    print("\n=== finite-bath convergence summary ===", flush=True)
    print(
        "nbath  bg_bath_err  bg_Gimp/Gc   lambda_max      mode_overlap  "
        "tangent_s  JF_wall_s  krylov",
        flush=True,
    )
    for r in sorted(rows, key=lambda x: x["nbath"]):
        print(
            f"{r['nbath']:5d}  {r['background_bath_fit']:11.3e}  "
            f"{r['background_Gimp_Gc']:11.3e}  {r['lambda_max']:+12.5e}  "
            f"{r['leading_mode_overlap_with_largest_nbath']:12.6f}  "
            f"{r['tangent_build_s']:9.1f}  {r['jf_wall_s']:9.1f}  "
            f"{r['mean_krylov_iterations']:6.1f}",
            flush=True,
        )
    print(f"saved {csv_path}", flush=True)
    print(f"saved {npz_path}", flush=True)


if __name__ == "__main__":
    main()
