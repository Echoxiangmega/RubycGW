#!/usr/bin/env python3
"""Run q=0 eigenmode-seeded physical-pair cluster-ED+GW ordered branches.

The script consumes a V-filling scan summary plus overlap-tracked JF modes.
At one selected (V,filling) point it

1. collects physically CO/LC-like q=0 tracked modes,
2. deduplicates nearly identical static source form factors,
3. ramps a temporary source H_src=-h O_mode to zero using continuation,
4. evaluates the zero-source cluster-ED+GW Luttinger-Ward free energy, and
5. compares all converged zero-source endpoints on the same numerical footing.

Finite-q candidates are *not* discarded.  They are written to
pending_finite_q_candidates.csv because a genuine q!=0 ordered state requires
an enlarged translation-breaking multi-impurity supercell solver; the primitive
6-site nonlinear solver cannot represent k <-> k+q mixing.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = _REPO_ROOT / "research" / "run_cluster_ed_gw_vprime_vcross_orientation.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rubycgw.cluster_orientation import build_oriented_lattice_fields
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import build_h0
from rubycgw.pseudospin import primitive_cell_pseudospin_channels
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from vprime_study.cross_model import VPrimeCrossParameters


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--V", type=float, required=True)
    p.add_argument("--filling", type=float, required=True)
    p.add_argument("--tracked", type=Path, default=None)
    p.add_argument("--jf-dir", type=Path, default=None)
    p.add_argument("--branch-ids", type=int, nargs="*", default=None)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--source-overlap-dedup", type=float, default=0.985)
    p.add_argument(
        "--source-sequence",
        type=float,
        nargs="+",
        default=[0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
    )
    p.add_argument("--order-threshold", type=float, default=1e-5)
    p.add_argument("--free-energy-mismatch-max", type=float, default=5e-2)

    p.add_argument("--gw-max", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--embed-max", type=int, default=400)
    p.add_argument("--embed-tol", type=float, default=1e-7)
    p.add_argument(
        "--embed-mixing-method",
        choices=("linear", "pulay", "broyden"),
        default="broyden",
    )
    p.add_argument("--embed-mixing", type=float, default=0.5)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--quiet-solvers", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _scalar(d, key, cast=float):
    return cast(np.asarray(d[key]).reshape(()))


def _paths(args, summary, meta):
    tracked = args.tracked or (args.summary.parent / "tracked_modes" / "vn_tracked_modes.npz")
    jfdir = args.jf_dir or (args.summary.parent / "jf_allq")
    bgdir = args.summary.parent / "backgrounds"
    chk = bgdir / (
        f"cluster_ed_gw_vprime_vcross_ori{meta['orientation']}_"
        f"L{meta['Lx']}x{meta['Ly']}_V{_tag(args.V)}_"
        f"Vp{_tag(meta['Vprime'])}_Vx{_tag(meta['Vcross'])}_"
        f"fill{_tag(args.filling)}.npz"
    )
    jf = jfdir / (
        f"lambda_allq_ori{meta['orientation']}_L{meta['Lx']}x{meta['Ly']}_"
        f"V{_tag(args.V)}_Vp{_tag(meta['Vprime'])}_Vx{_tag(meta['Vcross'])}_"
        f"fill{_tag(args.filling)}.npz"
    )
    return tracked, chk, jf


def _metadata(summary):
    return dict(
        orientation=_scalar(summary, "orientation", int),
        Lx=_scalar(summary, "Lx", int),
        Ly=_scalar(summary, "Ly", int),
        Vprime=_scalar(summary, "Vprime", float),
        Vcross=_scalar(summary, "Vcross", float),
    )


def _exact_rows(tracked, V, filling):
    mask = (
        np.isclose(np.asarray(tracked["V"], dtype=float), float(V), rtol=0.0, atol=1e-10)
        & np.isclose(
            np.asarray(tracked["filling"], dtype=float),
            float(filling),
            rtol=0.0,
            atol=1e-10,
        )
    )
    return np.where(mask)[0]


def _mode_record(tracked, i):
    q = tuple(int(x) for x in np.asarray(tracked["q_index"][i], dtype=int))
    return dict(
        row=int(i),
        branch_id=int(tracked["branch_id"][i]),
        q=q,
        sector=str(tracked["sector"][i]),
        lam=complex(tracked["lambda_value"][i]),
        distance=float(abs(1.0 - complex(tracked["lambda_value"][i]))),
        channel_group=str(tracked["channel_group"][i]),
        dominant_channel=str(tracked["dominant_channel"][i]),
        co_even=float(tracked["co_even"][i]),
        co_odd=float(tracked["co_odd"][i]),
        lc_same=float(tracked["lc_same"][i]),
        lc_opposite=float(tracked["lc_opposite"][i]),
        uniform=float(tracked["uniform"][i]),
        source_mode_row=int(tracked["source_mode_row"][i]),
        overlap_previous=float(tracked["overlap_previous"][i]),
    )


def _normalized_q0_source(M):
    M = np.asarray(M, dtype=complex)
    M = 0.5 * (M + M.conj().T)
    M -= np.trace(M) * np.eye(M.shape[0], dtype=complex) / float(M.shape[0])
    n = float(np.linalg.norm(M))
    if n <= 1e-14:
        return np.zeros_like(M)
    return M / n


def _source_overlap(A, B):
    a = _normalized_q0_source(A).ravel()
    b = _normalized_q0_source(B).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-14 or nb <= 1e-14:
        return 0.0
    return float(abs(np.vdot(a, b)) / (na * nb))


def _select_candidates(args, tracked, jf):
    if "source_mode_row" not in tracked:
        raise RuntimeError(
            "tracked file predates source-row support; rerun the updated tracker"
        )
    if "mode_static_matrix" not in jf:
        raise RuntimeError(
            "JF file predates static source form factors; rerun the updated JF analyzer"
        )
    rows = _exact_rows(tracked, args.V, args.filling)
    records = [_mode_record(tracked, i) for i in rows]
    records = [
        r for r in records
        if r["channel_group"] in {"CO", "LC"}
    ]
    if args.branch_ids:
        wanted = set(int(x) for x in args.branch_ids)
        records = [r for r in records if r["branch_id"] in wanted]

    finite = [r for r in records if r["q"] != (0, 0)]
    q0 = sorted(
        [r for r in records if r["q"] == (0, 0)],
        key=lambda r: (r["distance"], -max(
            r["co_even"], r["co_odd"], r["lc_same"], r["lc_opposite"]
        )),
    )

    mats = np.asarray(jf["mode_static_matrix"], dtype=complex)
    selected = []
    for r in q0:
        row = int(r["source_mode_row"])
        if row < 0 or row >= len(mats):
            continue
        M = mats[row]
        duplicate = False
        for old in selected:
            Mold = mats[int(old["source_mode_row"])]
            if _source_overlap(M, Mold) >= float(args.source_overlap_dedup):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(r)
        if len(selected) >= int(args.max_candidates):
            break
    return selected, finite


def _run(cmd):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


def _background_meta(path):
    d = _load_npz(path)
    return dict(
        orientation=_scalar(d, "cluster_orientation", int),
        Lx=_scalar(d, "Lx", int),
        Ly=_scalar(d, "Ly", int),
        V=_scalar(d, "V", float),
        Vprime=_scalar(d, "Vprime", float),
        Vcross=_scalar(d, "Vcross", float),
        filling=_scalar(d, "filling", float),
        T=_scalar(d, "T", float),
        ti=_scalar(d, "ti", float),
        t1=_scalar(d, "t1", float),
        t2=_scalar(d, "t2", float),
        nw=len(np.asarray(d["omega"])) // 2,
        nomega=(len(np.asarray(d["Omega"])) - 1) // 2,
        nbath=len(np.asarray(d["bath_energies"])),
    )


def _runner_command(args, meta, outdir, seed, *, jf=None, mode_row=None, h=0.0):
    cmd = [
        sys.executable,
        str(RUNNER),
        "--orientation", str(meta["orientation"]),
        "--Lx", str(meta["Lx"]), "--Ly", str(meta["Ly"]),
        "--V", repr(float(meta["V"])),
        "--Vp", repr(float(meta["Vprime"])),
        "--Vx", repr(float(meta["Vcross"])),
        "--ti", repr(float(meta["ti"])),
        "--t1", repr(float(meta["t1"])),
        "--t2", repr(float(meta["t2"])),
        "--filling", repr(float(meta["filling"])),
        "--T", repr(float(meta["T"])),
        "--nw", str(meta["nw"]),
        "--nomega", str(meta["nomega"]),
        "--gw-max", str(args.gw_max),
        "--gw-tol", repr(float(args.gw_tol)),
        "--embed-max", str(args.embed_max),
        "--embed-tol", repr(float(args.embed_tol)),
        "--embed-mixing-method", str(args.embed_mixing_method),
        "--embed-mixing", repr(float(args.embed_mixing)),
        "--nbath", str(meta["nbath"]),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-fit-max-nfev", str(args.bath_fit_max_nfev),
        "--bath-fit-metric", "delta",
        "--complex-bath",
        "--evaluate-free-energy",
        "--out", str(outdir),
        "--continue-from", str(seed),
    ]
    if jf is not None:
        cmd.extend([
            "--source-mode-file", str(jf),
            "--source-mode-row", str(int(mode_row)),
            "--source-strength", repr(float(h)),
        ])
    if bool(args.quiet_solvers):
        cmd.extend(["--quiet-gw", "--quiet-embed"])
    return cmd


def _expected_output(outdir, meta, *, mode_row=None, h=0.0):
    suffix = ""
    if mode_row is not None:
        suffix = f"_mode{int(mode_row)}_h{_tag(h)}"
    return outdir / (
        f"cluster_ed_gw_vprime_vcross_ori{meta['orientation']}_"
        f"L{meta['Lx']}x{meta['Ly']}_V{_tag(meta['V'])}_"
        f"Vp{_tag(meta['Vprime'])}_Vx{_tag(meta['Vcross'])}_"
        f"fill{_tag(meta['filling'])}{suffix}.npz"
    )


def _endpoint_diagnostics(path):
    d = _load_npz(path)
    grid = MatsubaraGrid(
        nk1=_scalar(d, "Lx", int),
        nk2=_scalar(d, "Ly", int),
        nw=len(np.asarray(d["omega"])) // 2,
        nOmega=(len(np.asarray(d["Omega"])) - 1) // 2,
        T=_scalar(d, "T", float),
    )
    params = VPrimeCrossParameters(
        ti=_scalar(d, "ti", float),
        t1=_scalar(d, "t1", float),
        t2=_scalar(d, "t2", float),
        V=_scalar(d, "V", float),
        Vprime=_scalar(d, "Vprime", float),
        Vcross=_scalar(d, "Vcross", float),
    )
    h0base = build_h0(grid.kmesh(), params)
    h0, _ = build_oriented_lattice_fields(
        h0base,
        np.zeros_like(h0base),
        _scalar(d, "cluster_orientation", int),
    )
    rho = one_body_density_matrix_tail(
        np.asarray(d["G"], dtype=complex),
        grid,
        h0,
        _scalar(d, "mu", float),
        np.asarray(d["Sigma_H"], dtype=complex),
    )
    rc = np.mean(rho, axis=(0, 1))

    ps = primitive_cell_pseudospin_channels()
    channel_amp = {}
    for name in ("x_even", "y_even", "x_odd", "y_odd", "z_even", "z_odd"):
        K = np.asarray(ps[name], dtype=complex)
        den = max(float(np.vdot(K, K).real), 1e-300)
        channel_amp[name] = complex(np.vdot(K, rc) / den)

    M = np.asarray(d.get("source_matrix", np.zeros((6, 6))), dtype=complex)
    den = max(float(np.vdot(M, M).real), 1e-300)
    source_order = complex(np.vdot(M, rc) / den) if np.max(np.abs(M)) > 0 else 0.0j

    return dict(
        converged=bool(_scalar(d, "converged", bool)),
        residual=float(_scalar(d, "final_error", float)),
        mu=float(_scalar(d, "mu", float)),
        source_order_real=float(source_order.real),
        source_order_imag=float(source_order.imag),
        source_order_abs=float(abs(source_order)),
        x_even_abs=float(abs(channel_amp["x_even"])),
        y_even_abs=float(abs(channel_amp["y_even"])),
        x_odd_abs=float(abs(channel_amp["x_odd"])),
        y_odd_abs=float(abs(channel_amp["y_odd"])),
        lc_same_abs=float(abs(channel_amp["z_even"])),
        lc_opposite_abs=float(abs(channel_amp["z_odd"])),
        F=float(np.asarray(d.get("helmholtz_free_energy", np.nan)).reshape(())),
        Omega=float(np.asarray(d.get("grand_potential", np.nan)).reshape(())),
        fe_mismatch=float(
            np.asarray(d.get("free_energy_gimp_gc_mismatch", np.nan)).reshape(())
        ),
        bath_error=float(np.asarray(d.get("bath_fit_error", np.nan)).reshape(())),
        path=str(path),
    )


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    args = _args()
    summary = _load_npz(args.summary)
    meta_summary = _metadata(summary)
    tracked_path, background, jf_path = _paths(args, summary, meta_summary)
    if not tracked_path.exists():
        raise FileNotFoundError(tracked_path)
    if not background.exists():
        raise FileNotFoundError(background)
    if not jf_path.exists():
        raise FileNotFoundError(jf_path)

    tracked = _load_npz(tracked_path)
    jf = _load_npz(jf_path)
    selected, finite = _select_candidates(args, tracked, jf)
    meta = _background_meta(background)

    outdir = args.out or (
        args.summary.parent
        / "ordered_branches"
        / f"V{_tag(args.V)}_fill{_tag(args.filling)}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    pending_rows = []
    for r in finite:
        pending_rows.append({
            "branch_id": r["branch_id"],
            "q1": r["q"][0],
            "q2": r["q"][1],
            "sector": r["sector"],
            "lambda_real": r["lam"].real,
            "lambda_imag": r["lam"].imag,
            "distance_to_one": r["distance"],
            "channel_group": r["channel_group"],
            "dominant_channel": r["dominant_channel"],
            "source_mode_row": r["source_mode_row"],
            "status": "requires_finite_q_multi_impurity_supercell",
        })
    _write_csv(outdir / "pending_finite_q_candidates.csv", pending_rows)

    print(
        f"q=0 unique candidates: {len(selected)}; "
        f"finite-q pending candidates: {len(pending_rows)}",
        flush=True,
    )

    endpoint_rows = []

    # Fair zero-source reference: use the same complex-bath variational class as
    # every ordered branch.
    normal_dir = outdir / "normal_complex_bath"
    normal_dir.mkdir(parents=True, exist_ok=True)
    normal_out = _expected_output(normal_dir, meta)
    if not normal_out.exists():
        _run(_runner_command(args, meta, normal_dir, background, h=0.0))
    normal_diag = _endpoint_diagnostics(normal_out)
    endpoint_rows.append({
        "branch_id": -1,
        "seed_channel": "normal",
        "seed_sector": "none",
        "lambda_real": np.nan,
        "lambda_imag": np.nan,
        "distance_to_one": np.nan,
        "source_mode_row": -1,
        "survives_h0": False,
        **normal_diag,
    })

    for ic, cand in enumerate(selected, start=1):
        bdir = outdir / (
            f"branch_{cand['branch_id']}_{cand['dominant_channel']}"
        )
        bdir.mkdir(parents=True, exist_ok=True)
        seed = background
        endpoint = None
        print(
            f"\n=== candidate {ic}/{len(selected)}: branch={cand['branch_id']}, "
            f"{cand['dominant_channel']}, lambda={cand['lam']} ===",
            flush=True,
        )
        for h in args.source_sequence:
            out = _expected_output(
                bdir,
                meta,
                mode_row=cand["source_mode_row"],
                h=float(h),
            )
            if not out.exists():
                _run(
                    _runner_command(
                        args,
                        meta,
                        bdir,
                        seed,
                        jf=jf_path,
                        mode_row=cand["source_mode_row"],
                        h=float(h),
                    )
                )
            endpoint = out
            seed = out
        if endpoint is None:
            continue

        diag = _endpoint_diagnostics(endpoint)
        survives = bool(
            diag["converged"]
            and diag["source_order_abs"] >= float(args.order_threshold)
        )
        endpoint_rows.append({
            "branch_id": cand["branch_id"],
            "seed_channel": cand["dominant_channel"],
            "seed_sector": cand["sector"],
            "lambda_real": cand["lam"].real,
            "lambda_imag": cand["lam"].imag,
            "distance_to_one": cand["distance"],
            "source_mode_row": cand["source_mode_row"],
            "survives_h0": survives,
            **diag,
        })

    reliable = [
        r for r in endpoint_rows
        if r["converged"]
        and np.isfinite(r["F"])
        and np.isfinite(r["fe_mismatch"])
        and r["fe_mismatch"] <= float(args.free_energy_mismatch_max)
    ]
    fmin = min((r["F"] for r in reliable), default=np.nan)
    for r in endpoint_rows:
        r["free_energy_reliable"] = bool(
            r["converged"]
            and np.isfinite(r["F"])
            and np.isfinite(r["fe_mismatch"])
            and r["fe_mismatch"] <= float(args.free_energy_mismatch_max)
        )
        r["DeltaF"] = float(r["F"] - fmin) if np.isfinite(fmin) and np.isfinite(r["F"]) else np.nan

    endpoint_rows.sort(
        key=lambda r: (
            not bool(r["free_energy_reliable"]),
            np.inf if not np.isfinite(r["F"]) else float(r["F"]),
        )
    )
    _write_csv(outdir / "zero_source_endpoints.csv", endpoint_rows)

    print("\n=== zero-source endpoint comparison ===")
    for r in endpoint_rows:
        print(
            f"branch={r['branch_id']:>4}, seed={r['seed_channel']:<14}, "
            f"survive={str(r['survives_h0']):<5}, "
            f"F={r['F']:+.10e}, dF={r['DeltaF']:+.3e}, "
            f"Gimp/Gc={r['fe_mismatch']:.3e}, "
            f"|Oseed|={r['source_order_abs']:.3e}, "
            f"reliable={r['free_energy_reliable']}"
        )
    print(f"saved branch comparison in {outdir}")


if __name__ == "__main__":
    main()
