#!/usr/bin/env python3
"""Compare all tracked q/channel ordered branches at one (V,filling).

Every candidate, including q=0, is solved in the same folded multi-impurity
cluster-ED+GW supercell.  Temporary eigenmode-derived sources are ramped to
zero, zero-source endpoints are deduplicated by their Green functions, and
their fixed-filling Luttinger-Ward Helmholtz free energies are compared.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = _REPO_ROOT / "research" / "run_multicell_cluster_ed_gw_eigenmode.py"


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--V", type=float, required=True)
    p.add_argument("--filling", type=float, required=True)
    p.add_argument("--tracked", type=Path, default=None)
    p.add_argument("--jf-dir", type=Path, default=None)
    p.add_argument("--branch-ids", type=int, nargs="*", default=None)
    p.add_argument("--max-candidates", type=int, default=16)
    p.add_argument("--source-overlap-dedup", type=float, default=0.985)
    p.add_argument(
        "--source-sequence",
        type=float,
        nargs="+",
        default=[0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
    )
    p.add_argument("--order-threshold", type=float, default=1e-5)
    p.add_argument("--endpoint-dedup-tol", type=float, default=2e-4)
    p.add_argument("--free-energy-mismatch-max", type=float, default=5e-2)
    p.add_argument("--max-iter", type=int, default=160)
    p.add_argument("--tol", type=float, default=2e-6)
    p.add_argument("--mixing", type=float, default=0.45)
    p.add_argument("--impurity-mixing", type=float, default=1.0)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=400)
    p.add_argument("--quiet-solvers", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _tag(x):
    return f"{float(x):g}"


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _scalar(d, key, cast=float):
    return cast(np.asarray(d[key]).reshape(()))


def _meta(summary):
    return dict(
        orientation=_scalar(summary, "orientation", int),
        Lx=_scalar(summary, "Lx", int),
        Ly=_scalar(summary, "Ly", int),
        Vprime=_scalar(summary, "Vprime", float),
        Vcross=_scalar(summary, "Vcross", float),
    )


def _locate(args, summary):
    m = _meta(summary)
    tracked = args.tracked or (
        args.summary.parent / "tracked_modes" / "vn_tracked_modes.npz"
    )
    jfdir = args.jf_dir or (args.summary.parent / "jf_allq")
    bg = args.summary.parent / "backgrounds" / (
        f"cluster_ed_gw_vprime_vcross_ori{m['orientation']}_"
        f"L{m['Lx']}x{m['Ly']}_V{_tag(args.V)}_"
        f"Vp{_tag(m['Vprime'])}_Vx{_tag(m['Vcross'])}_"
        f"fill{_tag(args.filling)}.npz"
    )
    jf = jfdir / (
        f"lambda_allq_ori{m['orientation']}_L{m['Lx']}x{m['Ly']}_"
        f"V{_tag(args.V)}_Vp{_tag(m['Vprime'])}_Vx{_tag(m['Vcross'])}_"
        f"fill{_tag(args.filling)}.npz"
    )
    return tracked, bg, jf


def _candidate_rows(args, tracked):
    mask = (
        np.isclose(tracked["V"], args.V, rtol=0.0, atol=1e-10)
        & np.isclose(tracked["filling"], args.filling, rtol=0.0, atol=1e-10)
    )
    idx = np.where(mask)[0]
    rows = []
    wanted = None if not args.branch_ids else set(int(x) for x in args.branch_ids)
    for i in idx:
        group = str(tracked["channel_group"][i])
        bid = int(tracked["branch_id"][i])
        if group not in {"CO", "LC"}:
            continue
        if wanted is not None and bid not in wanted:
            continue
        rows.append(dict(
            tracked_row=int(i),
            branch_id=bid,
            q=tuple(int(x) for x in tracked["q_index"][i]),
            sector=str(tracked["sector"][i]),
            channel_group=group,
            dominant_channel=str(tracked["dominant_channel"][i]),
            lam=complex(tracked["lambda_value"][i]),
            distance=float(abs(1.0 - complex(tracked["lambda_value"][i]))),
            source_mode_row=int(tracked["source_mode_row"][i]),
        ))
    return sorted(rows, key=lambda r: r["distance"])


def _canonical_source(jf, row):
    M = np.asarray(jf["mode_static_matrix"][int(row)], dtype=complex)
    q = tuple(int(x) for x in jf["mode_q_index"][int(row)])
    # Compare the primitive-cell form factor only within the same q.
    n = float(np.linalg.norm(M))
    return (M / n if n > 1e-14 else np.zeros_like(M)), q


def _deduplicate_candidates(args, rows, jf):
    selected = []
    for r in rows:
        M, q = _canonical_source(jf, r["source_mode_row"])
        duplicate = False
        for old in selected:
            Mold, qold = _canonical_source(jf, old["source_mode_row"])
            if q != qold:
                continue
            ov = float(abs(np.vdot(M.ravel(), Mold.ravel())))
            if ov >= float(args.source_overlap_dedup):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(r)
        if len(selected) >= int(args.max_candidates):
            break
    return selected


def _run(cmd):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


def _runner_cmd(args, background, outdir, *, jf=None, row=None, h=0.0, seed=None):
    cmd = [
        sys.executable, str(RUNNER),
        "--background", str(background),
        "--source-strength", repr(float(h)),
        "--max-iter", str(args.max_iter),
        "--tol", repr(float(args.tol)),
        "--mixing", repr(float(args.mixing)),
        "--impurity-mixing", repr(float(args.impurity_mixing)),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-fit-max-nfev", str(args.bath_fit_max_nfev),
        "--out", str(outdir),
    ]
    if jf is not None:
        cmd.extend(["--mode-file", str(jf), "--mode-row", str(int(row))])
    if seed is not None:
        cmd.extend(["--continue-from", str(seed)])
    if bool(args.quiet_solvers):
        cmd.append("--quiet")
    return cmd


def _expected_output(background, outdir, *, jf=None, row=None, h=0.0):
    bg = _load(background)
    Lx = _scalar(bg, "Lx", int)
    Ly = _scalar(bg, "Ly", int)
    ori = _scalar(bg, "cluster_orientation", int)
    V = _scalar(bg, "V", float)
    filling = _scalar(bg, "filling", float)
    if jf is None:
        qtag = "normal"
    else:
        j = _load(jf)
        q = tuple(int(x) for x in j["mode_q_index"][int(row)])
        qtag = f"q{q[0]}_{q[1]}_mode{int(row)}"
    return outdir / (
        f"multicell_ed_gw_ori{ori}_L{Lx}x{Ly}_"
        f"V{_tag(V)}_fill{_tag(filling)}_{qtag}_h{_tag(h)}.npz"
    )


def _diag(path):
    d = _load(path)
    mismatch = np.asarray(d["free_energy_gimp_gc_mismatch"], dtype=float)
    dominant_amp = complex(
        np.asarray(d.get("dominant_order_amplitude", 0.0j)).reshape(())
    )
    dominant_q = np.asarray(
        d.get("dominant_order_q", np.asarray([-1, -1], dtype=int)),
        dtype=int,
    ).reshape(2)
    dominant_channel = str(
        np.asarray(d.get("dominant_order_channel", "unknown")).reshape(())
    )
    return dict(
        path=str(path),
        converged=bool(_scalar(d, "converged", bool)),
        residual=float(_scalar(d, "final_error", float)),
        mu=float(_scalar(d, "mu", float)),
        source_expect_real=float(
            complex(np.asarray(d["source_expectation_per_pc"]).reshape(())).real
        ),
        source_expect_imag=float(
            complex(np.asarray(d["source_expectation_per_pc"]).reshape(())).imag
        ),
        source_expect_abs=float(
            abs(complex(np.asarray(d["source_expectation_per_pc"]).reshape(())))
        ),
        final_order_channel=dominant_channel,
        final_order_q1=int(dominant_q[0]),
        final_order_q2=int(dominant_q[1]),
        final_order_abs=float(abs(dominant_amp)),
        F_per_pc=float(_scalar(d, "helmholtz_free_energy_per_primitive_cell", float)),
        Omega_super=float(_scalar(d, "grand_potential_supercell", float)),
        fe_mismatch_max=float(np.max(mismatch)),
        impurity_mismatch_max=float(np.max(np.asarray(d["impurity_mismatch"], dtype=float))),
        bath_error_max=float(np.max(np.asarray(d["bath_fit_error"], dtype=float))),
    )


def _state_distance(path_a, path_b):
    a = _load(Path(path_a))
    b = _load(Path(path_b))
    Ga = np.asarray(a["G"], dtype=complex)
    Gb = np.asarray(b["G"], dtype=complex)
    den = max(float(np.linalg.norm(Ga.ravel())), float(np.linalg.norm(Gb.ravel())), 1e-300)
    return float(np.linalg.norm((Ga-Gb).ravel()) / den)


def _assign_endpoint_groups(rows, tol):
    reps = []
    for r in rows:
        group = None
        dist = np.nan
        for gid, rep in enumerate(reps):
            d = _state_distance(r["path"], rep["path"])
            if d <= float(tol):
                group = gid
                dist = d
                break
        if group is None:
            group = len(reps)
            reps.append(r)
            dist = 0.0
        r["endpoint_group"] = int(group)
        r["distance_to_group_rep"] = float(dist)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = _args()
    summary = _load(args.summary)
    tracked_path, background, jf_path = _locate(args, summary)
    for p in (tracked_path, background, jf_path):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    tracked = _load(tracked_path)
    jf = _load(jf_path)
    if "source_mode_row" not in tracked or "mode_static_matrix" not in jf:
        raise RuntimeError(
            "source-ready tracked/JF files required; rerun the updated scan + tracker"
        )

    rows = _candidate_rows(args, tracked)
    candidates = _deduplicate_candidates(args, rows, jf)
    outdir = args.out or (
        args.summary.parent
        / "all_mode_branches"
        / f"V{_tag(args.V)}_fill{_tag(args.filling)}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"tracked physical modes={len(rows)}, unique source candidates={len(candidates)}",
        flush=True,
    )

    endpoints = []

    normal_dir = outdir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    normal_out = _expected_output(background, normal_dir, h=0.0)
    if not normal_out.exists():
        _run(_runner_cmd(args, background, normal_dir, h=0.0))
    nd = _diag(normal_out)
    endpoints.append(dict(
        branch_id=-1,
        seed_q1=-1,
        seed_q2=-1,
        seed_sector="none",
        seed_channel="normal",
        lambda_real=np.nan,
        lambda_imag=np.nan,
        distance_to_one=np.nan,
        source_mode_row=-1,
        survives_h0=False,
        **nd,
    ))

    for ic, cand in enumerate(candidates, 1):
        q1, q2 = cand["q"]
        bdir = outdir / (
            f"branch_{cand['branch_id']}_q{q1}_{q2}_{cand['dominant_channel']}"
        )
        bdir.mkdir(parents=True, exist_ok=True)
        seed = None
        endpoint = None
        print(
            f"\n=== {ic}/{len(candidates)} branch={cand['branch_id']} "
            f"q={cand['q']} {cand['dominant_channel']} "
            f"lambda={cand['lam'].real:+.6f}{cand['lam'].imag:+.2e}i ===",
            flush=True,
        )
        for h in args.source_sequence:
            out = _expected_output(
                background,
                bdir,
                jf=jf_path,
                row=cand["source_mode_row"],
                h=float(h),
            )
            if not out.exists():
                _run(_runner_cmd(
                    args,
                    background,
                    bdir,
                    jf=jf_path,
                    row=cand["source_mode_row"],
                    h=float(h),
                    seed=seed,
                ))
            seed = out
            endpoint = out
        if endpoint is None:
            continue
        d = _diag(endpoint)
        endpoints.append(dict(
            branch_id=cand["branch_id"],
            seed_q1=q1,
            seed_q2=q2,
            seed_sector=cand["sector"],
            seed_channel=cand["dominant_channel"],
            lambda_real=cand["lam"].real,
            lambda_imag=cand["lam"].imag,
            distance_to_one=cand["distance"],
            source_mode_row=cand["source_mode_row"],
            survives_h0=bool(
                d["converged"]
                and d["final_order_abs"] >= float(args.order_threshold)
            ),
            **d,
        ))

    _assign_endpoint_groups(endpoints, args.endpoint_dedup_tol)
    reliable = [
        r for r in endpoints
        if r["converged"]
        and np.isfinite(r["F_per_pc"])
        and r["fe_mismatch_max"] <= float(args.free_energy_mismatch_max)
    ]
    fmin = min((r["F_per_pc"] for r in reliable), default=np.nan)
    for r in endpoints:
        r["free_energy_reliable"] = bool(
            r["converged"]
            and np.isfinite(r["F_per_pc"])
            and r["fe_mismatch_max"] <= float(args.free_energy_mismatch_max)
        )
        r["DeltaF_per_pc"] = (
            float(r["F_per_pc"] - fmin)
            if np.isfinite(fmin) and np.isfinite(r["F_per_pc"])
            else np.nan
        )

    endpoints.sort(
        key=lambda r: (
            not bool(r["free_energy_reliable"]),
            np.inf if not np.isfinite(r["F_per_pc"]) else r["F_per_pc"],
        )
    )
    _write_csv(outdir / "all_mode_zero_source_endpoints.csv", endpoints)

    print("\n=== unified zero-source free-energy comparison ===")
    for r in endpoints:
        print(
            f"group={r['endpoint_group']:>2}, branch={r['branch_id']:>4}, "
            f"q=({r['seed_q1']},{r['seed_q2']}), "
            f"seed={r['seed_channel']:<14}, survive={str(r['survives_h0']):<5}, "
            f"final={r['final_order_channel']}@({r['final_order_q1']},{r['final_order_q2']}), "
            f"|A|={r['final_order_abs']:.3e}, "
            f"F/pc={r['F_per_pc']:+.10e}, dF/pc={r['DeltaF_per_pc']:+.3e}, "
            f"max Gimp/Gc={r['fe_mismatch_max']:.3e}, "
            f"reliable={r['free_energy_reliable']}"
        )
    print(f"saved: {outdir / 'all_mode_zero_source_endpoints.csv'}")


if __name__ == "__main__":
    main()
