#!/usr/bin/env python3
"""Build the final V-filling ordered-phase map from all-mode branch free energies.

For each requested grid point this driver runs the unified multi-impurity
all-q branch search, then selects the lowest reliable zero-source Helmholtz
free-energy endpoint.  Distinct endpoint groups within --degeneracy-tol are
reported as near-degenerate rather than forcing a unique phase label.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_DRIVER = _REPO_ROOT / "research" / "run_cluster_ed_gw_all_mode_branches.py"


def _csv_floats(text):
    if text is None:
        return None
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    return vals or None


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--V-values", type=_csv_floats, default=None)
    p.add_argument("--fillings", type=_csv_floats, default=None)
    p.add_argument("--max-candidates", type=int, default=16)
    p.add_argument("--source-overlap-dedup", type=float, default=0.985)
    p.add_argument(
        "--source-sequence",
        type=float,
        nargs="+",
        default=[0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
    )
    p.add_argument("--order-threshold", type=float, default=1e-5)
    p.add_argument("--free-energy-mismatch-max", type=float, default=5e-2)
    p.add_argument("--degeneracy-tol", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=160)
    p.add_argument("--tol", type=float, default=2e-6)
    p.add_argument("--mixing", type=float, default=0.45)
    p.add_argument("--impurity-mixing", type=float, default=1.0)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=400)
    p.add_argument("--quiet-solvers", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _tag(x):
    return f"{float(x):g}"


def _read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def _float(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _phase_label(row, order_threshold):
    amp = _float(row, "final_order_abs", 0.0)
    if not np.isfinite(amp) or amp < float(order_threshold):
        return "symmetric"
    ch = str(row.get("final_order_channel", "unknown"))
    q1 = int(float(row.get("final_order_q1", -1)))
    q2 = int(float(row.get("final_order_q2", -1)))
    if ch in {"x_even", "y_even"}:
        family = "CO-even"
    elif ch in {"x_odd", "y_odd"}:
        family = "CO-odd"
    elif ch == "z_even":
        family = "LC-same"
    elif ch == "z_odd":
        family = "LC-opposite"
    else:
        family = ch
    return f"{family}@q({q1},{q2})"


def _distinct_groups(rows):
    groups = {}
    for r in rows:
        gid = int(float(r.get("endpoint_group", -1)))
        old = groups.get(gid)
        if old is None or _float(r, "F_per_pc") < _float(old, "F_per_pc"):
            groups[gid] = r
    return list(groups.values())


def _summarize_point(csv_path, order_threshold, mismatch_max, degeneracy_tol):
    rows = _read_csv(csv_path)
    reliable = [
        r for r in rows
        if _bool(r.get("converged", False))
        and _bool(r.get("free_energy_reliable", False))
        and np.isfinite(_float(r, "F_per_pc"))
        and _float(r, "fe_mismatch_max", np.inf) <= float(mismatch_max)
    ]
    groups = sorted(_distinct_groups(reliable), key=lambda r: _float(r, "F_per_pc"))
    if not groups:
        return dict(
            resolved=False,
            phase="unresolved",
            best_F=np.nan,
            second_F=np.nan,
            free_energy_gap=np.nan,
            best_group=-1,
            best_branch=-1,
            best_q=(-1, -1),
            best_channel="",
            best_order=np.nan,
            near_degenerate=False,
            competing_phase="",
        )
    best = groups[0]
    best_F = _float(best, "F_per_pc")
    best_phase = _phase_label(best, order_threshold)
    second_F = np.nan
    gap = np.nan
    competing = ""
    near = False
    if len(groups) > 1:
        second = groups[1]
        second_F = _float(second, "F_per_pc")
        gap = float(second_F - best_F)
        competing = _phase_label(second, order_threshold)
        near = bool(np.isfinite(gap) and gap <= float(degeneracy_tol))
    phase = (
        f"degenerate:{best_phase}|{competing}"
        if near and competing and competing != best_phase
        else best_phase
    )
    return dict(
        resolved=True,
        phase=phase,
        best_F=best_F,
        second_F=second_F,
        free_energy_gap=gap,
        best_group=int(float(best.get("endpoint_group", -1))),
        best_branch=int(float(best.get("branch_id", -1))),
        best_q=(
            int(float(best.get("final_order_q1", -1))),
            int(float(best.get("final_order_q2", -1))),
        ),
        best_channel=str(best.get("final_order_channel", "")),
        best_order=_float(best, "final_order_abs"),
        near_degenerate=near,
        competing_phase=competing,
    )


def _run_point(args, V, filling, point_dir):
    cmd = [
        sys.executable,
        str(BRANCH_DRIVER),
        str(args.summary),
        "--V", repr(float(V)),
        "--filling", repr(float(filling)),
        "--max-candidates", str(args.max_candidates),
        "--source-overlap-dedup", repr(float(args.source_overlap_dedup)),
        "--source-sequence",
        *[repr(float(x)) for x in args.source_sequence],
        "--order-threshold", repr(float(args.order_threshold)),
        "--free-energy-mismatch-max", repr(float(args.free_energy_mismatch_max)),
        "--max-iter", str(args.max_iter),
        "--tol", repr(float(args.tol)),
        "--mixing", repr(float(args.mixing)),
        "--impurity-mixing", repr(float(args.impurity_mixing)),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-fit-max-nfev", str(args.bath_fit_max_nfev),
        "--out", str(point_dir),
    ]
    if bool(args.quiet_solvers):
        cmd.append("--quiet-solvers")
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot(path, rows):
    phases = sorted(set(r["phase"] for r in rows if r["resolved"]))
    fig, ax = plt.subplots()
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
    for i, phase in enumerate(phases):
        rr = [r for r in rows if r["resolved"] and r["phase"] == phase]
        ax.scatter(
            [r["V"] for r in rr],
            [r["filling"] for r in rr],
            marker=markers[i % len(markers)],
            label=phase,
        )
    bad = [r for r in rows if not r["resolved"]]
    if bad:
        ax.scatter(
            [r["V"] for r in bad],
            [r["filling"] for r in bad],
            marker="x",
            label="unresolved",
        )
    ax.set_xlabel("V")
    ax.set_ylabel("filling n")
    ax.set_title("Cluster-ED+GW ordered-phase map from zero-source free energies")
    if phases or bad:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = _args()
    summary = _load(args.summary)
    Vs = (
        np.asarray(args.V_values, dtype=float)
        if args.V_values is not None
        else np.asarray(summary["V_values"], dtype=float)
    )
    fillings = (
        np.asarray(args.fillings, dtype=float)
        if args.fillings is not None
        else np.asarray(summary["fillings"], dtype=float)
    )
    outdir = args.out or (args.summary.parent / "ordered_phase_map")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for filling in fillings:
        for V in Vs:
            point_dir = outdir / f"V{_tag(V)}_fill{_tag(filling)}"
            csv_path = point_dir / "all_mode_zero_source_endpoints.csv"
            try:
                if bool(args.rerun) or not csv_path.exists():
                    point_dir.mkdir(parents=True, exist_ok=True)
                    _run_point(args, float(V), float(filling), point_dir)
                s = _summarize_point(
                    csv_path,
                    args.order_threshold,
                    args.free_energy_mismatch_max,
                    args.degeneracy_tol,
                )
            except BaseException as exc:
                print(f"[FAILED] V={V:g}, filling={filling:g}: {exc}", flush=True)
                s = dict(
                    resolved=False,
                    phase="unresolved",
                    best_F=np.nan,
                    second_F=np.nan,
                    free_energy_gap=np.nan,
                    best_group=-1,
                    best_branch=-1,
                    best_q=(-1, -1),
                    best_channel="",
                    best_order=np.nan,
                    near_degenerate=False,
                    competing_phase="",
                )
                if not bool(args.keep_going):
                    raise
            rows.append(dict(
                V=float(V),
                filling=float(filling),
                resolved=bool(s["resolved"]),
                phase=str(s["phase"]),
                best_F_per_pc=float(s["best_F"]),
                second_F_per_pc=float(s["second_F"]),
                free_energy_gap_per_pc=float(s["free_energy_gap"]),
                best_endpoint_group=int(s["best_group"]),
                best_branch_id=int(s["best_branch"]),
                best_q1=int(s["best_q"][0]),
                best_q2=int(s["best_q"][1]),
                best_channel=str(s["best_channel"]),
                best_order_abs=float(s["best_order"]),
                near_degenerate=bool(s["near_degenerate"]),
                competing_phase=str(s["competing_phase"]),
            ))
            _write_csv(outdir / "vn_ordered_phase_table.csv", rows)
            _plot(outdir / "vn_ordered_phase_map.png", rows)

    np.savez_compressed(
        outdir / "vn_ordered_phase_map.npz",
        V=np.asarray([r["V"] for r in rows], dtype=float),
        filling=np.asarray([r["filling"] for r in rows], dtype=float),
        resolved=np.asarray([r["resolved"] for r in rows], dtype=bool),
        phase=np.asarray([r["phase"] for r in rows]),
        best_F_per_pc=np.asarray([r["best_F_per_pc"] for r in rows], dtype=float),
        free_energy_gap_per_pc=np.asarray(
            [r["free_energy_gap_per_pc"] for r in rows], dtype=float
        ),
        best_branch_id=np.asarray([r["best_branch_id"] for r in rows], dtype=int),
        best_q_index=np.asarray(
            [[r["best_q1"], r["best_q2"]] for r in rows], dtype=int
        ),
        best_channel=np.asarray([r["best_channel"] for r in rows]),
        best_order_abs=np.asarray([r["best_order_abs"] for r in rows], dtype=float),
        near_degenerate=np.asarray([r["near_degenerate"] for r in rows], dtype=bool),
    )
    print(f"saved final table/map in {outdir}")


if __name__ == "__main__":
    main()
