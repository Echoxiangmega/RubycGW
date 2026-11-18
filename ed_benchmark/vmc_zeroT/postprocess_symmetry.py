#!/usr/bin/env python3
"""Postprocess existing Hex18 VMC states with ED-sector space-group projection.

No VMC optimization is rerun. The script reads final_state.npz files from an
existing benchmark directory, determines the exact ordinary space-group irrep
of the unique ED ground state, applies projection-after-variation (PAV), and
recomputes energy, fidelity, and the same CO/BOW/LC observables exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

import geometry as native_geometry
import observables as native_observables

from benchmark_observables import benchmark_observables
from run_benchmark import solve_ed, atomic_json
from symmetry_projection import (
    build_ordinary_space_group,
    characterize_unique_state,
    project_to_characters,
    projected_sector_residual,
    commutator_error,
)


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def add_differences(row, ed):
    row["delta_E_vs_ED"] = float(row["E"]) - float(ed["E"])
    row["relative_E_error"] = abs(row["delta_E_vs_ED"]) / max(abs(float(ed["E"])), 1e-15)
    for key in ("CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"):
        ref = float(ed[key]); val = float(row[key])
        row[f"delta_{key}_vs_ED"] = val - ref
        row[f"relative_{key}_error"] = abs(val-ref) / max(abs(ref), 1e-12)
    return row


def projected_row(summary, psi_p, weight, H, ed_gs, ed_row, group, chars, native_states):
    t0 = time.perf_counter()
    E = float(np.vdot(psi_p, H.operator @ psi_p).real)
    obs = benchmark_observables(H.g, native_states, psi_p, native_observables)
    fid = float(np.sum(np.abs(ed_gs.conj().T @ psi_p) ** 2).real)
    residual = projected_sector_residual(group, chars, psi_p)
    elapsed = time.perf_counter() - t0

    row = {
        "method": str(summary["method"]) + "-PAV",
        "seed": summary.get("seed", ""),
        "projection": "ED ordinary-space-group sector",
        "symmetry_sector_weight": float(weight),
        "projected_sector_residual": float(residual),
        "E": E,
        "E_per_site": E / H.g.Ns,
        "fidelity_ground_subspace": fid,
        "time_projection_postprocess_s": elapsed,
        **obs.as_dict(),
    }
    return add_differences(row, ed_row)


def best_rows(rows):
    out = []
    ed = [r for r in rows if r["method"] == "ED"]
    out.extend(ed)
    for prefix in ("VMC-Slater", "VMC-Slater-Jastrow"):
        raw = [r for r in rows if r["method"] == prefix]
        pav = [r for r in rows if r["method"] == prefix + "-PAV"]
        if raw:
            out.append(min(raw, key=lambda r: float(r["E"])))
        if pav:
            out.append(min(pav, key=lambda r: float(r["E"])))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--input", required=True, help="Existing VMC benchmark output directory")
    ap.add_argument("--ed-k", type=int, default=8)
    ap.add_argument("--ed-degeneracy-tol", type=float, default=1e-8)
    ap.add_argument("--commutator-tol", type=float, default=1e-9)
    args = ap.parse_args()

    root = Path(args.input).resolve()
    settings_path = root / "settings.json"
    if not settings_path.is_file():
        raise FileNotFoundError(settings_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    nf = int(settings["nf"])
    t = float(settings["t"]); t1 = float(settings["t1"]); t2 = float(settings["t2"])
    V = float(settings["V"])
    g = native_geometry.RubyPBC("ring3", t=t, t1=t1, t2=t2)

    H, evals, ed_gs, ed_row = solve_ed(
        g, nf, V, k=args.ed_k, degeneracy_tol=args.ed_degeneracy_tol
    )
    if ed_gs.shape[1] != 1:
        raise RuntimeError(
            f"This first PAV implementation requires a unique ED ground state; "
            f"found degeneracy={ed_gs.shape[1]}. Use a nondegenerate benchmark point "
            "or extend the projector by diagonalizing irreps inside the ground manifold."
        )

    group = build_ordinary_space_group(g, H.states)
    if len(group) < 1:
        raise RuntimeError("No ordinary space-group operations generated")
    comm_err, comm_name = commutator_error(group, H.operator, n_tests=2)
    if comm_err > args.commutator_tol:
        raise RuntimeError(
            f"Second-quantized symmetry does not commute with H: "
            f"worst={comm_name}, error={comm_err:.3e}"
        )
    chars = characterize_unique_state(group, ed_gs[:, 0], tol=1e-8)

    sector = {
        "cluster": "ring3",
        "Ns": g.Ns,
        "Nf": nf,
        "t": t, "t1": t1, "t2": t2, "V": V,
        "group": "ordinary translations x C3v",
        "group_order": len(group),
        "commutator_error": comm_err,
        "characters": [
            {
                "name": op.name,
                "real": float(chi.real),
                "imag": float(chi.imag),
                "phase_over_pi": float(np.angle(chi) / np.pi),
            }
            for op, chi in zip(group, chars)
        ],
    }
    atomic_json(root / "ed_symmetry_sector.json", sector)

    ed = dict(ed_row)
    ed["projection"] = "exact ED"
    ed["symmetry_sector_weight"] = 1.0
    ed["projected_sector_residual"] = 0.0
    rows = [ed]

    state_files = sorted(root.glob("*/final_state.npz"))
    if not state_files:
        raise RuntimeError(f"No */final_state.npz found under {root}")

    for state_file in state_files:
        sub = state_file.parent
        summary_path = sub / "summary.json"
        if not summary_path.is_file():
            print(f"[skip] missing {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(state_file, allow_pickle=False) as z:
            psi = np.asarray(z["psi_native"], dtype=np.complex128)
            saved_states = np.asarray(z["native_states"], dtype=np.uint64)
        if not np.array_equal(saved_states, H.states):
            raise RuntimeError(f"Basis mismatch in {state_file}")

        raw = dict(summary)
        raw["projection"] = "none"
        raw["symmetry_sector_weight"] = None
        raw["projected_sector_residual"] = None
        raw = add_differences(raw, ed_row)
        rows.append(raw)

        psi_p, weight = project_to_characters(group, chars, psi)
        if psi_p is None:
            print(f"[PAV] {sub.name}: target-sector weight ~ 0")
            continue
        prow = projected_row(
            summary, psi_p, weight, H, ed_gs, ed_row, group, chars, H.states
        )
        rows.append(prow)
        np.savez_compressed(
            sub / "projected_state.npz",
            psi_native=psi_p,
            native_states=H.states,
            symmetry_sector_weight=float(weight),
        )
        atomic_json(sub / "projected_summary.json", prow)
        print(
            f"[PAV] {sub.name:36s} w={weight:.6f} "
            f"E {float(summary['E']):.9f} -> {prow['E']:.9f}; "
            f"F {float(summary['fidelity_ground_subspace']):.5f} -> "
            f"{prow['fidelity_ground_subspace']:.5f}; "
            f"SLC-/SLC+ {prow['S_LC_opposite']/max(prow['S_LC_same'],1e-15):.3g}"
        )

    write_csv(root / "symmetry_projection_all.csv", rows)
    best = best_rows(rows)
    write_csv(root / "symmetry_projection_best.csv", best)
    atomic_json(root / "symmetry_projection_best.json", best)

    print("\n=== Symmetry-projected best comparison ===")
    for r in best:
        print(
            f"{r['method']:30s} seed={str(r.get('seed','')):11s} "
            f"E={float(r['E']):.12g} F={float(r['fidelity_ground_subspace']):.6f} "
            f"SCO={float(r['CO_qmax']):.5g} "
            f"SLC+={float(r['S_LC_same']):.5g} "
            f"SLC-={float(r['S_LC_opposite']):.5g} "
            f"w={r.get('symmetry_sector_weight')}"
        )
    print(f"Saved {root/'ed_symmetry_sector.json'}")
    print(f"Saved {root/'symmetry_projection_all.csv'}")
    print(f"Saved {root/'symmetry_projection_best.csv'}")


if __name__ == "__main__":
    main()
