#!/usr/bin/env python3
"""Compare ED, full PAV/VAP, and C3-only PAV/VAP on the same Hex18 problem.

The unprojected reference is always the same full complex Slater x density
Jastrow state. From each optimized raw state we construct:
  - PAV: projection to each ED-ground full symmetry sector
  - C3-PAV: projection only to each ED-ground C3 sector (momentum unfixed)

We also optimize directly after projection:
  - Full-VAP: full orbital matrix + Jastrow in the full target sector
  - C3-VAP: full orbital matrix + Jastrow with only C3 fixed

For a unique ED ground, the full group is ordinary T x C3v.
For a degenerate ED ground multiplet, the full comparison uses the abelian
ordinary T x C3 sectors present in the exact ground space.
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
from run_benchmark import (
    atomic_json,
    run_vmc_one,
    solve_ed,
)
from run_parametric import run_full_vap_one
from symmetry_projection import (
    build_ordinary_space_group,
    build_translation_c3_group,
    build_c3_group,
    characterize_unique_state,
    decompose_ground_manifold_abelian,
    project_to_characters,
    projected_sector_residual,
    commutator_error,
    sector_quantum_numbers,
)


METHOD_ORDER = [
    "ED",
    "PAV-Slater-Jastrow",
    "Full-VAP-Slater-Jastrow",
    "C3-PAV-Slater-Jastrow",
    "C3-VAP-Slater-Jastrow",
]


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def add_ed_differences(row, ed):
    row["delta_E_vs_ED"] = float(row["E"]) - float(ed["E"])
    row["relative_E_error"] = abs(row["delta_E_vs_ED"]) / max(abs(float(ed["E"])), 1e-15)
    for key in ("CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"):
        ref = float(ed[key]); val = float(row[key])
        row[f"delta_{key}_vs_ED"] = val-ref
        row[f"relative_{key}_error"] = abs(val-ref)/max(abs(ref), 1e-12)
    return row


def build_ed_sectors(g, H, ed_gs, tol):
    """Return full target sectors and C3-only target sectors."""
    if ed_gs.shape[1] == 1:
        full_group = build_ordinary_space_group(g, H.states)
        err, name = commutator_error(full_group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"full symmetry/H mismatch: {name} {err:.3e}")
        chars = characterize_unique_state(full_group, ed_gs[:, 0], tol=1e-8)
        full_sectors = [{
            "characters": chars, "basis": ed_gs,
            "dimension": 1, "residual": 0.0
        }]
        full_group_name = "ordinary translations x C3v"
    else:
        full_group = build_translation_c3_group(g, H.states)
        err, name = commutator_error(full_group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"T x C3/H mismatch: {name} {err:.3e}")
        full_sectors = decompose_ground_manifold_abelian(
            full_group, ed_gs, tol=max(1e-8, 10*tol)
        )
        full_group_name = "ordinary translations x C3"

    c3_group = build_c3_group(g, H.states)
    err, name = commutator_error(c3_group, H.operator, n_tests=2)
    if err > 1e-9:
        raise RuntimeError(f"C3/H mismatch: {name} {err:.3e}")
    if ed_gs.shape[1] == 1:
        chars = characterize_unique_state(c3_group, ed_gs[:, 0], tol=1e-8)
        c3_sectors = [{
            "characters": chars, "basis": ed_gs,
            "dimension": 1, "residual": 0.0
        }]
    else:
        c3_sectors = decompose_ground_manifold_abelian(
            c3_group, ed_gs, tol=max(1e-8, 10*tol)
        )

    for sec in full_sectors:
        sec["quantum_numbers"] = sector_quantum_numbers(
            g, full_group, sec["characters"]
        )
    for sec in c3_sectors:
        sec["quantum_numbers"] = sector_quantum_numbers(
            g, c3_group, sec["characters"]
        )
    return full_group, full_group_name, full_sectors, c3_group, c3_sectors


def sector_record(i, sec):
    qn = sec["quantum_numbers"]
    return {
        "sector_id": int(i),
        "dimension": int(sec["dimension"]),
        "residual": float(sec["residual"]),
        "k": qn["momentum"]["name"],
        "q": qn["momentum"]["q"],
        "c3": qn["c3"]["label"],
    }


def projected_row(
    *, method, seed, psi, H, ed_gs, ed_row, group, sec,
    sector_id, projection_name,
):
    t0 = time.perf_counter()
    psi_p, weight = project_to_characters(group, sec["characters"], psi)
    if psi_p is None:
        return None
    Hpsi = H.operator @ psi_p
    E = float(np.vdot(psi_p, Hpsi).real)
    var = float(max(0.0, np.vdot(Hpsi, Hpsi).real-E**2))
    obs = benchmark_observables(H.g, H.states, psi_p, native_observables)
    fid = float(np.sum(np.abs(ed_gs.conj().T @ psi_p)**2).real)
    residual = projected_sector_residual(group, sec["characters"], psi_p)
    qn = sec["quantum_numbers"]
    row = {
        "method": method,
        "seed": seed,
        "projection": projection_name,
        "target_sector_id": int(sector_id),
        "target_sector_ground_dimension": int(sec["dimension"]),
        "target_k": qn["momentum"]["name"],
        "target_q": json.dumps(qn["momentum"]["q"]),
        "target_c3": qn["c3"]["label"],
        "symmetry_sector_weight": float(weight),
        "projected_sector_residual": float(residual),
        "E": E,
        "E_per_site": E/H.g.Ns,
        "exact_energy_variance": var,
        "exact_energy_variance_per_site": var/H.g.Ns,
        "fidelity_ground_subspace": fid,
        "time_projection_postprocess_s": time.perf_counter()-t0,
        **obs.as_dict(),
    }
    return add_ed_differences(row, ed_row), psi_p


def load_summary_state(directory, H):
    summary_path = directory/"summary.json"
    state_path = directory/"final_state.npz"
    if not summary_path.exists() or not state_path.exists():
        return None, None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(state_path, allow_pickle=False) as z:
        psi = np.asarray(z["psi_native"], np.complex128)
        states = np.asarray(z["native_states"], np.uint64)
    if not np.array_equal(states, H.states):
        raise RuntimeError(f"basis mismatch in {state_path}")
    psi /= np.linalg.norm(psi)
    return summary, psi


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--nf", type=int, default=6)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--t1", type=float, default=0.5)
    p.add_argument("--t2", type=float, default=-0.5)
    p.add_argument("--V", type=float, default=2.0)
    p.add_argument(
        "--seeds", nargs="+",
        default=["normal", "sd", "isd", "lc_same", "lc_opposite"]
    )
    p.add_argument("--seed-amplitude", type=float, default=0.08)
    p.add_argument("--jastrow-shells", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--n-discard", type=int, default=16)
    p.add_argument("--n-chains", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--diag-shift", type=float, default=0.05)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--rng-seed", type=int, default=2026)
    p.add_argument("--ed-k", type=int, default=12)
    p.add_argument("--ed-degeneracy-tol", type=float, default=1e-8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-vap", action="store_true",
                   help="Only generate ED and PAV/C3-PAV from raw full SJ states")
    p.add_argument("--output", default="results/projection_compare_hex18")
    return p


def main():
    a = parser().parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    g = native_geometry.RubyPBC("ring3", t=a.t, t1=a.t1, t2=a.t2)
    H, evals, ed_gs, ed_row = solve_ed(
        g, a.nf, a.V, k=a.ed_k, degeneracy_tol=a.ed_degeneracy_tol
    )
    if ed_row.get("ground_manifold_truncated"):
        raise RuntimeError("ED ground manifold truncated; increase --ed-k")

    (
        full_group, full_group_name, full_sectors,
        c3_group, c3_sectors
    ) = build_ed_sectors(g, H, ed_gs, a.ed_degeneracy_tol)

    settings = vars(a).copy()
    settings.update({
        "cluster": "ring3",
        "Ns": g.Ns,
        "full_projection_group": full_group_name,
        "full_sector_count": len(full_sectors),
        "c3_sector_count": len(c3_sectors),
    })
    atomic_json(out/"settings.json", settings)
    atomic_json(out/"geometry.json", g.metadata())
    atomic_json(
        out/"ed_projection_sectors.json",
        {
            "ground_manifold_dimension": int(ed_gs.shape[1]),
            "full_group": full_group_name,
            "full_sectors": [sector_record(i,s) for i,s in enumerate(full_sectors)],
            "c3_group": "ordinary C3 only; momentum unfixed",
            "c3_sectors": [sector_record(i,s) for i,s in enumerate(c3_sectors)],
        },
    )

    ed = dict(ed_row)
    ed["method"] = "ED"
    ed["target_k"] = "ground manifold"
    ed["target_c3"] = "ground manifold"
    rows = [ed]

    print(
        f"[ED] E0={ed['E']:.12g}, ground_dim={ed_gs.shape[1]}, "
        f"full_sectors={len(full_sectors)}, C3_sectors={len(c3_sectors)}"
    )
    for i, sec in enumerate(full_sectors):
        qn = sec["quantum_numbers"]
        print(
            f"  [full sector {i}] k={qn['momentum']['name']} "
            f"q={qn['momentum']['q']} {qn['c3']['label']} "
            f"dim={sec['dimension']}"
        )
    for i, sec in enumerate(c3_sectors):
        qn = sec["quantum_numbers"]
        print(
            f"  [C3 sector {i}] k=unfixed {qn['c3']['label']} "
            f"dim={sec['dimension']}"
        )

    raw_states = {}
    run_id = 0

    # 1) Optimize the same unprojected full SJ reference for every seed.
    for seed in a.seeds:
        run_id += 1
        d = out/f"slater_jastrow__{seed}"
        summary, psi = load_summary_state(d, H) if a.resume else (None, None)
        if summary is None:
            print(f"\n[raw SJ] seed={seed}", flush=True)
            row = run_vmc_one(
                g, a.nf, a.V, H, ed_gs, float(evals[0]),
                ansatz="slater_jastrow",
                seed_name=seed,
                seed_amplitude=a.seed_amplitude,
                n_shells=a.jastrow_shells,
                n_samples=a.n_samples,
                n_discard=a.n_discard,
                n_chains=a.n_chains,
                n_iter=a.n_iter,
                learning_rate=a.learning_rate,
                diag_shift=a.diag_shift,
                rng_seed=a.rng_seed+1009*run_id,
                chunk_size=a.chunk_size,
                output_dir=out,
                validate_hamiltonian=(run_id==1),
            )
            summary, psi = load_summary_state(d, H)
        else:
            print(f"\n[resume raw SJ] seed={seed}", flush=True)
        raw_states[seed] = psi

        # 2) PAV into each full ED-ground sector.
        for sid, sec in enumerate(full_sectors):
            result = projected_row(
                method="PAV-Slater-Jastrow",
                seed=seed, psi=psi, H=H, ed_gs=ed_gs, ed_row=ed_row,
                group=full_group, sec=sec, sector_id=sid,
                projection_name=full_group_name,
            )
            if result is None:
                continue
            prow, ppsi = result
            rows.append(prow)
            dproj = out/f"pav__sector{sid}__{seed}"
            dproj.mkdir(parents=True, exist_ok=True)
            atomic_json(dproj/"summary.json", prow)
            np.savez_compressed(
                dproj/"final_state.npz",
                psi_native=ppsi, native_states=H.states
            )

        # 3) C3-only PAV.
        for sid, sec in enumerate(c3_sectors):
            result = projected_row(
                method="C3-PAV-Slater-Jastrow",
                seed=seed, psi=psi, H=H, ed_gs=ed_gs, ed_row=ed_row,
                group=c3_group, sec=sec, sector_id=sid,
                projection_name="ordinary C3 only; momentum unfixed",
            )
            if result is None:
                continue
            prow, ppsi = result
            rows.append(prow)
            dproj = out/f"c3_pav__sector{sid}__{seed}"
            dproj.mkdir(parents=True, exist_ok=True)
            atomic_json(dproj/"summary.json", prow)
            np.savez_compressed(
                dproj/"final_state.npz",
                psi_native=ppsi, native_states=H.states
            )

        write_csv(out/"projection_comparison_all.csv", rows)

    if not a.skip_vap:
        # 4) Full VAP and 5) C3-only VAP.
        for label, group, sectors, method, prefix, projection in [
            (
                "full VAP", full_group, full_sectors,
                "Full-VAP-Slater-Jastrow", "full_vap_sj", full_group_name
            ),
            (
                "C3 VAP", c3_group, c3_sectors,
                "C3-VAP-Slater-Jastrow", "c3_vap_sj",
                "ordinary C3 only; momentum unfixed"
            ),
        ]:
            for sid, sec in enumerate(sectors):
                qn = sec["quantum_numbers"]
                for seed in a.seeds:
                    run_id += 1
                    d = out/f"{prefix}__sector{sid}__{seed}"
                    summary, _ = load_summary_state(d, H) if a.resume else (None, None)
                    if summary is None:
                        print(
                            f"\n[{label}] seed={seed}, sector={sid}, "
                            f"k={qn['momentum']['name']}, {qn['c3']['label']}",
                            flush=True,
                        )
                        row = run_full_vap_one(
                            g, a.nf, a.V, H, ed_gs, float(evals[0]),
                            group, sec["characters"],
                            ground_sectors=sectors,
                            sector_id=sid,
                            sector_dim=int(sec["dimension"]),
                            method_name=method,
                            family_name=prefix,
                            dir_prefix=prefix,
                            projection_name=projection,
                            seed_name=seed,
                            seed_amplitude=a.seed_amplitude,
                            n_shells=a.jastrow_shells,
                            n_samples=a.n_samples,
                            n_discard=a.n_discard,
                            n_chains=a.n_chains,
                            n_iter=a.n_iter,
                            learning_rate=a.learning_rate,
                            diag_shift=a.diag_shift,
                            rng_seed=a.rng_seed+1009*run_id,
                            chunk_size=a.chunk_size,
                            output_dir=out,
                        )
                    else:
                        row = summary
                        print(
                            f"\n[resume {label}] seed={seed}, sector={sid}",
                            flush=True,
                        )
                    row["target_k"] = qn["momentum"]["name"]
                    row["target_q"] = json.dumps(qn["momentum"]["q"])
                    row["target_c3"] = qn["c3"]["label"]
                    rows.append(row)
                    write_csv(out/"projection_comparison_all.csv", rows)

    # Best-energy representative of each requested method.
    best = [ed]
    for method in METHOD_ORDER[1:]:
        cand = [
            r for r in rows
            if r.get("method") == method and r.get("E") is not None
        ]
        if cand:
            best.append(min(cand, key=lambda r: float(r["E"])))

    write_csv(out/"projection_comparison_best.csv", best)
    atomic_json(out/"projection_comparison_best.json", best)

    print("\n=== ED / PAV / Full-VAP / C3-PAV / C3-VAP ===")
    for r in best:
        print(
            f"{r['method']:28s} seed={str(r.get('seed','')):11s} "
            f"k={str(r.get('target_k','')):7s} "
            f"C3={str(r.get('target_c3','')):7s} "
            f"E={float(r['E']):.12g} "
            f"dE={float(r.get('delta_E_vs_ED',0.0)):.3e} "
            f"F={float(r.get('fidelity_ground_subspace',1.0)):.6f} "
            f"var={float(r.get('exact_energy_variance',0.0)):.4g} "
            f"SLC+={float(r['S_LC_same']):.5g} "
            f"SLC-={float(r['S_LC_opposite']):.5g}"
        )
    print("Saved", out/"projection_comparison_all.csv")
    print("Saved", out/"projection_comparison_best.csv")


if __name__ == "__main__":
    main()
