#!/usr/bin/env python3
"""Compare ED, full PAV, and full VAP on the same Hex18 problem.

The variational family is always the unrestricted full complex Slater matrix
times a real density Jastrow. No C3 constraint is imposed on the reference
state.

By default the Jastrow is bond-aware:
  v_t, v_t1, v_t2
plus a requested number of non-graph-pair distance shells. Thus physically
different Ruby bonds are not forced to share the same density-correlation
parameter merely because they have the same Euclidean distance.

Methods:
  ED
  Raw-Slater-Jastrow
      optimized unrestricted full complex Slater x Jastrow before projection.
  PAV-Slater-Jastrow
      project the optimized raw state afterward into an exact ED ground sector.
  PAV-Slater-Jastrow+1LS
      one exact Lanczos/Krylov step applied to the PAV state.
  Full-VAP-Slater-Jastrow
      optimize the unrestricted Slater x Jastrow directly after projection.
  Full-VAP-Slater-Jastrow+1LS
      one exact Lanczos/Krylov step applied to the optimized VAP state.

For a unique ED ground state the full target group is ordinary T x C3v.
For a degenerate ED ground manifold, target sectors are the ordinary abelian
T x C3 sectors present in the exact ground space.
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
from vmc_ansatz import jastrow_channels
from run_benchmark import atomic_json, run_vmc_one, solve_ed
from run_parametric import run_full_vap_one
from symmetry_projection import (
    build_ordinary_space_group,
    build_translation_c3_group,
    characterize_unique_state,
    decompose_ground_manifold_abelian,
    project_to_characters,
    projected_sector_residual,
    commutator_error,
    sector_quantum_numbers,
)


METHOD_ORDER = [
    "ED",
    "Raw-Slater-Jastrow",
    "PAV-Slater-Jastrow",
    "PAV-Slater-Jastrow+1LS",
    "Full-VAP-Slater-Jastrow",
    "Full-VAP-Slater-Jastrow+1LS",
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
    if ed_gs.shape[1] == 1:
        group = build_ordinary_space_group(g, H.states)
        err, name = commutator_error(group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"full symmetry/H mismatch: {name} {err:.3e}")
        chars = characterize_unique_state(group, ed_gs[:, 0], tol=1e-8)
        sectors = [{
            "characters": chars,
            "basis": ed_gs,
            "dimension": 1,
            "residual": 0.0,
        }]
        group_name = "ordinary translations x C3v"
    else:
        group = build_translation_c3_group(g, H.states)
        err, name = commutator_error(group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"T x C3/H mismatch: {name} {err:.3e}")
        sectors = decompose_ground_manifold_abelian(
            group, ed_gs, tol=max(1e-8, 10*tol)
        )
        group_name = "ordinary translations x C3"

    for sec in sectors:
        sec["quantum_numbers"] = sector_quantum_numbers(
            g, group, sec["characters"]
        )
    return group, group_name, sectors


def sector_record(i, sec):
    qn = sec["quantum_numbers"]
    return {
        "sector_id": int(i),
        "dimension": int(sec["dimension"]),
        "residual": float(sec["residual"]),
        "k": qn["momentum"]["name"],
        "q": qn["momentum"]["q"],
        "c3": qn["c3"]["label"],
        "c3_m": qn["c3"]["m"],
    }


def projected_row(
    *, seed, psi, H, ed_gs, ed_row, group, sec, sector_id, projection_name,
):
    psi_p, weight = project_to_characters(group, sec["characters"], psi)
    if psi_p is None:
        return None

    t0 = time.perf_counter()
    Hpsi = H.operator @ psi_p
    E = float(np.vdot(psi_p, Hpsi).real)
    variance = float(max(0.0, np.vdot(Hpsi, Hpsi).real-E**2))
    obs = benchmark_observables(H.g, H.states, psi_p, native_observables)
    fidelity = float(np.sum(np.abs(ed_gs.conj().T @ psi_p)**2).real)
    residual = projected_sector_residual(group, sec["characters"], psi_p)
    qn = sec["quantum_numbers"]

    row = {
        "method": "PAV-Slater-Jastrow",
        "seed": seed,
        "projection": projection_name,
        "target_sector_id": int(sector_id),
        "target_sector_ground_dimension": int(sec["dimension"]),
        "target_k": qn["momentum"]["name"],
        "target_q": json.dumps(qn["momentum"]["q"]),
        "target_c3": qn["c3"]["label"],
        "target_c3_m": qn["c3"]["m"],
        "symmetry_sector_weight": float(weight),
        "projected_sector_residual": float(residual),
        "E": E,
        "E_per_site": E/H.g.Ns,
        "exact_energy_variance": variance,
        "exact_energy_variance_per_site": variance/H.g.Ns,
        "fidelity_ground_subspace": fidelity,
        "time_projection_postprocess_s": time.perf_counter()-t0,
        **obs.as_dict(),
    }
    return add_ed_differences(row, ed_row), psi_p



def exact_state_row(
    *, method, seed, psi, H, ed_gs, ed_row,
    target_k="mixed", target_q=None, target_c3="mixed",
    projection="none", extra=None,
):
    """Exact full-basis diagnostics for any normalized Hex18 state."""
    psi = np.asarray(psi, np.complex128)
    psi = psi / np.linalg.norm(psi)
    Hpsi = H.operator @ psi
    E = float(np.vdot(psi, Hpsi).real)
    variance = float(max(0.0, np.vdot(Hpsi, Hpsi).real - E**2))
    obs = benchmark_observables(H.g, H.states, psi, native_observables)
    fidelity = float(np.sum(np.abs(ed_gs.conj().T @ psi)**2).real)
    row = {
        "method": method,
        "seed": seed,
        "projection": projection,
        "target_k": target_k,
        "target_q": None if target_q is None else json.dumps(target_q),
        "target_c3": target_c3,
        "E": E,
        "E_per_site": E/H.g.Ns,
        "exact_energy_variance": variance,
        "exact_energy_variance_per_site": variance/H.g.Ns,
        "fidelity_ground_subspace": fidelity,
        **obs.as_dict(),
    }
    if extra:
        row.update(extra)
    return add_ed_differences(row, ed_row)


def one_lanczos_step(psi, H, *, tol=1e-13):
    """Lowest state in span{|psi>, H|psi>} using exact full-basis algebra.

    Uses the normalized residual r=(H-E)|psi>, which is orthogonal to psi.
    This is numerically equivalent to optimizing (a+b H)|psi> but avoids a
    generalized-eigenproblem conditioning issue when psi is already close to
    an eigenstate.
    """
    psi = np.asarray(psi, np.complex128)
    psi = psi / np.linalg.norm(psi)
    Hpsi = H.operator @ psi
    E0 = float(np.vdot(psi, Hpsi).real)
    residual = Hpsi - E0*psi
    sigma = float(np.linalg.norm(residual))

    if sigma < tol:
        return psi.copy(), {
            "lanczos_improved": False,
            "lanczos_residual_norm_before": sigma,
            "lanczos_energy_before": E0,
            "lanczos_energy_after": E0,
            "lanczos_delta_E": 0.0,
            "lanczos_krylov_dim": 1,
            "lanczos_alpha_H": 0.0,
        }

    q1 = residual/sigma
    Hq1 = H.operator @ q1
    h11 = float(np.vdot(q1, Hq1).real)
    K = np.array([[E0, sigma], [sigma, h11]], dtype=np.float64)
    evals, vecs = np.linalg.eigh(K)
    coeff = vecs[:, 0]
    psi_ls = coeff[0]*psi + coeff[1]*q1
    psi_ls /= np.linalg.norm(psi_ls)
    Els = float(np.vdot(psi_ls, H.operator @ psi_ls).real)

    # Express the optimized vector as A*(1+alpha_H H)|psi> when possible.
    denom = coeff[0] - coeff[1]*E0/sigma
    alpha_H = None
    if abs(denom) > 1e-14:
        alpha_H = float(np.real((coeff[1]/sigma)/denom))

    if Els > E0 + 1e-10:
        raise RuntimeError(
            f"One-Lanczos variational energy increased: {E0:.12g} -> {Els:.12g}"
        )
    return psi_ls, {
        "lanczos_improved": True,
        "lanczos_residual_norm_before": sigma,
        "lanczos_energy_before": E0,
        "lanczos_energy_after": Els,
        "lanczos_delta_E": Els-E0,
        "lanczos_krylov_dim": 2,
        "lanczos_alpha_H": alpha_H,
        "lanczos_krylov_h00": E0,
        "lanczos_krylov_h01": sigma,
        "lanczos_krylov_h11": h11,
    }


def lanczos_row(
    *, base_method, seed, psi, H, ed_gs, ed_row,
    target_k, target_q, target_c3, projection,
    group=None, characters=None, extra=None,
):
    psi_ls, info = one_lanczos_step(psi, H)
    row = exact_state_row(
        method=base_method + "+1LS",
        seed=seed,
        psi=psi_ls,
        H=H,
        ed_gs=ed_gs,
        ed_row=ed_row,
        target_k=target_k,
        target_q=target_q,
        target_c3=target_c3,
        projection=projection + " + one exact Lanczos step",
        extra={**(extra or {}), **info},
    )
    if group is not None and characters is not None:
        row["projected_sector_residual"] = float(
            projected_sector_residual(group, characters, psi_ls)
        )
    return row, psi_ls


def load_summary_state(directory, H, *, expected_jastrow_mode):
    sf = directory/"summary.json"
    wf = directory/"final_state.npz"
    if not sf.exists() or not wf.exists():
        return None, None
    summary = json.loads(sf.read_text(encoding="utf-8"))
    if summary.get("jastrow_mode", "distance") != expected_jastrow_mode:
        return None, None
    with np.load(wf, allow_pickle=False) as z:
        psi = np.asarray(z["psi_native"], np.complex128)
        states = np.asarray(z["native_states"], np.uint64)
    if not np.array_equal(states, H.states):
        raise RuntimeError(f"basis mismatch in {wf}")
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
        default=["normal", "sd", "isd", "lc_same", "lc_opposite"],
    )
    p.add_argument("--seed-amplitude", type=float, default=0.08)
    p.add_argument(
        "--jastrow-mode", choices=["bond_aware", "distance"], default="bond_aware"
    )
    p.add_argument(
        "--jastrow-shells", type=int, default=4,
        help=(
            "For bond_aware: number of additional NON-BOND distance shells "
            "after v_t,v_t1,v_t2. For distance: total number of distance shells."
        ),
    )
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
    p.add_argument("--skip-vap", action="store_true")
    p.add_argument("--output", default="results/full_pav_vap_hex18")
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

    group, group_name, sectors = build_ed_sectors(
        g, H, ed_gs, a.ed_degeneracy_tol
    )

    jlabels, jdist, _, jcounts = jastrow_channels(
        g, mode=a.jastrow_mode, n_shells=a.jastrow_shells
    )
    jmeta = {
        "mode": a.jastrow_mode,
        "n_channels": len(jlabels),
        "channels": [
            {
                "index": i,
                "label": str(label),
                "representative_distance": float(d),
                "pair_count": int(n),
            }
            for i, (label, d, n) in enumerate(zip(jlabels, jdist, jcounts))
        ],
    }

    atomic_json(out/"settings.json", {
        **vars(a),
        "cluster": "ring3",
        "Ns": g.Ns,
        "projection_group": group_name,
        "sector_count": len(sectors),
        "jastrow_channel_count": len(jlabels),
    })
    atomic_json(out/"geometry.json", g.metadata())
    atomic_json(out/"jastrow_channels.json", jmeta)
    atomic_json(out/"ed_projection_sectors.json", {
        "ground_manifold_dimension": int(ed_gs.shape[1]),
        "group": group_name,
        "sectors": [sector_record(i, s) for i, s in enumerate(sectors)],
    })

    print(
        f"[Jastrow] mode={a.jastrow_mode}, channels={len(jlabels)}"
    )
    for i, (label, d, n) in enumerate(zip(jlabels, jdist, jcounts)):
        print(
            f"  v[{i}] {label:16s} pairs={int(n):3d} "
            f"representative_r={float(d):.8g}"
        )

    ed = dict(ed_row)
    ed["method"] = "ED"
    ed["target_k"] = "ground manifold"
    ed["target_c3"] = "ground manifold"
    rows = [ed]

    print(
        f"[ED] E0={ed['E']:.12g}, ground_dim={ed_gs.shape[1]}, "
        f"sectors={len(sectors)}"
    )
    for i, sec in enumerate(sectors):
        qn = sec["quantum_numbers"]
        print(
            f"  [ED sector {i}] k={qn['momentum']['name']} "
            f"q={qn['momentum']['q']} {qn['c3']['label']} "
            f"dim={sec['dimension']} residual={sec['residual']:.2e}"
        )

    run_id = 0

    # A. Same unrestricted full raw SJ state -> PAV.
    for seed in a.seeds:
        run_id += 1
        d = out/f"slater_jastrow__{seed}"
        summary, psi = (
            load_summary_state(
                d, H, expected_jastrow_mode=a.jastrow_mode
            ) if a.resume else (None, None)
        )
        if summary is None:
            print(f"\n[full raw SJ] seed={seed}", flush=True)
            run_vmc_one(
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
                validate_hamiltonian=(run_id == 1),
                jastrow_mode=a.jastrow_mode,
            )
            summary, psi = load_summary_state(
                d, H, expected_jastrow_mode=a.jastrow_mode
            )
        else:
            print(f"\n[resume full raw SJ] seed={seed}", flush=True)

        for sid, sec in enumerate(sectors):
            result = projected_row(
                seed=seed, psi=psi, H=H, ed_gs=ed_gs, ed_row=ed_row,
                group=group, sec=sec, sector_id=sid,
                projection_name=group_name,
            )
            if result is None:
                print(
                    f"[PAV skip] seed={seed}, sector={sid}: zero target weight",
                    flush=True,
                )
                continue
            row, ppsi = result
            row["jastrow_mode"] = a.jastrow_mode
            row["jastrow_labels"] = ";".join(map(str, jlabels))
            row["jastrow_channel_counts"] = ";".join(
                str(int(x)) for x in jcounts
            )
            rows.append(row)
            pd = out/f"pav__sector{sid}__{seed}"
            pd.mkdir(parents=True, exist_ok=True)
            atomic_json(pd/"summary.json", row)
            np.savez_compressed(
                pd/"final_state.npz",
                psi_native=ppsi,
                native_states=H.states,
                jastrow_labels=np.asarray(jlabels),
            )
            print(
                f"[PAV] seed={seed:11s} sector={sid} "
                f"k={row['target_k']:6s} w={row['symmetry_sector_weight']:.6f} "
                f"E={row['E']:.10f} F={row['fidelity_ground_subspace']:.6f} "
                f"var={row['exact_energy_variance']:.4g}",
                flush=True,
            )
        write_csv(out/"comparison_all.csv", rows)

    # B. Full VAP with exactly the same Jastrow channels.
    if not a.skip_vap:
        for sid, sec in enumerate(sectors):
            qn = sec["quantum_numbers"]
            for seed in a.seeds:
                run_id += 1
                d = out/f"full_vap_sj__sector{sid}__{seed}"
                summary, _ = (
                    load_summary_state(
                        d, H, expected_jastrow_mode=a.jastrow_mode
                    ) if a.resume else (None, None)
                )
                if summary is None:
                    print(
                        f"\n[Full VAP] seed={seed}, sector={sid}, "
                        f"k={qn['momentum']['name']} {qn['c3']['label']}",
                        flush=True,
                    )
                    row = run_full_vap_one(
                        g, a.nf, a.V, H, ed_gs, float(evals[0]),
                        group, sec["characters"],
                        ground_sectors=sectors,
                        sector_id=sid,
                        sector_dim=int(sec["dimension"]),
                        method_name="Full-VAP-Slater-Jastrow",
                        family_name="full_vap_sj",
                        dir_prefix="full_vap_sj",
                        projection_name=group_name,
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
                        jastrow_mode=a.jastrow_mode,
                    )
                else:
                    row = summary
                    print(
                        f"\n[resume Full VAP] seed={seed}, sector={sid}",
                        flush=True,
                    )
                row["target_k"] = qn["momentum"]["name"]
                row["target_q"] = json.dumps(qn["momentum"]["q"])
                row["target_c3"] = qn["c3"]["label"]
                row["target_c3_m"] = qn["c3"]["m"]
                rows.append(row)
                write_csv(out/"comparison_all.csv", rows)

    best = [ed]
    for method in METHOD_ORDER[1:]:
        cand = [
            r for r in rows
            if r.get("method") == method and r.get("E") is not None
        ]
        if cand:
            best.append(min(cand, key=lambda r: float(r["E"])))

    write_csv(out/"comparison_best.csv", best)
    atomic_json(out/"comparison_best.json", best)

    print("\n=== ED / Raw SJ / PAV / PAV+1LS / Full VAP / Full VAP+1LS ===")
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
    print("Saved", out/"jastrow_channels.json")
    print("Saved", out/"comparison_all.csv")
    print("Saved", out/"comparison_best.csv")


if __name__ == "__main__":
    main()
