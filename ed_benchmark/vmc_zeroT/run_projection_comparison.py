#!/usr/bin/env python3
"""Hex18 benchmark: ED, raw SJ, spatial+TR PAV/VAP, and one Lanczos step.

The variational base is always the unrestricted full complex Slater matrix
times a real density Jastrow. The raw state is intentionally unprojected.
Finite-size benchmark states restore both:
  1) the exact ordinary spatial symmetry sector(s), and
  2) spinless time reversal T=K in the occupation basis.

For a complex spatial irrep, time reversal maps Gamma -> Gamma*. The benchmark
therefore treats Gamma + Gamma* as one TR target pair instead of incorrectly
forcing a T-invariant state to remain in only one member of the pair.

Methods:
  ED
  Raw-Slater-Jastrow
  PAV-TR-Slater-Jastrow
  PAV-TR-Slater-Jastrow+1LS
  Full-VAP-TR-Slater-Jastrow
  Full-VAP-TR-Slater-Jastrow+1LS
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
    commutator_error,
    sector_quantum_numbers,
)


METHOD_ORDER = [
    "ED",
    "Raw-Slater-Jastrow",
    "PAV-TR-Slater-Jastrow",
    "PAV-TR-Slater-Jastrow+1LS",
    "Full-VAP-TR-Slater-Jastrow",
    "Full-VAP-TR-Slater-Jastrow+1LS",
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


def build_tr_targets(sectors, tol=1e-7):
    """Pair each spatial character sector with its time-reversed conjugate."""
    targets = []
    used = set()
    for i, sec in enumerate(sectors):
        if i in used:
            continue
        chars = np.asarray(sec["characters"], complex)
        conj_chars = np.conj(chars)
        self_conjugate = np.max(np.abs(chars-conj_chars)) < tol
        partner = None
        partner_id = None
        if not self_conjugate:
            for j, other in enumerate(sectors):
                if j == i or j in used:
                    continue
                if np.max(np.abs(np.asarray(other["characters"], complex)-conj_chars)) < tol:
                    partner = other
                    partner_id = j
                    break
            if partner is None:
                raise RuntimeError(
                    f"Spatial ED sector {i} is not TR self-conjugate and its "
                    "conjugate partner is absent from the complete ground manifold"
                )

        used.add(i)
        if partner_id is not None:
            used.add(partner_id)

        q1 = sec["quantum_numbers"]
        if partner is None:
            klabel = q1["momentum"]["name"]
            qlabel = q1["momentum"]["q"]
            c3label = q1["c3"]["label"]
            dim = int(sec["dimension"])
        else:
            q2 = partner["quantum_numbers"]
            klabel = f"{q1['momentum']['name']}/{q2['momentum']['name']}"
            qlabel = [q1["momentum"]["q"], q2["momentum"]["q"]]
            c3label = f"{q1['c3']['label']}/{q2['c3']['label']}"
            dim = int(sec["dimension"]) + int(partner["dimension"])

        targets.append({
            "target_id": len(targets),
            "primary_sector_id": i,
            "partner_sector_id": partner_id,
            "primary": sec,
            "partner": partner,
            "characters": chars,
            "partner_characters": (
                None if partner is None
                else np.asarray(partner["characters"], complex)
            ),
            "dimension": dim,
            "k": klabel,
            "q": qlabel,
            "c3": c3label,
            "self_conjugate": bool(partner is None),
        })
    return targets


def target_record(target):
    return {
        "target_id": int(target["target_id"]),
        "primary_sector_id": int(target["primary_sector_id"]),
        "partner_sector_id": (
            None if target["partner_sector_id"] is None
            else int(target["partner_sector_id"])
        ),
        "dimension": int(target["dimension"]),
        "k": target["k"],
        "q": target["q"],
        "c3": target["c3"],
        "tr_self_conjugate": bool(target["self_conjugate"]),
    }


def reconstruct_projected_component(group, chars, psi):
    p, w = project_to_characters(group, chars, psi)
    if p is None:
        return np.zeros_like(np.asarray(psi, complex)), float(w)
    return np.sqrt(max(float(w), 0.0))*p, float(w)


def spatial_pair_project(group, target, psi, tol=1e-14):
    """Project into Gamma or Gamma+Gamma* before antiunitary TR restoration."""
    raw, w1 = reconstruct_projected_component(
        group, target["characters"], psi
    )
    w2 = 0.0
    if target["partner_characters"] is not None:
        p2, w2 = reconstruct_projected_component(
            group, target["partner_characters"], psi
        )
        raw = raw + p2
    nrm = float(np.linalg.norm(raw))
    weight = float(w1+w2)
    if nrm < tol:
        return None, weight
    return raw/nrm, weight


def time_reversal_residual(psi):
    """Gauge-invariant direct distance from a spinless T=K invariant ray."""
    x = np.asarray(psi, complex)
    x /= np.linalg.norm(x)
    s = np.dot(x, x)
    if abs(s) < 1e-15:
        return float(np.sqrt(2.0))
    phase = s/abs(s)
    return float(np.linalg.norm(x-phase*np.conj(x)))


def target_support_weight(group, target, psi):
    _, w1 = project_to_characters(group, target["characters"], psi)
    w = float(w1)
    if target["partner_characters"] is not None:
        _, w2 = project_to_characters(group, target["partner_characters"], psi)
        w += float(w2)
    return float(min(max(w, 0.0), 1.0))


def target_support_residual(group, target, psi):
    """Direct norm ||(1-P_target) psi||.

    Do not infer this as sqrt(1-weight): near unit weight that expression
    amplifies ordinary floating-point roundoff (for example 1-w ~ 1e-15
    becomes a spurious residual ~3e-8).
    """
    x = np.asarray(psi, complex)
    x /= np.linalg.norm(x)
    raw, _ = reconstruct_projected_component(
        group, target["characters"], x
    )
    if target["partner_characters"] is not None:
        raw2, _ = reconstruct_projected_component(
            group, target["partner_characters"], x
        )
        raw = raw + raw2
    return float(np.linalg.norm(x-raw))


def restore_time_reversal_pav(psi, H, tol=1e-13):
    """Canonical spinless-TR restoration for projection AFTER variation.

    PAV should not introduce a new energy minimization. For T=K, first fix the
    arbitrary global phase so x^T x is real and nonnegative, then take the
    normalized T-even representative x + x*. This is the closest T-invariant
    ray to the spatially projected parent in Hilbert-space norm.

    Full VAP-TR remains different: its TR phase is variational during training.
    """
    x = np.asarray(psi, complex)
    x /= np.linalg.norm(x)
    bilinear = np.dot(x, x)
    phase_angle = 0.0
    if abs(bilinear) > tol:
        phase_angle = -0.5*float(np.angle(bilinear))
    xg = np.exp(1j*phase_angle)*x
    y = xg + np.conj(xg)
    nrm = float(np.linalg.norm(y))
    if nrm < tol:
        # This should only occur at pathological numerical cancellation.
        # The quadrature shifted by pi/2 supplies the equivalent real ray.
        xg = 1j*xg
        y = xg + np.conj(xg)
        nrm = float(np.linalg.norm(y))
    if nrm < tol:
        raise RuntimeError("Canonical TR-PAV restoration produced zero norm")
    y = np.asarray(y/nrm, np.complex128)
    Ebefore = float(np.vdot(x, H.operator@x).real)
    Eafter = float(np.vdot(y, H.operator@y).real)
    return y, {
        "time_reversal_restored": True,
        "time_reversal_pav_phase": phase_angle,
        "time_reversal_residual": time_reversal_residual(y),
        "time_reversal_energy_before": Ebefore,
        "time_reversal_energy_after": Eafter,
        "time_reversal_delta_E": Eafter-Ebefore,
        "time_reversal_pav_definition": "canonical closest-ray x+T x; no energy minimization",
    }


def exact_state_row(
    *, method, seed, psi, H, ed_gs, ed_row,
    target_k="mixed", target_q=None, target_c3="mixed",
    projection="none", extra=None,
):
    psi = np.asarray(psi, np.complex128)
    psi /= np.linalg.norm(psi)
    Hpsi = H.operator @ psi
    E = float(np.vdot(psi, Hpsi).real)
    variance = float(max(0.0, np.vdot(Hpsi, Hpsi).real-E**2))
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
        "time_reversal_residual": time_reversal_residual(psi),
        **obs.as_dict(),
    }
    if extra:
        row.update(extra)
    return add_ed_differences(row, ed_row)


def one_lanczos_step(psi, H, *, tol=1e-13):
    """Lowest state in span{|psi>,H|psi>} using exact full-basis algebra."""
    psi = np.asarray(psi, np.complex128)
    psi /= np.linalg.norm(psi)
    Hpsi = H.operator @ psi
    E0 = float(np.vdot(psi, Hpsi).real)
    residual = Hpsi-E0*psi
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
    K = np.array([[E0, sigma], [sigma, h11]], dtype=float)
    evals, vecs = np.linalg.eigh(K)
    coeff = vecs[:, 0]
    out = coeff[0]*psi+coeff[1]*q1
    out /= np.linalg.norm(out)
    E1 = float(np.vdot(out, H.operator@out).real)
    denom = coeff[0]-coeff[1]*E0/sigma
    alpha = None if abs(denom) <= 1e-14 else float((coeff[1]/sigma)/denom)
    if E1 > E0+1e-10:
        raise RuntimeError(f"1LS energy increased: {E0:.12g} -> {E1:.12g}")
    return out, {
        "lanczos_improved": True,
        "lanczos_residual_norm_before": sigma,
        "lanczos_energy_before": E0,
        "lanczos_energy_after": E1,
        "lanczos_delta_E": E1-E0,
        "lanczos_krylov_dim": 2,
        "lanczos_alpha_H": alpha,
        "lanczos_krylov_h00": E0,
        "lanczos_krylov_h01": sigma,
        "lanczos_krylov_h11": h11,
    }


def lanczos_row(
    *, base_method, seed, psi, H, ed_gs, ed_row,
    target, group, projection, extra=None,
):
    out, info = one_lanczos_step(psi, H)
    support = target_support_weight(group, target, out)
    row = exact_state_row(
        method=base_method+"+1LS",
        seed=seed,
        psi=out,
        H=H,
        ed_gs=ed_gs,
        ed_row=ed_row,
        target_k=target["k"],
        target_q=target["q"],
        target_c3=target["c3"],
        projection=projection+" + one exact Lanczos step",
        extra={
            **(extra or {}),
            **info,
            "target_support_weight": support,
            "projected_sector_residual": target_support_residual(group, target, out),
            "time_reversal_restored": True,
        },
    )
    return row, out


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
    p.add_argument("--output", default="results/full_pav_vap_tr_hex18")
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
    tr_targets = build_tr_targets(sectors)

    jlabels, jdist, _, jcounts = jastrow_channels(
        g, mode=a.jastrow_mode, n_shells=a.jastrow_shells
    )
    atomic_json(out/"settings.json", {
        **vars(a),
        "cluster": "ring3",
        "Ns": g.Ns,
        "spatial_projection_group": group_name,
        "spatial_sector_count": len(sectors),
        "time_reversal_target_count": len(tr_targets),
        "spinless_time_reversal": "complex conjugation K",
        "jastrow_channel_count": len(jlabels),
    })
    atomic_json(out/"geometry.json", g.metadata())
    atomic_json(out/"jastrow_channels.json", {
        "mode": a.jastrow_mode,
        "n_channels": len(jlabels),
        "channels": [
            {
                "index": i, "label": str(label),
                "representative_distance": float(d),
                "pair_count": int(n),
            }
            for i,(label,d,n) in enumerate(zip(jlabels,jdist,jcounts))
        ],
    })
    atomic_json(out/"ed_projection_targets.json", {
        "ground_manifold_dimension": int(ed_gs.shape[1]),
        "spatial_group": group_name,
        "time_reversal": "spinless K",
        "targets": [target_record(t) for t in tr_targets],
    })

    print(f"[Jastrow] mode={a.jastrow_mode}, channels={len(jlabels)}")
    for i,(label,d,n) in enumerate(zip(jlabels,jdist,jcounts)):
        print(
            f"  v[{i}] {label:16s} pairs={int(n):3d} "
            f"representative_r={float(d):.8g}"
        )

    ed = dict(ed_row)
    ed["method"] = "ED"
    ed["target_k"] = "ground manifold"
    ed["target_c3"] = "ground manifold"
    ed["time_reversal_restored"] = True
    ed["time_reversal_residual"] = max(
        time_reversal_residual(ed_gs[:,i]) for i in range(ed_gs.shape[1])
    ) if ed_gs.shape[1] == 1 else None
    rows = [ed]

    print(
        f"[ED] E0={ed['E']:.12g}, ground_dim={ed_gs.shape[1]}, "
        f"spatial_sectors={len(sectors)}, TR_targets={len(tr_targets)}"
    )
    for t in tr_targets:
        print(
            f"  [TR target {t['target_id']}] k={t['k']} C3={t['c3']} "
            f"dim={t['dimension']} self_conjugate={t['self_conjugate']}"
        )

    run_id = 0
    proj_name = group_name + " + spinless TR"

    # A. Unrestricted raw SJ, then PAV with spatial + TR restoration.
    for seed in a.seeds:
        run_id += 1
        d = out/f"slater_jastrow__{seed}"
        summary, psi = (
            load_summary_state(d,H,expected_jastrow_mode=a.jastrow_mode)
            if a.resume else (None,None)
        )
        if summary is None:
            print(f"\n[full raw SJ] seed={seed}", flush=True)
            run_vmc_one(
                g,a.nf,a.V,H,ed_gs,float(evals[0]),
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
                jastrow_mode=a.jastrow_mode,
            )
            summary,psi = load_summary_state(
                d,H,expected_jastrow_mode=a.jastrow_mode
            )
        else:
            print(f"\n[resume full raw SJ] seed={seed}", flush=True)

        raw_row = exact_state_row(
            method="Raw-Slater-Jastrow", seed=seed, psi=psi,
            H=H, ed_gs=ed_gs, ed_row=ed_row,
            target_k="mixed", target_c3="mixed", projection="none",
            extra={
                "time_reversal_restored": False,
                "jastrow_mode": a.jastrow_mode,
                "jastrow_labels": ";".join(map(str,jlabels)),
                "jastrow_channel_counts": ";".join(str(int(x)) for x in jcounts),
                "n_parameters": summary.get("n_parameters"),
                "time_total_s": summary.get("time_total_s"),
                "mc_E": summary.get("mc_E"),
                "mc_E_error": summary.get("mc_E_error"),
            },
        )
        rows.append(raw_row)
        print(
            f"[RAW] seed={seed:11s} E={raw_row['E']:.10f} "
            f"F={raw_row['fidelity_ground_subspace']:.6f} "
            f"var={raw_row['exact_energy_variance']:.4g} "
            f"TRres={raw_row['time_reversal_residual']:.3e}",
            flush=True,
        )

        for target in tr_targets:
            pair, spatial_weight = spatial_pair_project(group,target,psi)
            if pair is None:
                print(
                    f"[PAV-TR skip] seed={seed}, target={target['target_id']}: "
                    "zero spatial target weight",
                    flush=True,
                )
                continue
            spatial_pretr = exact_state_row(
                method="__diagnostic_spatial_PAV__",
                seed=seed, psi=pair, H=H, ed_gs=ed_gs, ed_row=ed_row,
                target_k=target["k"], target_q=target["q"],
                target_c3=target["c3"], projection=group_name,
            )
            ppsi, trinfo = restore_time_reversal_pav(pair,H)
            support = target_support_weight(group,target,ppsi)
            row = exact_state_row(
                method="PAV-TR-Slater-Jastrow",
                seed=seed, psi=ppsi, H=H, ed_gs=ed_gs, ed_row=ed_row,
                target_k=target["k"], target_q=target["q"],
                target_c3=target["c3"], projection=proj_name,
                extra={
                    "target_id": target["target_id"],
                    "target_sector_ground_dimension": target["dimension"],
                    "target_spatial_weight_before_TR": spatial_weight,
                    "spatial_PAV_E_before_TR": spatial_pretr["E"],
                    "spatial_PAV_fidelity_before_TR": spatial_pretr["fidelity_ground_subspace"],
                    "spatial_PAV_variance_before_TR": spatial_pretr["exact_energy_variance"],
                    "spatial_PAV_J_same_before_TR": spatial_pretr["J_same_mean"],
                    "spatial_PAV_J_opposite_before_TR": spatial_pretr["J_opposite_mean"],
                    "target_support_weight": support,
                    "projected_sector_residual": target_support_residual(
                        group,target,ppsi
                    ),
                    "time_reversal_restored": True,
                    **trinfo,
                    "jastrow_mode": a.jastrow_mode,
                    "jastrow_labels": ";".join(map(str,jlabels)),
                    "jastrow_channel_counts": ";".join(str(int(x)) for x in jcounts),
                },
            )
            rows.append(row)
            pd = out/f"pav_tr__target{target['target_id']}__{seed}"
            pd.mkdir(parents=True,exist_ok=True)
            atomic_json(pd/"summary.json",row)
            np.savez_compressed(
                pd/"final_state.npz",psi_native=ppsi,native_states=H.states
            )
            print(
                f"[PAV-TR] seed={seed:11s} target={target['target_id']} "
                f"k={target['k']:7s} E={row['E']:.10f} "
                f"F={row['fidelity_ground_subspace']:.6f} "
                f"J+= {row['J_same_mean']:.3e} J-= {row['J_opposite_mean']:.3e} "
                f"TRres={row['time_reversal_residual']:.2e}",
                flush=True,
            )

            ls_row,ls_psi = lanczos_row(
                base_method="PAV-TR-Slater-Jastrow",
                seed=seed, psi=ppsi, H=H, ed_gs=ed_gs, ed_row=ed_row,
                target=target, group=group, projection=proj_name,
                extra={
                    "target_id": target["target_id"],
                    "target_sector_ground_dimension": target["dimension"],
                    "jastrow_mode": a.jastrow_mode,
                    "jastrow_labels": ";".join(map(str,jlabels)),
                    "jastrow_channel_counts": ";".join(str(int(x)) for x in jcounts),
                },
            )
            rows.append(ls_row)
            lsd = out/f"pav_tr_1ls__target{target['target_id']}__{seed}"
            lsd.mkdir(parents=True,exist_ok=True)
            atomic_json(lsd/"summary.json",ls_row)
            np.savez_compressed(
                lsd/"final_state.npz",psi_native=ls_psi,native_states=H.states
            )
        write_csv(out/"comparison_all.csv",rows)

    # B. Full VAP with spatial-pair projection and TR restoration inside model.
    if not a.skip_vap:
        for target in tr_targets:
            for seed in a.seeds:
                run_id += 1
                tid = target["target_id"]
                d = out/f"full_vap_tr_sj__sector{tid}__{seed}"
                summary,_ = (
                    load_summary_state(d,H,expected_jastrow_mode=a.jastrow_mode)
                    if a.resume else (None,None)
                )
                if summary is None:
                    print(
                        f"\n[Full VAP-TR] seed={seed}, target={tid}, "
                        f"k={target['k']} C3={target['c3']}",
                        flush=True,
                    )
                    row = run_full_vap_one(
                        g,a.nf,a.V,H,ed_gs,float(evals[0]),
                        group,target["characters"],
                        ground_sectors=sectors,
                        sector_id=tid,
                        sector_dim=int(target["dimension"]),
                        method_name="Full-VAP-TR-Slater-Jastrow",
                        family_name="full_vap_tr_sj",
                        dir_prefix="full_vap_tr_sj",
                        projection_name=proj_name,
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
                        partner_chars=target["partner_characters"],
                        time_reversal_even=True,
                    )
                else:
                    row = summary
                    print(f"\n[resume Full VAP-TR] seed={seed}, target={tid}",flush=True)

                row["target_id"] = tid
                row["target_k"] = target["k"]
                row["target_q"] = json.dumps(target["q"])
                row["target_c3"] = target["c3"]
                row["time_reversal_restored"] = True
                rows.append(row)

                _,vap_psi = load_summary_state(
                    d,H,expected_jastrow_mode=a.jastrow_mode
                )
                if vap_psi is None:
                    raise RuntimeError(f"Missing VAP-TR state: {d}")
                # Recompute uniform target/TR diagnostics for the comparison.
                row["target_support_weight"] = target_support_weight(
                    group,target,vap_psi
                )
                row["projected_sector_residual"] = target_support_residual(
                    group,target,vap_psi
                )
                row["time_reversal_residual"] = time_reversal_residual(vap_psi)

                ls_row,ls_psi = lanczos_row(
                    base_method="Full-VAP-TR-Slater-Jastrow",
                    seed=seed, psi=vap_psi, H=H, ed_gs=ed_gs, ed_row=ed_row,
                    target=target, group=group, projection=proj_name,
                    extra={
                        "target_id": tid,
                        "target_sector_ground_dimension": target["dimension"],
                        "jastrow_mode": a.jastrow_mode,
                        "jastrow_labels": ";".join(map(str,jlabels)),
                        "jastrow_channel_counts": ";".join(str(int(x)) for x in jcounts),
                    },
                )
                rows.append(ls_row)
                lsd = out/f"full_vap_tr_1ls__target{tid}__{seed}"
                lsd.mkdir(parents=True,exist_ok=True)
                atomic_json(lsd/"summary.json",ls_row)
                np.savez_compressed(
                    lsd/"final_state.npz",psi_native=ls_psi,native_states=H.states
                )
                print(
                    f"[Full VAP-TR+1LS] seed={seed:11s} target={tid} "
                    f"E={ls_row['E']:.10f} F={ls_row['fidelity_ground_subspace']:.6f} "
                    f"J+= {ls_row['J_same_mean']:.3e} J-= {ls_row['J_opposite_mean']:.3e} "
                    f"TRres={ls_row['time_reversal_residual']:.2e}",
                    flush=True,
                )
                write_csv(out/"comparison_all.csv",rows)

    best = [ed]
    for method in METHOD_ORDER[1:]:
        cand = [r for r in rows if r.get("method")==method and r.get("E") is not None]
        if cand:
            best.append(min(cand,key=lambda r:float(r["E"])))

    write_csv(out/"comparison_best.csv",best)
    atomic_json(out/"comparison_best.json",best)

    print("\n=== ED / Raw SJ / PAV-TR / PAV-TR+1LS / Full VAP-TR / Full VAP-TR+1LS ===")
    for r in best:
        print(
            f"{r['method']:31s} seed={str(r.get('seed','')):11s} "
            f"k={str(r.get('target_k','')):9s} "
            f"C3={str(r.get('target_c3','')):13s} "
            f"E={float(r['E']):.12g} "
            f"dE={float(r.get('delta_E_vs_ED',0.0)):.3e} "
            f"F={float(r.get('fidelity_ground_subspace',1.0)):.6f} "
            f"var={float(r.get('exact_energy_variance',0.0)):.4g} "
            f"J+={float(r.get('J_same_mean',0.0)):.3e} "
            f"J-={float(r.get('J_opposite_mean',0.0)):.3e} "
            f"TRres={float(r.get('time_reversal_residual',0.0) or 0.0):.2e} "
            f"SLC+={float(r['S_LC_same']):.5g} "
            f"SLC-={float(r['S_LC_opposite']):.5g}"
        )
    print("Saved",out/"ed_projection_targets.json")
    print("Saved",out/"comparison_all.csv")
    print("Saved",out/"comparison_best.csv")


if __name__=="__main__":
    main()
