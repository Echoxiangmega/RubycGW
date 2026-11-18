#!/usr/bin/env python3
"""Hex18 benchmark with a symmetry-tied Varma-style one-body parameterization.

Runs three reduced-parameter variational families on the same finite torus:
  1) Param-Slater
  2) Param-Slater-Jastrow
  3) Param-VAP-Slater-Jastrow

The one-body Hamiltonian contains only translation+C3 independent onsite and
complex bond corrections. Mirrors and time reversal are left free. Final
energies/observables are evaluated exactly on the full fixed-N basis.
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

from benchmark_observables import benchmark_observables, ruby_star_formfactor
from vmc_ansatz import minimum_image_shells, one_body_seed_hamiltonian
from parametric_ansatz import (
    build_symmetry_tied_basis,
    seed_tied_parameters,
    reconstruct_one_body,
    build_parametric_netket_model,
)
from run_benchmark import (
    atomic_json,
    make_netket_hamiltonian,
    reorder_netket_vector_to_native,
    solve_ed,
    stats_error,
)
from run_vap import initial_base_state_native
from symmetry_projection import (
    build_ordinary_space_group,
    build_translation_c3_group,
    characterize_unique_state,
    decompose_ground_manifold_abelian,
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


def add_ed_differences(row, ed):
    row["delta_E_vs_ED"] = float(row["E"]) - float(ed["E"])
    row["relative_E_error"] = abs(row["delta_E_vs_ED"]) / max(abs(float(ed["E"])), 1e-15)
    for key in ("CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"):
        ref = float(ed[key]); val = float(row[key])
        row[f"delta_{key}_vs_ED"] = val - ref
        row[f"relative_{key}_error"] = abs(val-ref) / max(abs(ref), 1e-12)
    return row


def exact_state_from_tied_params(basis, params, native_H):
    h = reconstruct_one_body(basis, params)
    _, u = np.linalg.eigh(h)
    M = np.asarray(u[:, :native_H.nf], complex)
    return initial_base_state_native(M, native_H.states, native_H.g.Ns, native_H.nf)


def target_sector_weight(basis, params, native_H, group, chars):
    psi = exact_state_from_tied_params(basis, params, native_H)
    _, w = project_to_characters(group, chars, psi)
    return float(w)


def union_sector_weight(basis, params, native_H, group, sectors):
    """Total weight in the direct sum of all ED-ground symmetry sectors."""
    psi = exact_state_from_tied_params(basis, params, native_H)
    total = 0.0
    for sec in sectors:
        _, w = project_to_characters(group, sec["characters"], psi)
        total += float(w)
    return float(min(max(total, 0.0), 1.0))


def state_union_sector_weight(psi, group, sectors):
    total = 0.0
    for sec in sectors:
        _, w = project_to_characters(group, sec["characters"], psi)
        total += float(w)
    return float(min(max(total, 0.0), 1.0))


def make_projectable_tied_params(
    basis, params, native_H, group, chars, *, rng_seed,
    min_weight=1e-8, initial_jitter=1e-4, max_tries=8,
):
    p = {k: np.array(v, dtype=float, copy=True) for k, v in params.items()}
    w0 = target_sector_weight(basis, p, native_H, group, chars)
    if w0 >= min_weight:
        return p, w0, 0.0

    rng = np.random.default_rng(int(rng_seed))
    scale = float(initial_jitter)
    for _ in range(max_tries):
        q = {k: np.array(v, copy=True) for k, v in p.items()}
        for key in ("onsite", "bond_re", "bond_im"):
            if q[key].size:
                q[key] += scale * rng.normal(size=q[key].shape)
        w = target_sector_weight(basis, q, native_H, group, chars)
        if w >= min_weight:
            return q, float(w), scale
        scale *= 10.0
    raise RuntimeError(
        f"Could not make tied seed projectable: initial w={w0:.3e}, "
        f"last jitter={scale/10:.3e}"
    )


def run_one(
    g, nf, V, native_H, ed_gs, ed_E0, group, chars, basis, *,
    ground_sectors, sector_id=None, sector_dim=None,
    family, seed_name, seed_amplitude, n_shells, n_samples, n_discard,
    n_chains, n_iter, learning_rate, diag_shift, rng_seed, chunk_size,
    output_dir,
):
    import jax
    jax.config.update("jax_enable_x64", True)
    import netket as nk

    projected = family == "param_vap_sj"
    use_jastrow = family in ("param_sj", "param_vap_sj")

    t0 = time.perf_counter()
    hi, _, H_fast, graph = make_netket_hamiltonian(g, nf, V)
    star = ruby_star_formfactor(g)
    distances, shells, shell_counts = minimum_image_shells(g, n_shells=n_shells)
    h_seed = one_body_seed_hamiltonian(g, seed_name, seed_amplitude, star)
    init_params = seed_tied_parameters(basis, h_seed)

    if projected:
        seed_weight = target_sector_weight(basis, init_params, native_H, group, chars)
    else:
        seed_weight = union_sector_weight(
            basis, init_params, native_H, group, ground_sectors
        )
    seed_jitter = 0.0
    if projected and seed_weight < 1e-8:
        init_params, seed_weight, seed_jitter = make_projectable_tied_params(
            basis, init_params, native_H, group, chars,
            rng_seed=rng_seed + 7919,
            min_weight=1e-8,
            initial_jitter=1e-4,
        )
        print(
            f"[param init] {seed_name}: target sector {sector_id} weight was negligible; "
            f"jitter={seed_jitter:.1e}, new w={seed_weight:.3e}",
            flush=True,
        )
    elif projected:
        print(
            f"[param init] {seed_name}: target sector {sector_id} w={seed_weight:.6f}",
            flush=True,
        )
    else:
        print(
            f"[param init] {seed_name}: total ED-ground-sector weight={seed_weight:.6f}",
            flush=True,
        )

    maps = np.asarray([op.site_map for op in group], np.int32) if projected else None
    pchars = np.asarray(chars, complex) if projected else None
    model = build_parametric_netket_model(
        basis,
        nf,
        init_params,
        shells,
        use_jastrow,
        projection_site_maps=maps,
        projection_characters=pchars,
    )

    sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph, n_chains=n_chains)
    vstate = nk.vqs.MCState(
        sampler, model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        chunk_size=chunk_size,
        seed=rng_seed,
        sampler_seed=rng_seed + 104729,
    )
    setup_s = time.perf_counter() - t0

    tw = time.perf_counter()
    warm_stats = vstate.expect(H_fast)
    warmup_s = time.perf_counter() - tw

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    sr = nk.optimizer.SR(
        qgt=nk.optimizer.qgt.QGTJacobianDense,
        solver=nk.optimizer.solver.cholesky_with_fallback,
        diag_shift=diag_shift,
    )
    driver = nk.driver.VMC(
        H_fast,
        optimizer,
        variational_state=vstate,
        preconditioner=sr,
    )

    print(
        f"[optimizer] parameters={vstate.n_parameters}, "
        f"one_body={basis.n_one_body_parameters}, jastrow={len(distances) if use_jastrow else 0}",
        flush=True,
    )
    log = nk.logging.RuntimeLog()
    topt = time.perf_counter()
    driver.run(n_iter=n_iter, out=log, show_progress=True)
    opt_s = time.perf_counter() - topt

    te = time.perf_counter()
    vstate.sample()
    final_stats = vstate.expect(H_fast)
    mc_eval_s = time.perf_counter() - te

    tp = time.perf_counter()
    psi_nk = np.asarray(vstate.to_array(normalize=True))
    psi = reorder_netket_vector_to_native(hi, psi_nk, native_H.states, g.Ns)
    E_exact = float(np.vdot(psi, native_H.operator @ psi).real)
    obs = benchmark_observables(g, native_H.states, psi, native_observables)
    fidelity = float(np.sum(np.abs(ed_gs.conj().T @ psi) ** 2).real)
    if projected:
        _, final_sector_weight = project_to_characters(group, chars, psi)
        sector_residual = projected_sector_residual(group, chars, psi)
    else:
        final_sector_weight = state_union_sector_weight(
            psi, group, ground_sectors
        )
        sector_residual = None
    post_s = time.perf_counter() - tp

    method = {
        "param_slater": "Param-Slater",
        "param_sj": "Param-Slater-Jastrow",
        "param_vap_sj": "Param-VAP-Slater-Jastrow",
    }[family]
    suffix = f"__sector{sector_id}" if projected and sector_id is not None else ""
    d = output_dir / f"{family}{suffix}__{seed_name}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "final_state.npz",
        psi_native=psi,
        native_states=native_H.states,
        shell_distances=distances,
        shell_counts=shell_counts,
    )
    history = {}
    try:
        en = log.data["Energy"]
        history["iteration"] = [int(x) for x in np.asarray(en.iters)]
        history["E_mean"] = [float(np.real(x)) for x in np.asarray(en.Mean)]
        history["E_sigma"] = [float(x) for x in np.asarray(en.Sigma)]
    except Exception:
        pass
    atomic_json(d / "history.json", history)

    row = {
        "method": method,
        "seed": seed_name,
        "family": family,
        "target_sector_id": sector_id,
        "target_sector_ground_dimension": sector_dim,
        "E": E_exact,
        "E_per_site": E_exact / g.Ns,
        "delta_E_vs_ED": E_exact - ed_E0,
        "relative_E_error": abs((E_exact-ed_E0)/ed_E0) if ed_E0 else None,
        "fidelity_ground_subspace": fidelity,
        "target_sector_weight": float(final_sector_weight),
        "ground_manifold_dimension": int(ed_gs.shape[1]),
        "projected_sector_residual": sector_residual,
        "initial_target_sector_weight": float(seed_weight),
        "initial_seed_parameter_jitter": float(seed_jitter),
        "n_parameters": int(vstate.n_parameters),
        "n_one_body_parameters": int(basis.n_one_body_parameters),
        "n_onsite_parameters": int(basis.n_onsite),
        "n_complex_bond_parameters": int(basis.n_bond),
        "n_jastrow_parameters": int(len(distances) if use_jastrow else 0),
        "mc_E": float(np.real(final_stats.mean)),
        "mc_E_error": stats_error(final_stats),
        "warmup_mc_E": float(np.real(warm_stats.mean)),
        "n_samples": int(vstate.n_samples),
        "n_iter": int(n_iter),
        "n_chains": int(n_chains),
        "jastrow_shell_distances": ";".join(f"{x:.12g}" for x in distances),
        "time_build_s": setup_s,
        "time_jit_warmup_s": warmup_s,
        "time_solve_or_opt_s": opt_s,
        "time_mc_final_eval_s": mc_eval_s,
        "time_postprocess_s": post_s,
        "time_total_s": setup_s + warmup_s + opt_s + mc_eval_s + post_s,
        **obs.as_dict(),
    }
    row = add_ed_differences(row, {"E": ed_E0, **benchmark_observables(
        g, native_H.states, ed_gs, native_observables
    ).as_dict()})
    atomic_json(d / "summary.json", row)
    return row


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
        "--families", nargs="+",
        choices=["param_slater", "param_sj", "param_vap_sj"],
        default=["param_slater", "param_sj", "param_vap_sj"],
    )
    p.add_argument(
        "--seeds", nargs="+",
        default=["normal", "sd", "isd", "lc_same", "lc_opposite"],
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
    p.add_argument("--ed-k", type=int, default=8)
    p.add_argument("--ed-degeneracy-tol", type=float, default=1e-8)
    p.add_argument("--output", default="results/parametric_hex18_N6_V2")
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
        raise RuntimeError(
            "ED ground manifold fills all requested --ed-k states; increase --ed-k "
            "before doing symmetry decomposition."
        )

    # Unique grounds keep the stronger ordinary T x C3v projector used before.
    # Degenerate grounds are decomposed in the abelian T x C3 subgroup so every
    # target branch has a well-defined 1D character.
    if ed_gs.shape[1] == 1:
        group = build_ordinary_space_group(g, H.states)
        err, name = commutator_error(group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"space-group/H mismatch: {name} {err:.3e}")
        chars = characterize_unique_state(group, ed_gs[:, 0], tol=1e-8)
        ground_sectors = [{
            "characters": chars,
            "basis": ed_gs,
            "dimension": 1,
            "residual": 0.0,
        }]
        projection_group_name = "ordinary translations x C3v"
    else:
        group = build_translation_c3_group(g, H.states)
        err, name = commutator_error(group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"T x C3/H mismatch: {name} {err:.3e}")
        ground_sectors = decompose_ground_manifold_abelian(
            group, ed_gs, tol=max(1e-8, 10*a.ed_degeneracy_tol)
        )
        projection_group_name = "ordinary translations x C3"
        if not ground_sectors:
            raise RuntimeError("Failed to identify any symmetry sector in ED ground manifold")

    basis = build_symmetry_tied_basis(g)
    metadata = {
        "site_orbits": basis.site_orbits,
        "bond_orbits": [
            [{"edge": int(i), "orientation": int(s)} for i, s in orb]
            for orb in basis.bond_orbits
        ],
        "n_onsite_parameters": basis.n_onsite,
        "n_complex_bond_parameters": basis.n_bond,
        "n_one_body_parameters": basis.n_one_body_parameters,
        "subgroup_order": len(basis.subgroup_maps),
        "subgroup": "primitive translations x C3",
        "projected_group": projection_group_name,
        "ed_ground_manifold_dimension": int(ed_gs.shape[1]),
        "ed_ground_sector_count": int(len(ground_sectors)),
    }
    atomic_json(out / "parameterization.json", metadata)
    atomic_json(out / "settings.json", vars(a))
    atomic_json(
        out / "ed_ground_sectors.json",
        {
            "group": projection_group_name,
            "ground_manifold_dimension": int(ed_gs.shape[1]),
            "sector_count": int(len(ground_sectors)),
            "sectors": [
                {
                    "sector_id": i,
                    "dimension": int(sec["dimension"]),
                    "residual": float(sec["residual"]),
                    "characters": [
                        {
                            "operation": op.name,
                            "real": float(ch.real),
                            "imag": float(ch.imag),
                            "phase_over_pi": float(np.angle(ch)/np.pi),
                        }
                        for op, ch in zip(group, sec["characters"])
                    ],
                }
                for i, sec in enumerate(ground_sectors)
            ],
        },
    )

    ed = dict(ed_row)
    ed["method"] = "ED"
    ed["target_sector_weight"] = 1.0
    rows = [ed]
    print(
        f"[parametric] site_orbits={len(basis.site_orbits)}, "
        f"bond_orbits={basis.n_bond}, one-body parameters={basis.n_one_body_parameters}"
    )
    print(
        f"[ED] E0={ed['E']:.12g}, ground_dim={ed_gs.shape[1]}, "
        f"ground_symmetry_sectors={len(ground_sectors)}, F_groundspace=1"
    )
    for i, sec in enumerate(ground_sectors):
        print(
            f"  [ED sector {i}] dim={sec['dimension']} "
            f"residual={sec['residual']:.2e}"
        )

    run_id = 0
    for family in a.families:
        targets = (
            list(enumerate(ground_sectors))
            if family == "param_vap_sj"
            else [(None, None)]
        )
        for sector_id, sec in targets:
            chars = (
                sec["characters"] if sec is not None
                else ground_sectors[0]["characters"]
            )
            sector_dim = int(sec["dimension"]) if sec is not None else None
            for seed in a.seeds:
                run_id += 1
                sector_txt = f", sector={sector_id}" if sector_id is not None else ""
                print(
                    f"\n[run {run_id}] {family}, seed={seed}{sector_txt}",
                    flush=True,
                )
                row = run_one(
                    g, a.nf, a.V, H, ed_gs, float(evals[0]), group, chars, basis,
                    ground_sectors=ground_sectors,
                    sector_id=sector_id,
                    sector_dim=sector_dim,
                    family=family,
                    seed_name=seed,
                    seed_amplitude=a.seed_amplitude,
                    n_shells=a.jastrow_shells,
                    n_samples=a.n_samples,
                    n_discard=a.n_discard,
                    n_chains=a.n_chains,
                    n_iter=a.n_iter,
                    learning_rate=a.learning_rate,
                    diag_shift=a.diag_shift,
                    rng_seed=a.rng_seed + 1009*run_id,
                    chunk_size=a.chunk_size,
                    output_dir=out,
                )
                rows.append(row)
                write_csv(out / "parametric_all.csv", rows)
                print(
                    f"[result] E={row['E']:.12g}, dE={row['delta_E_vs_ED']:.4e}, "
                    f"F_groundspace={row['fidelity_ground_subspace']:.6f}, "
                    f"w={row['target_sector_weight']:.6f}, "
                    f"SLC-/SLC+={row['S_LC_opposite']/max(row['S_LC_same'],1e-15):.3g}, "
                    f"time={row['time_total_s']:.1f}s"
                )

    best = [ed]
    for method in ("Param-Slater", "Param-Slater-Jastrow", "Param-VAP-Slater-Jastrow"):
        cand = [r for r in rows if r.get("method") == method]
        if cand:
            best.append(min(cand, key=lambda r: float(r["E"])))
    write_csv(out / "parametric_best.csv", best)
    atomic_json(out / "parametric_best.json", best)

    print("\n=== Parameterized best comparison ===")
    for r in best:
        print(
            f"{r['method']:30s} seed={str(r.get('seed','')):11s} "
            f"E={float(r['E']):.12g} "
            f"F={float(r.get('fidelity_ground_subspace',1.0)):.6f} "
            f"SCO={float(r['CO_qmax']):.5g} "
            f"SLC+={float(r['S_LC_same']):.5g} "
            f"SLC-={float(r['S_LC_opposite']):.5g} "
            f"npar={r.get('n_parameters','')}"
        )
    print(f"Saved {out/'parametric_all.csv'}")
    print(f"Saved {out/'parametric_best.csv'}")


if __name__ == "__main__":
    main()
