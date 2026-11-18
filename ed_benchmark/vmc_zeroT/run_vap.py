#!/usr/bin/env python3
"""Variation-after-projection (VAP) for the Hex18 Ruby benchmark.

The optimized wavefunction is the ordinary-space-group projection of the same
Varma-style complex Slater or Slater x real density-Jastrow state used in the
unprojected benchmark. The target one-dimensional irrep is read automatically
from the unique exact ED ground state.
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
from vmc_ansatz import (
    minimum_image_shells,
    one_body_seed,
    build_projected_netket_model,
)
from run_benchmark import (
    atomic_json,
    make_netket_hamiltonian,
    reorder_netket_vector_to_native,
    solve_ed,
    stats_error,
)
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


def add_ed_differences(row, ed):
    row["delta_E_vs_ED"] = float(row["E"]) - float(ed["E"])
    row["relative_E_error"] = abs(row["delta_E_vs_ED"]) / max(abs(float(ed["E"])), 1e-15)
    for key in ("CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"):
        ref = float(ed[key]); val = float(row[key])
        row[f"delta_{key}_vs_ED"] = val - ref
        row[f"relative_{key}_error"] = abs(val-ref) / max(abs(ref), 1e-12)
    return row


def initial_base_state_native(M0, native_states, ns, nf):
    """Exact unprojected initial Slater amplitude in native fixed-N basis."""
    Q, _ = np.linalg.qr(np.asarray(M0, complex), mode="reduced")
    out = np.empty(len(native_states), complex)
    for a, s0 in enumerate(native_states):
        s = int(s0)
        occ = [i for i in range(ns) if s & (1 << (ns - 1 - i))]
        if len(occ) != nf:
            raise RuntimeError("Fixed-N basis inconsistency")
        out[a] = np.linalg.det(Q[np.asarray(occ, int), :])
    nrm = np.linalg.norm(out)
    if nrm < 1e-14:
        raise RuntimeError("Initial Slater state has zero norm")
    return out / nrm


def make_projectable_seed(
    M0, native_H, group, chars, *, rng_seed=0, min_weight=1e-8,
    initial_jitter=1e-4, max_tries=8,
):
    """Ensure the unprojected seed has nonzero weight in the target irrep.

    A symmetry-pure broken-symmetry seed can be exactly orthogonal to the
    finite-torus ED irrep, so P_Gamma|Psi_0>=0 and VAP cannot start there.
    We add the smallest deterministic complex orbital perturbation needed to
    give the seed a finite target-sector component. This changes only the
    initialization; the projected variational family and final symmetry are
    unchanged.
    """
    M0 = np.asarray(M0, np.complex128)
    base0 = initial_base_state_native(
        M0, native_H.states, native_H.g.Ns, native_H.nf
    )
    _, w0 = project_to_characters(group, chars, base0)
    if w0 >= min_weight:
        return M0, float(w0), 0.0

    rng = np.random.default_rng(int(rng_seed))
    scale = float(initial_jitter)
    for _ in range(max_tries):
        noise = (
            rng.normal(size=M0.shape) + 1j*rng.normal(size=M0.shape)
        ) / np.sqrt(2.0)
        trial = M0 + scale * noise
        q, _ = np.linalg.qr(trial, mode="reduced")
        base = initial_base_state_native(
            q, native_H.states, native_H.g.Ns, native_H.nf
        )
        _, w = project_to_characters(group, chars, base)
        if w >= min_weight:
            return np.asarray(q, np.complex128), float(w), scale
        scale *= 10.0

    raise RuntimeError(
        "Could not construct a projectable VAP seed after deterministic "
        f"orbital jitter; initial target weight={w0:.3e}, "
        f"last jitter scale={scale/10:.3e}."
    )


def validate_initial_projected_state(
    vstate, hi, native_H, M0, group, chars, *, tol=2e-8
):
    """Compare NetKet projected model to an independent full-basis PAV construction."""
    psi_vap_nk = np.asarray(vstate.to_array(normalize=True))
    psi_vap = reorder_netket_vector_to_native(
        hi, psi_vap_nk, native_H.states, native_H.g.Ns
    )

    base = initial_base_state_native(
        M0, native_H.states, native_H.g.Ns, native_H.nf
    )
    expected, weight = project_to_characters(group, chars, base)
    if expected is None:
        raise RuntimeError("Initial seed has zero weight in the ED symmetry sector")

    overlap = abs(np.vdot(expected, psi_vap))
    residual = np.linalg.norm(psi_vap - expected * np.vdot(expected, psi_vap))
    if 1.0 - overlap > tol or residual > 10 * tol:
        raise RuntimeError(
            "Projected-model convention mismatch: "
            f"|<PAV|VAP-init>|={overlap:.12g}, residual={residual:.3e}. "
            "Check U_g direction, fermion signs, or character conjugation."
        )
    return float(overlap), float(weight)


def run_one(
    g, nf, V, native_H, ed_gs, ed_E0, group, chars, *,
    ansatz, seed_name, seed_amplitude, n_shells, n_samples, n_discard,
    n_chains, n_iter, learning_rate, diag_shift, rng_seed, chunk_size,
    output_dir, validate_projection=False,
):
    import jax
    jax.config.update("jax_enable_x64", True)
    import netket as nk

    t0 = time.perf_counter()
    hi, _, H_fast, graph = make_netket_hamiltonian(g, nf, V)
    star = ruby_star_formfactor(g)
    distances, shells, shell_counts = minimum_image_shells(g, n_shells=n_shells)
    M0_raw = one_body_seed(g, nf, seed_name, seed_amplitude, star)
    M0, initial_target_weight_prebuild, seed_jitter = make_projectable_seed(
        M0_raw, native_H, group, chars,
        rng_seed=rng_seed + 7919,
        min_weight=1e-8,
        initial_jitter=1e-4,
    )
    if seed_jitter > 0:
        print(
            f"[VAP init] seed={seed_name} had zero/negligible target-sector weight; "
            f"applied complex orbital jitter={seed_jitter:.1e}, "
            f"new w={initial_target_weight_prebuild:.3e}",
            flush=True,
        )
    else:
        print(
            f"[VAP init] seed={seed_name} target-sector weight="
            f"{initial_target_weight_prebuild:.6f}",
            flush=True,
        )
    maps = np.asarray([op.site_map for op in group], dtype=np.int32)
    model = build_projected_netket_model(
        M0,
        shells,
        use_jastrow=(ansatz == "slater_jastrow"),
        site_maps=maps,
        characters=np.asarray(chars, complex),
    )

    sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph, n_chains=n_chains)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        chunk_size=chunk_size,
        seed=rng_seed,
        sampler_seed=rng_seed + 104729,
    )
    setup_s = time.perf_counter() - t0

    validation_overlap = None
    initial_target_weight = None
    if validate_projection:
        tv = time.perf_counter()
        validation_overlap, initial_target_weight = validate_initial_projected_state(
            vstate, hi, native_H, M0, group, chars
        )
        validation_s = time.perf_counter() - tv
        print(
            f"[projection-check] overlap={validation_overlap:.12f}, "
            f"unprojected initial w={initial_target_weight:.6f}",
            flush=True,
        )
    else:
        validation_s = 0.0

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
    optimizer_engine = "VAP:VMC+SR(QGTJacobianDense)"

    print(f"[optimizer] {optimizer_engine}", flush=True)
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
    sector_residual = projected_sector_residual(group, chars, psi)
    post_s = time.perf_counter() - tp

    d = output_dir / f"vap_{ansatz}__{seed_name}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "final_state.npz",
        psi_native=psi,
        native_states=native_H.states,
        shell_distances=distances,
        shell_counts=shell_counts,
        star_formfactor=star,
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
        "method": "VAP-Slater" if ansatz == "slater" else "VAP-Slater-Jastrow",
        "seed": seed_name,
        "projection": "variation after ED ordinary-space-group projection",
        "E": E_exact,
        "E_per_site": E_exact / g.Ns,
        "delta_E_vs_ED": E_exact - ed_E0,
        "relative_E_error": abs((E_exact-ed_E0)/ed_E0) if ed_E0 != 0 else None,
        "fidelity_ground_subspace": fidelity,
        "projected_sector_residual": float(sector_residual),
        "mc_E": float(np.real(final_stats.mean)),
        "mc_E_error": stats_error(final_stats),
        "warmup_mc_E": float(np.real(warm_stats.mean)),
        "n_parameters": int(vstate.n_parameters),
        "n_samples": int(vstate.n_samples),
        "n_iter": int(n_iter),
        "n_chains": int(n_chains),
        "optimizer_engine": optimizer_engine,
        "initial_projection_validation_overlap": validation_overlap,
        "initial_unprojected_sector_weight": (
            initial_target_weight if initial_target_weight is not None
            else initial_target_weight_prebuild
        ),
        "initial_seed_orbital_jitter": float(seed_jitter),
        "jastrow_shell_distances": ";".join(f"{x:.12g}" for x in distances),
        "time_build_s": setup_s,
        "time_projection_validation_s": validation_s,
        "time_jit_warmup_s": warmup_s,
        "time_solve_or_opt_s": opt_s,
        "time_mc_final_eval_s": mc_eval_s,
        "time_postprocess_s": post_s,
        "time_total_s": (
            setup_s + validation_s + warmup_s + opt_s + mc_eval_s + post_s
        ),
        **obs.as_dict(),
    }
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
        "--ansatz", choices=["slater", "slater_jastrow", "both"],
        default="slater_jastrow"
    )
    p.add_argument(
        "--seeds", nargs="+", default=["lc_same", "sd", "normal"],
        help="PAV suggests lc_same first for the current Hex18 reference point"
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
    p.add_argument("--commutator-tol", type=float, default=1e-9)
    p.add_argument("--no-projection-check", action="store_true")
    p.add_argument("--output", default="results/vap_hex18_N6_V2")
    return p


def main():
    a = parser().parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    g = native_geometry.RubyPBC("ring3", t=a.t, t1=a.t1, t2=a.t2)
    expected_M = np.array([[1, 1], [-1, 2]], dtype=int)
    if g.name != "ring3" or not np.array_equal(g.M, expected_M):
        raise RuntimeError("VAP benchmark is restricted to verified Hex18 ring3")

    H, evals, ed_gs, ed_row = solve_ed(
        g, a.nf, a.V, k=a.ed_k, degeneracy_tol=a.ed_degeneracy_tol
    )
    if ed_gs.shape[1] != 1:
        raise RuntimeError(
            f"First VAP implementation requires a unique ED ground state; "
            f"found degeneracy={ed_gs.shape[1]}"
        )

    group = build_ordinary_space_group(g, H.states)
    comm_err, comm_name = commutator_error(group, H.operator, n_tests=2)
    if comm_err > a.commutator_tol:
        raise RuntimeError(
            f"Symmetry/H mismatch: {comm_name}, commutator error={comm_err:.3e}"
        )
    chars = characterize_unique_state(group, ed_gs[:, 0], tol=1e-8)

    atomic_json(out / "settings.json", vars(a))
    atomic_json(out / "geometry.json", g.metadata())
    atomic_json(
        out / "target_symmetry_sector.json",
        {
            "group": "ordinary translations x C3v",
            "group_order": len(group),
            "commutator_error": comm_err,
            "characters": [
                {
                    "name": op.name,
                    "real": float(chi.real),
                    "imag": float(chi.imag),
                    "phase_over_pi": float(np.angle(chi)/np.pi),
                }
                for op, chi in zip(group, chars)
            ],
        },
    )

    ed = dict(ed_row)
    ed["method"] = "ED"
    ed["projection"] = "exact ED"
    rows = [ed]
    print(
        f"[Hex18 VAP] Ns={g.Ns}, Nf={a.nf}, "
        f"t={a.t:g}, t1={a.t1:g}, t2={a.t2:g}, V={a.V:g}"
    )
    print(
        f"[target] ordinary space-group order={len(group)}, "
        f"commutator_error={comm_err:.3e}"
    )
    print(f"[ED] E0={ed['E']:.12g}, F=1")

    ansatzes = ["slater", "slater_jastrow"] if a.ansatz == "both" else [a.ansatz]
    run_id = 0
    for ansatz in ansatzes:
        for seed in a.seeds:
            run_id += 1
            print(f"\n[VAP {run_id}] {ansatz}, seed={seed}", flush=True)
            row = run_one(
                g, a.nf, a.V, H, ed_gs, float(evals[0]), group, chars,
                ansatz=ansatz,
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
                validate_projection=(not a.no_projection_check and run_id == 1),
            )
            row = add_ed_differences(row, ed)
            rows.append(row)
            write_csv(out / "vap_all.csv", rows)
            print(
                f"[VAP] E_exact={row['E']:.12g}, "
                f"dE={row['delta_E_vs_ED']:.4e}, "
                f"F={row['fidelity_ground_subspace']:.6f}, "
                f"SLC-/SLC+={row['S_LC_opposite']/max(row['S_LC_same'],1e-15):.3g}, "
                f"time={row['time_total_s']:.1f}s"
            )

    best = [ed]
    for method in ("VAP-Slater", "VAP-Slater-Jastrow"):
        cand = [r for r in rows if r.get("method") == method]
        if cand:
            best.append(min(cand, key=lambda r: float(r["E"])))
    write_csv(out / "vap_best.csv", best)
    atomic_json(out / "vap_best.json", best)

    print("\n=== VAP best comparison ===")
    for r in best:
        print(
            f"{r['method']:24s} seed={str(r.get('seed','')):11s} "
            f"E={float(r['E']):.12g} "
            f"F={float(r['fidelity_ground_subspace']):.6f} "
            f"SCO={float(r['CO_qmax']):.5g} "
            f"SLC+={float(r['S_LC_same']):.5g} "
            f"SLC-={float(r['S_LC_opposite']):.5g}"
        )
    print(f"Saved {out/'vap_all.csv'}")
    print(f"Saved {out/'vap_best.csv'}")


if __name__ == "__main__":
    main()
