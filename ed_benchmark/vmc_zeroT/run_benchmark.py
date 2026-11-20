#!/usr/bin/env python3
"""Benchmark Hex18 exact diagonalization against Varma-style VMC.

Final VMC observables are evaluated by full fixed-N enumeration after training,
so the reported ED/VMC differences contain ansatz/optimization error but no
Monte-Carlo measurement noise. The stochastic VMC energy and its error bar
are retained separately.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from math import comb
import sys
import time
import warnings

import numpy as np
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
if not NATIVE.is_dir():
    raise RuntimeError(
        "Place this directory at ed_benchmark/vmc_zeroT inside RubycGW; "
        f"native ED directory not found: {NATIVE}"
    )
sys.path.insert(0, str(NATIVE))

import geometry as native_geometry
import core as native_core
import observables as native_observables
import symmetry as native_symmetry

from benchmark_observables import benchmark_observables, ruby_star_formfactor
from vmc_ansatz import (
    jastrow_channels, bond_backflow_channels, one_body_seed, build_netket_model
)


def _json_default(obj):
    """Serialize NumPy/JAX scalar and array types in metadata/results."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.complexfloating):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    # JAX arrays and other array-like objects used only in metadata/results.
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _sanitize_json_nonfinite(obj):
    """Recursively replace NaN/Inf metadata by None before strict JSON output."""
    if isinstance(obj, dict):
        return {k: _sanitize_json_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_nonfinite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_json_nonfinite(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (complex, np.complexfloating)):
        z = complex(obj)
        return {
            "real": float(z.real) if np.isfinite(z.real) else None,
            "imag": float(z.imag) if np.isfinite(z.imag) else None,
        }
    return obj


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    clean = _sanitize_json_nonfinite(obj)
    tmp.write_text(
        json.dumps(clean, indent=2, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def _solve_ed_full(g, nf, V, k=8, degeneracy_tol=1e-8):
    """Original full fixed-N sparse ED used by the Hex18 benchmark."""
    t0 = time.perf_counter()
    H = native_core.FixedHamiltonian(g, nf, V, method="sparse")
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    op = H.operator
    if hasattr(op, "data") and np.max(np.abs(np.imag(op.data))) < 1e-14:
        op = op.real
    n = op.shape[0]
    kk = min(int(k), max(1, n - 2))
    evals, evecs = eigsh(op, k=kk, which="SA", tol=1e-10, maxiter=max(3000, 8 * n))
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    t_diag = time.perf_counter() - t1
    if not len(evals):
        raise RuntimeError("Empty ED spectrum")
    mask = (evals - evals[0]) <= degeneracy_tol
    deg = int(np.count_nonzero(mask))
    truncated = deg == len(evals)
    gs = evecs[:, mask]
    t2 = time.perf_counter()
    obs = benchmark_observables(g, H.states, gs, native_observables)
    t_obs = time.perf_counter() - t2
    row = {
        "method": "ED",
        "seed": "exact",
        "E": float(evals[0]),
        "E_per_site": float(evals[0] / g.Ns),
        "ground_degeneracy": deg,
        "ground_manifold_truncated": bool(truncated),
        "ed_backend": "full_sparse",
        "mc_E": None,
        "mc_E_error": None,
        "fidelity_ground_subspace": 1.0,
        "time_build_s": t_build,
        "time_solve_or_opt_s": t_diag,
        "time_postprocess_s": t_obs,
        "time_total_s": t_build + t_diag + t_obs,
        **obs.as_dict(),
    }
    return H, evals, gs, row


def _solve_ed_symmetry(g, nf, V, k=8, degeneracy_tol=1e-8, sector_low=4):
    """Pure NumPy/SciPy symmetry-blocked ED using native_ed/symmetry.py.

    This reuses the repository's existing finite-group projectors: translation
    sectors are resolved first and compatible point-group little-group
    generators are added.  No QuSpin binary extension is required.
    """
    t0=time.perf_counter()
    states=native_core.make_states(g.Ns,nf)
    H=native_core.FixedHamiltonian(g,nf,V,states=states,method="sparse")
    t_build=time.perf_counter()-t0

    keep=max(2,int(k))
    candidates=[]
    records=[]
    dim_total=0
    t1=time.perf_counter()

    for label,Q,states_check,_ in native_symmetry.symmetry_projectors(
        g,nf,kind="point",max_basis=max(900000,len(states)),use_signed=True
    ):
        if not np.array_equal(states_check,states):
            raise RuntimeError("Native symmetry projector basis mismatch")
        d=int(Q.shape[1])
        dim_total+=d
        clean_label={kk:vv for kk,vv in label.items() if kk!="generators"}
        if d==0:
            records.append({"labels":clean_label,"dim":0,"eigenvalues":[]})
            continue

        HB=(Q.conj().T@(H.operator@Q)).tocsr()
        diff=HB-HB.getH()
        if diff.nnz and np.max(np.abs(diff.data))>1e-8:
            raise RuntimeError("Native symmetry block Hamiltonian is non-Hermitian")

        # The existing native 2x2 solver used only a few low states per
        # symmetry block. Four is enough to detect ordinary finite-size
        # multiplets while keeping the sector scan much cheaper than asking
        # for all global --ed-k states in every block.
        nev=min(max(2,int(sector_low)),d)
        evals_block,vecs_block=native_core.lowest(HB,nev)
        records.append({
            "labels":clean_label,
            "dim":d,
            "eigenvalues":[float(x) for x in evals_block],
        })

        for j,en in enumerate(evals_block):
            psi=np.asarray(Q@vecs_block[:,j]).ravel().astype(np.complex128,copy=False)
            nrm=float(np.linalg.norm(psi))
            if nrm<1e-14:
                raise RuntimeError("Zero-norm state reconstructed from native symmetry block")
            psi=psi/nrm
            candidates.append((float(en),psi,clean_label,d,len(evals_block)))

        # Retain only the globally lowest requested states, so large-projector
        # matrices can be released sector by sector instead of accumulating.
        candidates.sort(key=lambda x:x[0])
        if len(candidates)>keep:
            candidates=candidates[:keep]

    t_diag=time.perf_counter()-t1
    if dim_total!=len(states):
        raise RuntimeError(
            f"Incomplete native symmetry decomposition: {dim_total} != {len(states)}"
        )
    if not candidates:
        raise RuntimeError("Native symmetry ED returned no eigenstates")

    candidates.sort(key=lambda x:x[0])
    evals=np.asarray([x[0] for x in candidates],dtype=float)
    e0=float(evals[0])
    mask=(evals-e0)<=degeneracy_tol
    ground=[x for x,m in zip(candidates,mask) if m]
    truncated=bool(len(candidates)>=keep and evals[-1]-e0<=degeneracy_tol)
    for rec in records:
        vals=np.asarray(rec["eigenvalues"],dtype=float)
        if (
            int(rec["dim"])>len(vals)
            and len(vals)>0
            and vals[-1]-e0<=degeneracy_tol
        ):
            truncated=True
            break

    gs=np.column_stack([x[1] for x in ground])
    gs,_=np.linalg.qr(gs)

    t2=time.perf_counter()
    obs=benchmark_observables(g,H.states,gs,native_observables)
    t_obs=time.perf_counter()-t2

    row={
        "method":"ED",
        "seed":"exact",
        "E":e0,
        "E_per_site":e0/g.Ns,
        "ground_degeneracy":int(gs.shape[1]),
        "ground_manifold_truncated":truncated,
        "ed_backend":"native_symmetry_blocks",
        "ed_symmetry_sector_count":int(len(records)),
        "ed_eigenpairs_per_sector_max":int(max((len(r["eigenvalues"]) for r in records),default=0)),
        "ed_full_dimension":int(comb(g.Ns,nf)),
        "mc_E":None,
        "mc_E_error":None,
        "fidelity_ground_subspace":1.0,
        "time_build_s":t_build,
        "time_solve_or_opt_s":t_diag,
        "time_postprocess_s":t_obs,
        "time_total_s":t_build+t_diag+t_obs,
        **obs.as_dict(),
    }
    return H,evals,gs,row


def solve_ed(g, nf, V, k=8, degeneracy_tol=1e-8, backend="auto", sector_low=4):
    """ED frontend shared by Hex18 and larger finite-cluster benchmarks.

    backend='auto' keeps the original full sparse ED for small Hilbert spaces
    and switches to the native NumPy/SciPy symmetry-blocked solver once the full
    fixed-N dimension exceeds 2e5.
    """
    if backend not in ("auto", "full", "symmetry"):
        raise ValueError("backend must be auto, full, or symmetry")
    chosen = backend
    if chosen == "auto":
        chosen = "symmetry" if comb(g.Ns, nf) > 200000 else "full"
    if chosen == "symmetry":
        return _solve_ed_symmetry(
            g, nf, V, k=k, degeneracy_tol=degeneracy_tol,
            sector_low=sector_low,
        )
    return _solve_ed_full(g, nf, V, k=k, degeneracy_tol=degeneracy_tol)


def make_netket_hamiltonian(g, nf, V):
    import netket as nk
    from netket.operator.fermion import create as cdag, destroy as c, number as nc

    hi = nk.hilbert.SpinOrbitalFermions(g.Ns, s=None, n_fermions=nf)
    H = 0.0
    for e in g.edges:
        t = float(g.t[e.kind])
        H += t * (cdag(hi, e.i) @ c(hi, e.j) + cdag(hi, e.j) @ c(hi, e.i))
        if e.kind == "t" and V != 0:
            H += float(V) * (nc(hi, e.i) @ nc(hi, e.j))
    # Explicitly use the fixed-particle-number representation when available.
    # This reduces the number of connected configurations entering every local
    # energy evaluation, which is especially important for projected
    # multi-image wavefunctions.
    H_fast = None
    try:
        from netket.experimental.operator import ParticleNumberConservingFermioperator2nd
        H_fast = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(H)
        try:
            H_fast = H_fast.collect()
        except Exception:
            pass
    except Exception:
        H_fast = None
    if H_fast is None:
        try:
            H_fast = H.collect()
        except Exception:
            H_fast = H
    graph = nk.graph.Graph(edges=[(e.i, e.j) for e in g.edges], n_nodes=g.Ns)
    return hi, H, H_fast, graph


def netket_to_native_indices(hi, native_states, ns):
    """Index map: NetKet basis row -> native ED basis row.

    NetKet enumerates fixed-N states as occupation vectors, while native ED
    stores each occupation pattern as a uint64 bit string with site 0 as the
    most-significant bit. Return order[a] = native row for NetKet row a.
    """
    nk_states = np.asarray(hi.all_states())
    if nk_states.ndim != 2 or nk_states.shape[1] != ns:
        raise RuntimeError(f"Unexpected NetKet state shape: {nk_states.shape}")
    weights = np.uint64(1) << np.arange(ns - 1, -1, -1, dtype=np.uint64)
    encoded = np.sum(
        nk_states.astype(np.uint64) * weights[None, :],
        axis=1,
        dtype=np.uint64,
    )
    native_states = np.asarray(native_states, dtype=np.uint64)
    order = np.searchsorted(native_states, encoded)
    if np.any(order >= len(native_states)) or np.any(native_states[order] != encoded):
        raise RuntimeError("NetKet/native basis sets do not match")
    if len(np.unique(order)) != len(order):
        raise RuntimeError("NetKet/native basis mapping is not one-to-one")
    return order.astype(np.int64)


def validate_netket_native_hamiltonian(hi, H_generic, native_H, ns, tol=1e-10):
    """Fail early if fermion ordering/sign conventions differ between engines."""
    order = netket_to_native_indices(hi, native_H.states, ns)
    h_nk = H_generic.to_sparse().tocsr()
    h_native_nk = native_H.operator[order, :][:, order].tocsr()
    d = (h_nk - h_native_nk).tocsr()
    err = float(np.max(np.abs(d.data))) if d.nnz else 0.0
    if err > tol:
        raise RuntimeError(
            f"NetKet/native Hamiltonian mismatch: max|dH|={err:.3e}. "
            "Do not interpret VMC results until basis/sign conventions are fixed."
        )
    return err


def reorder_netket_vector_to_native(hi, psi_nk, native_states, ns):
    order = netket_to_native_indices(hi, native_states, ns)
    out = np.empty(len(native_states), dtype=np.complex128)
    out[order] = np.asarray(psi_nk, np.complex128)
    out /= np.linalg.norm(out)
    return out


def stats_error(stats):
    for name in ("error_of_mean", "error", "sigma"):
        if hasattr(stats, name):
            try:
                return float(np.real(getattr(stats, name)))
            except Exception:
                pass
    return None


def run_vmc_one(
    g, nf, V, native_H, ed_gs, ed_E0, *, ansatz, seed_name, seed_amplitude,
    n_shells, n_samples, n_discard, n_chains, n_iter, learning_rate,
    diag_shift, rng_seed, chunk_size, output_dir, validate_hamiltonian=False,
    jastrow_mode="distance", use_backflow=False,
    method_name=None, dir_prefix=None,
):
    import jax
    jax.config.update("jax_enable_x64", True)
    import netket as nk

    t0 = time.perf_counter()
    hi, H_generic, H_fast, graph = make_netket_hamiltonian(g, nf, V)
    hamiltonian_max_abs_diff = None
    if validate_hamiltonian:
        hamiltonian_max_abs_diff = validate_netket_native_hamiltonian(
            hi, H_generic, native_H, g.Ns
        )
    star = ruby_star_formfactor(g)
    jastrow_labels, distances, shells, shell_counts = jastrow_channels(
        g, mode=jastrow_mode, n_shells=n_shells
    )
    M0 = one_body_seed(g, nf, seed_name, seed_amplitude, star)
    bf_labels, bf_mats, bf_counts = bond_backflow_channels(g)
    model = build_netket_model(
        M0,
        shells,
        use_jastrow=(ansatz == "slater_jastrow"),
        backflow_matrices=(bf_mats if use_backflow else None),
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

    tw = time.perf_counter()
    warm_stats = vstate.expect(H_fast)
    warmup_s = time.perf_counter() - tw

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    # VMC_SR/SRt requires all parameter leaves to have the same dtype.
    # Our Varma-style ansatz intentionally has a complex Slater matrix M
    # and a *real* density-Jastrow vector v. For that mixed real/complex
    # PyTree use the standard VMC driver with an explicit dense-Jacobian SR
    # preconditioner, which NetKet supports for mixed parameter types.
    if ansatz == "slater_jastrow":
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
        optimizer_engine = "VMC+SR(QGTJacobianDense,mixed-real-complex)"
    else:
        driver = nk.driver.VMC_SR(
            H_fast,
            optimizer,
            variational_state=vstate,
            diag_shift=diag_shift,
            mode="complex",
        )
        optimizer_engine = "VMC_SR(complex)"
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
    Hpsi = native_H.operator @ psi
    E_exact = float(np.vdot(psi, Hpsi).real)
    obs = benchmark_observables(g, native_H.states, psi, native_observables)
    overlap = ed_gs.conj().T @ psi
    fidelity = float(np.sum(np.abs(overlap) ** 2).real)
    post_s = time.perf_counter() - tp

    if dir_prefix is None:
        dir_prefix = ansatz
    d = output_dir / f"{dir_prefix}__{seed_name}"
    d.mkdir(parents=True, exist_ok=True)
    params_np = vstate.parameters
    M_opt = np.asarray(params_np["M"], np.complex128)
    v_opt = (
        np.asarray(params_np["v"], np.float64)
        if "v" in params_np else np.empty(0, dtype=np.float64)
    )
    eta_saved = (
        np.asarray(params_np["eta"], np.complex128)
        if "eta" in params_np else np.empty(0, dtype=np.complex128)
    )
    np.savez_compressed(
        d / "final_state.npz",
        psi_native=psi,
        native_states=native_H.states,
        shell_distances=distances,
        shell_counts=shell_counts,
        jastrow_labels=np.asarray(jastrow_labels),
        star_formfactor=star,
        variational_M=M_opt,
        variational_v=v_opt,
        variational_eta=eta_saved,
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

    eta_opt = None
    if use_backflow:
        try:
            eta_opt = np.asarray(vstate.parameters["eta"], np.complex128)
        except Exception:
            eta_opt = None

    if method_name is None:
        method_name = "VMC-Slater" if ansatz == "slater" else "VMC-Slater-Jastrow"
        if use_backflow:
            method_name += "-Backflow"

    row = {
        "method": method_name,
        "seed": seed_name,
        "E": E_exact,
        "E_per_site": E_exact / g.Ns,
        "delta_E_vs_ED": E_exact - ed_E0,
        "relative_E_error": abs((E_exact - ed_E0) / ed_E0) if ed_E0 != 0 else None,
        "ground_degeneracy": int(ed_gs.shape[1]),
        "ground_manifold_truncated": False,
        "mc_E": float(np.real(final_stats.mean)),
        "mc_E_error": stats_error(final_stats),
        "warmup_mc_E": float(np.real(warm_stats.mean)),
        "fidelity_ground_subspace": fidelity,
        "n_parameters": int(vstate.n_parameters),
        "n_samples": int(vstate.n_samples),
        "n_iter": int(n_iter),
        "n_chains": int(n_chains),
        "optimizer_engine": optimizer_engine,
        "hamiltonian_max_abs_diff": hamiltonian_max_abs_diff,
        "jastrow_mode": jastrow_mode,
        "jastrow_labels": ";".join(map(str, jastrow_labels)),
        "jastrow_shell_distances": ";".join(f"{x:.12g}" for x in distances),
        "jastrow_channel_counts": ";".join(str(int(x)) for x in shell_counts),
        "backflow_enabled": bool(use_backflow),
        "backflow_labels": ";".join(bf_labels) if use_backflow else "",
        "backflow_channel_counts": (
            ";".join(str(int(x)) for x in bf_counts) if use_backflow else ""
        ),
        "n_backflow_parameters_complex": int(len(bf_labels) if use_backflow else 0),
        "backflow_eta_real": (
            "" if eta_opt is None else ";".join(f"{z.real:.16g}" for z in eta_opt)
        ),
        "backflow_eta_imag": (
            "" if eta_opt is None else ";".join(f"{z.imag:.16g}" for z in eta_opt)
        ),
        "time_build_s": setup_s,
        "time_jit_warmup_s": warmup_s,
        "time_solve_or_opt_s": opt_s,
        "time_mc_final_eval_s": mc_eval_s,
        "time_postprocess_s": post_s,
        "time_total_s": setup_s + warmup_s + opt_s + mc_eval_s + post_s,
        **obs.as_dict(),
    }
    atomic_json(d / "summary.json", row)
    return row


def add_ed_differences(row: dict, ed_row: dict) -> dict:
    if row.get("method") == "ED":
        row["time_ratio_vs_ED"] = 1.0
    else:
        row["time_ratio_vs_ED"] = float(row["time_total_s"]) / max(
            float(ed_row["time_total_s"]), 1e-15
        )
    for key in ("CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"):
        ref = float(ed_row[key])
        val = float(row[key])
        row[f"delta_{key}_vs_ED"] = val - ref
        row[f"relative_{key}_error"] = abs(val - ref) / max(abs(ref), 1e-12)
    return row


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def best_rows(rows):
    out = []
    out.extend([r for r in rows if r["method"] == "ED"])
    for method in ("VMC-Slater", "VMC-Slater-Jastrow"):
        cand = [r for r in rows if r["method"] == method]
        if cand:
            out.append(min(cand, key=lambda r: float(r["E"])))
    return out


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--nf", type=int, default=6)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--t1", type=float, default=0.5)
    p.add_argument("--t2", type=float, default=0.5)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--ansatz", choices=["slater", "slater_jastrow", "both"], default="both")
    p.add_argument(
        "--seeds",
        nargs="+",
        default=["normal", "sd", "isd", "lc_same", "lc_opposite"],
    )
    p.add_argument("--seed-amplitude", type=float, default=0.08)
    p.add_argument("--jastrow-shells", type=int, default=4)
    p.add_argument(
        "--jastrow-mode", choices=["distance", "bond_aware"], default="distance",
        help="distance: old distance shells; bond_aware: separate t/t1/t2 bonds plus nonbond shells"
    )
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--n-discard", type=int, default=16)
    p.add_argument("--n-chains", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--diag-shift", type=float, default=0.05)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--rng-seed", type=int, default=2026)
    p.add_argument("--ed-k", type=int, default=8)
    p.add_argument("--ed-degeneracy-tol", type=float, default=1e-8)
    p.add_argument("--output", default="results/vmc_hex18_N6_V1")
    return p


def main():
    a = parser().parse_args()
    if not 0 < a.nf < 18:
        raise ValueError("Require 0 < Nf < 18")
    if a.n_samples < 128 or a.n_iter < 1 or a.n_chains < 1:
        raise ValueError("Invalid VMC sampling/iteration settings")

    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    g = native_geometry.RubyPBC("ring3", t=a.t, t1=a.t1, t2=a.t2)
    # Hard guard: this benchmark is the 18-site hexagonal ring torus (ring3),
    # never the distinct 3x1 torus that happens to contain the same number of
    # primitive cells/sites.
    expected_M = np.array([[1, 1], [-1, 2]], dtype=int)
    expected_reps = [(0, 0), (1, 0), (1, 1)]
    if g.name != "ring3" or not np.array_equal(g.M, expected_M) or list(g.reps) != expected_reps:
        raise RuntimeError(
            f"Hex18 geometry guard failed: name={g.name}, M={g.M.tolist()}, reps={g.reps}"
        )
    if np.array_equal(g.M, np.array(native_geometry.SPECS["3x1"][0], dtype=int)):
        raise RuntimeError("Refusing to run: geometry is the 3x1 torus, not Hex18 ring3")
    if g.point_maps().get("C3") is None:
        raise RuntimeError("Hex18 ring3 must retain the verified C3 point map")

    atomic_json(out / "geometry.json", g.metadata())
    settings = vars(a).copy()
    settings.update(
        {"cluster": "ring3", "Ns": g.Ns, "filling_per_primitive": a.nf / g.num_cells}
    )
    atomic_json(out / "settings.json", settings)

    print(f"[Hex18] Ns={g.Ns}, Nf={a.nf}, n={a.nf/g.num_cells:g} per primitive cell")
    print(f"[geometry] ring3 M={g.M.tolist()}, reps={g.reps}; verified C3=yes")
    print(f"[model] t={a.t:g}, t1={a.t1:g}, t2={a.t2:g}, Vtri={a.V:g}; PBC ring3")

    H, evals, ed_gs, ed_row = solve_ed(
        g, a.nf, a.V, k=a.ed_k, degeneracy_tol=a.ed_degeneracy_tol
    )
    ed_row = add_ed_differences(ed_row, ed_row)
    rows = [ed_row]
    print(
        f"[ED] E0={ed_row['E']:.12g}, deg={ed_row['ground_degeneracy']}, "
        f"total={ed_row['time_total_s']:.3f}s"
    )
    if ed_row["ground_manifold_truncated"]:
        warnings.warn("ED ground multiplet fills all requested ed-k states; increase --ed-k")

    ansatzes = ["slater", "slater_jastrow"] if a.ansatz == "both" else [a.ansatz]
    run_id = 0
    for ansatz in ansatzes:
        for seed in a.seeds:
            run_id += 1
            print(f"\n[VMC {run_id}] {ansatz}, seed={seed}", flush=True)
            row = run_vmc_one(
                g, a.nf, a.V, H, ed_gs, float(evals[0]),
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
                rng_seed=a.rng_seed + 1009 * run_id,
                chunk_size=a.chunk_size,
                output_dir=out,
                validate_hamiltonian=(run_id == 1),
                jastrow_mode=a.jastrow_mode,
            )
            row = add_ed_differences(row, ed_row)
            rows.append(row)
            write_csv(out / "benchmark_all.csv", rows)
            print(
                f"[VMC] E_exact={row['E']:.12g}, dE={row['delta_E_vs_ED']:.4e}, "
                f"F_gs={row['fidelity_ground_subspace']:.6f}, "
                f"total={row['time_total_s']:.2f}s"
            )

    best = best_rows(rows)
    write_csv(out / "benchmark_best.csv", best)
    atomic_json(out / "benchmark_best.json", best)

    print("\n=== Best comparison ===")
    for r in best:
        print(
            f"{r['method']:22s} seed={r['seed']:11s} E={r['E']:.12g} "
            f"SCO={r['CO_qmax']:.5g} SBOW={r['S_BOW_star']:.5g} "
            f"SLC+={r['S_LC_same']:.5g} SLC-={r['S_LC_opposite']:.5g} "
            f"time={r['time_total_s']:.3f}s"
        )
    print(f"Saved: {out/'benchmark_all.csv'}")
    print(f"Saved: {out/'benchmark_best.csv'}")


if __name__ == "__main__":
    main()
