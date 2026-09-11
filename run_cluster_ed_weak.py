#!/usr/bin/env python3
"""Run six-site ED self-energy embedding with a selectable weak solver.

``Lx,Ly`` label the optional finite-torus ED benchmark geometry.  The lattice
momentum integration is controlled independently by ``nk1,nk2``.  When nk1 or
nk2 is omitted it defaults to Lx or Ly respectively, reproducing the historical
finite-torus calculation exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_gw_fast import ClusterEDGWFastOptions
from rubycgw.cluster_ed_weak_fast import WEAK_SOLVERS, solve_cluster_ed_weak_fast
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions, GWResult
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.sox_covariant import SOXOptions


_CACHE_VERSION = 2


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=1)
    p.add_argument("--nk1", type=int, default=None, help="lattice k mesh; default Lx")
    p.add_argument("--nk2", type=int, default=None, help="lattice k mesh; default Ly")
    p.add_argument("--weak-solver", choices=WEAK_SOLVERS, default="gw")
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--weak-max", type=int, default=160)
    p.add_argument("--weak-tol", type=float, default=1e-8)
    p.add_argument("--weak-mixing", type=float, default=0.25)
    p.add_argument("--weak-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--gf2-nquad", type=int, default=64)
    p.add_argument("--gf2-tail-edge-points", type=int, default=2)

    p.add_argument("--embed-max", type=int, default=100)
    p.add_argument("--embed-tol", type=float, default=2e-5)
    p.add_argument("--embed-mixing", type=float, default=0.80)
    p.add_argument("--embed-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--embed-pulay-history", type=int, default=6)
    p.add_argument("--embed-pulay-start", type=int, default=3)
    p.add_argument("--embed-pulay-regularization", type=float, default=1e-7)
    p.add_argument("--embed-pulay-step-cap", type=float, default=3.0)
    p.add_argument("--impurity-mixing", type=float, default=1.0)

    p.add_argument("--nbath", type=int, default=6)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)

    p.add_argument("--benchmark-ed", action="store_true")
    p.add_argument("--quiet-weak", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/cluster_ed_weak"))
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    return p.parse_args()


def _resolved_mesh(args) -> tuple[int, int]:
    nk1 = int(args.Lx if args.nk1 is None else args.nk1)
    nk2 = int(args.Ly if args.nk2 is None else args.nk2)
    if nk1 < 1 or nk2 < 1:
        raise ValueError("nk1,nk2 must be positive")
    return nk1, nk2


def _hash_payload(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _background_cache_path(cache_dir: Path, args, grid: MatsubaraGrid) -> Path:
    payload = {
        "version": _CACHE_VERSION,
        "kind": "weak_background",
        "weak_solver": str(args.weak_solver),
        "nk1": int(grid.nk1),
        "nk2": int(grid.nk2),
        "V": float(args.V),
        "filling": float(args.filling),
        "T": float(args.T),
        "ti": float(args.ti),
        "t1": float(args.t1),
        "t2": float(args.t2),
        "nw": int(args.nw),
        "nomega": int(args.nomega),
        "weak_tol": float(args.weak_tol),
        "gf2_nquad": int(args.gf2_nquad),
    }
    return cache_dir / f"{args.weak_solver}_{_hash_payload(payload)}.npz"


def _save_background(path: Path, result: GWResult):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H),
        Sigma_GW=np.asarray(result.Sigma_GW),
        mu=float(result.mu),
        density=np.asarray(result.density),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        mixing_method=str(result.mixing_method),
        min_screening_singular_value=float(result.min_screening_singular_value),
        min_screening_m=int(result.min_screening_m),
        min_screening_Omega=float(result.min_screening_Omega),
        min_screening_q1=float(result.min_screening_q1),
        min_screening_q2=float(result.min_screening_q2),
        min_screening_mode=np.asarray(result.min_screening_mode),
        min_density_mode=np.asarray(result.min_density_mode),
        min_density_mode_residual=float(result.min_density_mode_residual),
    )


def _load_background(path: Path) -> GWResult:
    with np.load(path, allow_pickle=False) as z:
        return GWResult(
            G=np.asarray(z["G"]),
            W=np.asarray(z["W"]),
            P=np.asarray(z["P"]),
            Sigma_H=np.asarray(z["Sigma_H"]),
            Sigma_GW=np.asarray(z["Sigma_GW"]),
            mu=float(z["mu"]),
            density=np.asarray(z["density"]),
            converged=bool(z["converged"]),
            iterations=int(z["iterations"]),
            final_error=float(z["final_error"]),
            mixing_method=str(z["mixing_method"]),
            min_screening_singular_value=float(z["min_screening_singular_value"]),
            min_screening_m=int(z["min_screening_m"]),
            min_screening_Omega=float(z["min_screening_Omega"]),
            min_screening_q1=float(z["min_screening_q1"]),
            min_screening_q2=float(z["min_screening_q2"]),
            min_screening_mode=np.asarray(z["min_screening_mode"]),
            min_density_mode=np.asarray(z["min_density_mode"]),
            min_density_mode_residual=float(z["min_density_mode_residual"]),
        )


def _lattice_to_realspace(Gk: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    arr = np.asarray(Gk, dtype=complex)
    if arr.shape[1:3] != (int(Lx), int(Ly)):
        raise ValueError("finite-torus transform requires nk1=Lx and nk2=Ly")
    nf = arr.shape[0]
    cells = [(r1, r2) for r1 in range(Lx) for r2 in range(Ly)]
    out = np.zeros((nf, 6 * Lx * Ly, 6 * Lx * Ly), dtype=complex)
    for c, (r1, r2) in enumerate(cells):
        for d, (s1, s2) in enumerate(cells):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for i in range(Lx):
                for j in range(Ly):
                    phase = np.exp(
                        2j * np.pi * ((i / Lx) * (r1 - s1) + (j / Ly) * (r2 - s2))
                    )
                    block += phase * arr[:, i, j]
            out[:, 6*c:6*(c+1), 6*d:6*(d+1)] = block / float(Lx * Ly)
    return out


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    return float(np.linalg.norm((aa - bb).ravel()) / max(np.linalg.norm(bb.ravel()), 1e-300))


def main():
    args = _args()
    if args.Lx < 1 or args.Ly < 1:
        raise ValueError("Lx,Ly must be positive")
    nk1, nk2 = _resolved_mesh(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=nk1,
        nk2=nk2,
        nw=int(args.nw),
        nOmega=int(args.nomega),
        T=float(args.T),
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir is not None else args.out / "cache"
    use_cache = not bool(args.no_cache)

    weak_opts = GWOptions(
        target_filling=float(args.filling),
        max_iter=int(args.weak_max),
        tol=float(args.weak_tol),
        mixing=float(args.weak_mixing),
        mixing_method=str(args.weak_mixing_method),
        verbose=not bool(args.quiet_weak),
        momentum_backend="fft",
    )
    embed_opts = ClusterEDGWFastOptions(
        max_iter=int(args.embed_max),
        tol=float(args.embed_tol),
        mixing=float(args.embed_mixing),
        mixing_method=str(args.embed_mixing_method),
        pulay_history=int(args.embed_pulay_history),
        pulay_start=int(args.embed_pulay_start),
        pulay_regularization=float(args.embed_pulay_regularization),
        pulay_step_cap=float(args.embed_pulay_step_cap),
        impurity_mixing=float(args.impurity_mixing),
        nbath=int(args.nbath),
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_energy_window=float(args.bath_energy_window),
        bath_coupling_bound=float(args.bath_coupling_bound),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=True,
    )
    gf2_opts = SOXOptions(
        n_quad=int(args.gf2_nquad),
        tail_complete=True,
        tail_edge_points=int(args.gf2_tail_edge_points),
    )

    cache_path = _background_cache_path(cache_dir, args, grid)
    background = None
    if use_cache and cache_path.exists() and not args.refresh_cache:
        try:
            candidate = _load_background(cache_path)
            if candidate.converged and candidate.final_error <= max(args.weak_tol, 1e-12) * 1.05:
                background = candidate
                print(f"[cache] {args.weak_solver.upper()} hit: {cache_path}")
        except Exception as exc:
            print(f"[cache] ignoring unreadable background {cache_path}: {exc}")

    print("=== Ruby 6-site ED + selectable weak solver ===")
    print(
        f"benchmark L={args.Lx}x{args.Ly}; kmesh={nk1}x{nk2}; weak={args.weak_solver}; "
        f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, nw={args.nw}, "
        f"nOmega={args.nomega}, nbath={args.nbath}"
    )
    result = solve_cluster_ed_weak_fast(
        h0,
        Vq,
        params,
        grid,
        weak_solver=args.weak_solver,
        gw_opts=weak_opts,
        embed_opts=embed_opts,
        gf2_sox_opts=gf2_opts,
        background=background,
    )
    if use_cache and (background is None or args.refresh_cache):
        _save_background(cache_path, result.background)
        print(f"[cache] {args.weak_solver.upper()} saved: {cache_path}")

    print(
        f"final: converged={result.converged}, iterations={result.iterations}, "
        f"residual={result.final_error:.3e}, impurity_mismatch={result.impurity_mismatch:.3e}, "
        f"bath_fit={result.bath_fit_error:.3e}, mu={result.mu:+.10f}, "
        f"mix={result.mixing_method}, pulay_fallbacks={result.pulay_fallbacks}"
    )

    default_mesh = (nk1, nk2) == (args.Lx, args.Ly)
    stem = f"cluster_ed_{args.weak_solver}_L{args.Lx}x{args.Ly}"
    if not default_mesh:
        stem += f"_nk{nk1}x{nk2}"
    stem += f"_V{args.V:.6g}_fill{args.filling:.6g}"
    outfile = args.out / f"{stem}.npz"

    benchmark_status = "not_requested"
    Gerr_bg = Gerr_emb = mu_ed = np.nan
    G_ed = np.empty((0,), dtype=complex)
    if args.benchmark_ed:
        nsites = 6 * args.Lx * args.Ly
        if not default_mesh:
            benchmark_status = "mesh_differs_from_finite_torus"
            print(
                "[benchmark] skipped: exact finite-torus ED is only an apples-to-apples "
                "comparison when nk1=Lx and nk2=Ly."
            )
        elif nsites > 16:
            benchmark_status = "full_ed_unsupported"
            print(f"[benchmark] skipped: full-spectrum ED supports <=16 sites, got {nsites}")
        else:
            exact = ExactSmallRubyThermal(args.Lx, args.Ly, params)
            exact.diagonalize(float(args.V))
            mu_ed = exact.solve_mu(float(args.filling) * args.Lx * args.Ly, float(args.T))
            G_ed, _ = exact.green_iomega(1j * grid.omega, mu_ed, float(args.T))
            G_bg_real = _lattice_to_realspace(result.background.G, args.Lx, args.Ly)
            G_emb_real = _lattice_to_realspace(result.G, args.Lx, args.Ly)
            Gerr_bg = _relerr(G_bg_real, G_ed)
            Gerr_emb = _relerr(G_emb_real, G_ed)
            benchmark_status = "exact_ed_complete"
            print(
                f"[benchmark] mu_ED={mu_ed:+.10f}, Gerr_{args.weak_solver}={Gerr_bg:.6e}, "
                f"Gerr_clusterED+{args.weak_solver.upper()}={Gerr_emb:.6e}"
            )

    np.savez_compressed(
        outfile,
        Lx=int(args.Lx),
        Ly=int(args.Ly),
        nk1=int(nk1),
        nk2=int(nk2),
        weak_solver=str(args.weak_solver),
        gf2_nquad=int(args.gf2_nquad),
        V=float(args.V),
        filling=float(args.filling),
        T=float(args.T),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        omega=np.asarray(grid.omega),
        Omega=np.asarray(grid.Omega),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        impurity_mismatch=float(result.impurity_mismatch),
        bath_fit_error=float(result.bath_fit_error),
        mixing_method=str(result.mixing_method),
        pulay_fallbacks=int(result.pulay_fallbacks),
        residual_history=np.asarray(result.residual_history),
        impurity_residual_history=np.asarray(result.impurity_residual_history),
        impurity_mismatch_history=np.asarray(result.impurity_mismatch_history),
        bath_fit_history=np.asarray(result.bath_fit_history),
        mu_history=np.asarray(result.mu_history),
        bath_nfev_history=np.asarray(result.bath_nfev_history),
        elapsed_history=np.asarray(result.elapsed_history),
        mu=float(result.mu),
        density=np.asarray(result.density),
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H),
        Sigma_emb=np.asarray(result.Sigma_emb),
        Sigma_weak_lattice=np.asarray(result.Sigma_weak_lattice),
        Sigma_weak_cluster=np.asarray(result.Sigma_weak_cluster),
        # Legacy names are retained so old analysis scripts can still load the file.
        Sigma_GW_lattice=np.asarray(result.Sigma_weak_lattice),
        Sigma_GW_cluster=np.asarray(result.Sigma_weak_cluster),
        Sigma_ED_cluster=np.asarray(result.Sigma_ED_cluster),
        G_cluster=np.asarray(result.G_cluster),
        G_impurity=np.asarray(result.G_impurity),
        bath_energies=np.asarray(result.bath.energies),
        bath_couplings=np.asarray(result.bath.couplings),
        G_background=np.asarray(result.background.G),
        mu_background=float(result.background.mu),
        Gerr_background=float(Gerr_bg),
        Gerr_embedded=float(Gerr_emb),
        mu_ed=float(mu_ed),
        G_ed=np.asarray(G_ed),
        benchmark_status=str(benchmark_status),
        background_cache_path=str(cache_path if use_cache else ""),
    )
    print(f"saved {outfile}")


if __name__ == "__main__":
    main()
