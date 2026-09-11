#!/usr/bin/env python3
"""Run self-consistent 6-site cluster ED + GW on an Lx x Ly Ruby torus."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from rubycgw.cluster_ed_gw_fast import (
    ClusterEDGWFastOptions,
    solve_cluster_ed_gw_fast,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions, GWResult
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


_CACHE_VERSION = 1


class _ClusterEDGWLogFilter:
    """Pass through normal stdout, but keep only residual lines from cluster ED+GW."""

    def __init__(self, stream):
        self.stream = stream
        self._buffer = ""

    def write(self, text):
        text = str(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if (
                not line.startswith("[cluster-ED+GW]")
                or (" outer " in line and "residual=" in line)
            ):
                self.stream.write(line + "\n")
        return len(text)

    def flush(self):
        self.stream.flush()


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max", type=int, default=160)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")

    # The coupled cluster/impurity fixed point is smooth but appreciably slower
    # than the initial lattice GW solve.  The V=1, L=2x1 benchmark contracts
    # monotonically with no Pulay safeguard hits, so use a less conservative
    # Pulay damping and leave enough outer iterations to reach the requested tol.
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
    p.add_argument("--quiet-gw", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/cluster_ed_gw"))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="cache directory; default is OUT/cache",
    )
    p.add_argument("--no-cache", action="store_true", help="disable ED/GW cache reads and writes")
    p.add_argument("--refresh-cache", action="store_true", help="ignore old cache entries and overwrite them")
    return p.parse_args()


def _lattice_to_realspace(Gk: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Fourier-transform 6x6 G(k) to the finite-torus site basis."""
    arr = np.asarray(Gk, dtype=complex)
    nf = int(arr.shape[0])
    ncell = int(Lx) * int(Ly)
    out = np.zeros((nf, 6 * ncell, 6 * ncell), dtype=complex)
    cells = [(r1, r2) for r1 in range(int(Lx)) for r2 in range(int(Ly))]
    for c, (r1, r2) in enumerate(cells):
        for d, (s1, s2) in enumerate(cells):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for i in range(int(Lx)):
                for j in range(int(Ly)):
                    phase = np.exp(
                        2j * np.pi * (
                            (i / float(Lx)) * (r1 - s1)
                            + (j / float(Ly)) * (r2 - s2)
                        )
                    )
                    block += phase * arr[:, i, j]
            block /= float(ncell)
            out[:, 6*c:6*(c+1), 6*d:6*(d+1)] = block
    return out


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _hash_payload(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _physics_payload(args, grid: MatsubaraGrid) -> dict:
    return {
        "version": _CACHE_VERSION,
        "Lx": int(args.Lx),
        "Ly": int(args.Ly),
        "V": float(args.V),
        "filling": float(args.filling),
        "T": float(args.T),
        "ti": float(args.ti),
        "t1": float(args.t1),
        "t2": float(args.t2),
        "omega": [float(x) for x in grid.omega],
        "Omega": [float(x) for x in grid.Omega],
    }


def _background_cache_path(cache_dir: Path, args, grid: MatsubaraGrid) -> Path:
    payload = _physics_payload(args, grid) | {
        "kind": "sc_gw_background",
        "gw_tol": float(args.gw_tol),
    }
    return cache_dir / f"gw_{_hash_payload(payload)}.npz"


def _ed_cache_path(cache_dir: Path, args, grid: MatsubaraGrid) -> Path:
    payload = _physics_payload(args, grid) | {"kind": "exact_ed_green"}
    return cache_dir / f"ed_{_hash_payload(payload)}.npz"


def _save_gw_cache(path: Path, result: GWResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=_CACHE_VERSION,
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


def _load_gw_cache(path: Path) -> GWResult:
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


def _get_exact_ed(args, params, grid, cache_dir: Path, use_cache: bool):
    path = _ed_cache_path(cache_dir, args, grid)
    if use_cache and path.exists() and not args.refresh_cache:
        try:
            with np.load(path, allow_pickle=False) as z:
                mu_ed = float(z["mu_ed"])
                G_ed = np.asarray(z["G_ed"])
            if G_ed.shape == (grid.nf, 6 * args.Lx * args.Ly, 6 * args.Lx * args.Ly):
                print(f"[benchmark] ED cache hit: {path}", flush=True)
                return mu_ed, G_ed, path
        except Exception as exc:
            print(f"[benchmark] ignoring unreadable ED cache {path}: {exc}", flush=True)

    print("[benchmark] exact finite-torus ED ...", flush=True)
    exact = ExactSmallRubyThermal(int(args.Lx), int(args.Ly), params)
    exact.diagonalize(float(args.V))
    mu_ed = exact.solve_mu(float(args.filling) * args.Lx * args.Ly, float(args.T))
    G_ed, _ = exact.green_iomega(1j * grid.omega, mu_ed, float(args.T))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            cache_version=_CACHE_VERSION,
            mu_ed=float(mu_ed),
            G_ed=np.asarray(G_ed),
        )
        print(f"[benchmark] ED cache saved: {path}", flush=True)
    return float(mu_ed), np.asarray(G_ed), path


def main():
    args = _args()
    if args.Lx < 1 or args.Ly < 1:
        raise ValueError("Lx and Ly must be positive")
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=int(args.Lx),
        nk2=int(args.Ly),
        nw=int(args.nw),
        nOmega=int(args.nomega),
        T=float(args.T),
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir is not None else args.out / "cache"
    use_cache = not bool(args.no_cache)

    gw_opts = GWOptions(
        target_filling=float(args.filling),
        max_iter=int(args.gw_max),
        tol=float(args.gw_tol),
        mixing=float(args.gw_mixing),
        mixing_method=str(args.gw_mixing_method),
        verbose=not bool(args.quiet_gw),
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

    background = None
    gw_cache = _background_cache_path(cache_dir, args, grid)
    if use_cache and gw_cache.exists() and not args.refresh_cache:
        try:
            candidate = _load_gw_cache(gw_cache)
            if candidate.converged and candidate.final_error <= max(float(args.gw_tol), 1e-12) * 1.05:
                background = candidate
                print(
                    f"[cache] SC-GW hit: {gw_cache} "
                    f"(residual={candidate.final_error:.3e})",
                    flush=True,
                )
        except Exception as exc:
            print(f"[cache] ignoring unreadable SC-GW cache {gw_cache}: {exc}", flush=True)

    print("=== Ruby 6-site cluster ED + lattice GW ===", flush=True)
    print(
        f"L={args.Lx}x{args.Ly}, V={args.V:g}, filling={args.filling:g}, "
        f"T={args.T:g}, nw={args.nw}, nOmega={args.nomega}, nbath={args.nbath}, "
        f"embed_mix={args.embed_mixing_method}",
        flush=True,
    )
    log_filter = _ClusterEDGWLogFilter(sys.stdout)
    with redirect_stdout(log_filter):
        result = solve_cluster_ed_gw_fast(
            h0,
            Vq,
            params,
            grid,
            gw_opts=gw_opts,
            embed_opts=embed_opts,
            background=background,
        )
    log_filter.flush()
    if use_cache and (background is None or args.refresh_cache):
        _save_gw_cache(gw_cache, result.background)
        print(f"[cache] SC-GW saved: {gw_cache}", flush=True)

    print(
        f"final: converged={result.converged}, iterations={result.iterations}, "
        f"residual={result.final_error:.3e}, impurity_mismatch={result.impurity_mismatch:.3e}, "
        f"bath_fit={result.bath_fit_error:.3e}, mu={result.mu:+.10f}, "
        f"mix={result.mixing_method}, pulay_fallbacks={result.pulay_fallbacks}",
        flush=True,
    )

    Gerr_bg = Gerr_emb = np.nan
    mu_ed = np.nan
    G_ed = np.empty((0,), dtype=complex)
    ed_cache = None
    if args.benchmark_ed:
        nsites = 6 * int(args.Lx) * int(args.Ly)
        if nsites > 16:
            raise ValueError("--benchmark-ed is limited to <=16 sites by ExactSmallRubyThermal")
        mu_ed, G_ed, ed_cache = _get_exact_ed(
            args, params, grid, cache_dir, use_cache
        )
        G_bg_real = _lattice_to_realspace(result.background.G, args.Lx, args.Ly)
        G_emb_real = _lattice_to_realspace(result.G, args.Lx, args.Ly)
        Gerr_bg = _relerr(G_bg_real, G_ed)
        Gerr_emb = _relerr(G_emb_real, G_ed)
        print(
            f"[benchmark] mu_ED={mu_ed:+.10f}, Gerr_GW={Gerr_bg:.6e}, "
            f"Gerr_clusterED+GW={Gerr_emb:.6e}, ratio={Gerr_emb/Gerr_bg:.6f}",
            flush=True,
        )

    outfile = args.out / (
        f"cluster_ed_gw_L{args.Lx}x{args.Ly}_V{args.V:.6g}_fill{args.filling:.6g}.npz"
    )
    np.savez_compressed(
        outfile,
        Lx=int(args.Lx),
        Ly=int(args.Ly),
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
        Sigma_GW_lattice=np.asarray(result.Sigma_GW_lattice),
        Sigma_GW_cluster=np.asarray(result.Sigma_GW_cluster),
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
        gw_cache_path=str(gw_cache if use_cache else ""),
        ed_cache_path=str(ed_cache if (use_cache and ed_cache is not None) else ""),
    )
    print(f"saved {outfile}", flush=True)


if __name__ == "__main__":
    main()
