#!/usr/bin/env python3
"""Compare density/B/J/B+J GW-cGW Fierz representations against 2x1 ED.

The microscopic Hamiltonian is unchanged. Only the GW resummation channel is
changed. This is therefore a direct diagnostic of Fierz/channel ambiguity in
the strong-coupling Ruby current response.

Expensive ED reference data are cached persistently.  If the geometry,
Hamiltonian, filling, temperature, and fermionic Matsubara grid are unchanged,
a later run skips ED diagonalization and immediately proceeds to the requested
GW/cGW modes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from rubycgw.ed_benchmark_cache import load_ed_cache, save_ed_cache
from rubycgw.fierz_channel_gw import (
    build_channel_definition,
    solve_channel_gw_same_torus,
    solve_channel_vertex_q0,
    susceptibility_from_vertex_q0,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import SupercellVertexOptions


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--low-count", type=int, default=8)
    p.add_argument(
        "--modes", nargs="+", default=("density", "J", "B", "BJ"),
        help="any subset of density J B BJ",
    )
    p.add_argument("--max-iter", type=int, default=1600)
    p.add_argument("--tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.06)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=8)
    p.add_argument("--pulay-start", type=int, default=4)
    p.add_argument("--pulay-regularization", type=float, default=1e-9)
    p.add_argument("--vertex-max-iter", type=int, default=220)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--gmres-restart", type=int, default=14)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--ed-cache-dir", type=Path, default=Path("results/ed_cache"),
        help="persistent directory for exact ED reference data",
    )
    p.add_argument(
        "--refresh-ed-cache", action="store_true",
        help="ignore an existing matching ED cache entry and recompute it",
    )
    p.add_argument(
        "--out", type=Path, default=Path("results/ed_fierz_channel_cgw")
    )
    return p.parse_args()


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _green_errors(G, Ged, grid, low_count):
    arr = np.asarray(G, dtype=complex)
    if arr.ndim == 5:
        arr = arr[:, 0, 0]
    exact = np.asarray(Ged, dtype=complex)
    idx = np.argsort(np.abs(np.asarray(grid.omega)))[:min(int(low_count), grid.nf)]
    return _relerr(arr, exact), _relerr(arr[idx], exact[idx])


def _physical_chi(chi):
    herm = 0.5 * (np.asarray(chi) + np.asarray(chi).conj().T)
    return np.asarray(herm.real, dtype=float), float(np.max(np.abs(herm.imag)))


def _ed_signature(args):
    """Parameters that actually determine the cached ED reference data."""
    return {
        "L1": int(args.L1),
        "L2": int(args.L2),
        "V": float(args.V),
        "filling": float(args.filling),
        "T": float(args.T),
        "ti": float(args.ti),
        "t1": float(args.t1),
        "t2": float(args.t2),
        "nw": int(args.nw),
        "channels": list(CHANNELS),
    }


def main():
    args = _args()
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal supports at most 16 sites")
    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    # Constructing the geometry is cheap and is still needed by GW.  The costly
    # sector-by-sector diagonalization is done only on an ED cache miss.
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    norb = int(exact.n_sites)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    h0 = np.asarray(exact.h0, dtype=complex)[None, None]
    operators = np.stack([
        np.asarray(exact.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])

    ed_signature = _ed_signature(args)
    ed_cached = None
    if not args.refresh_ed_cache:
        ed_cached = load_ed_cache(
            args.ed_cache_dir,
            ed_signature,
            np.asarray(grid.omega),
            norb,
        )

    if ed_cached is not None:
        mu_ed = float(ed_cached["mu_ed"])
        Ged = np.asarray(ed_cached["G_ed"], dtype=complex)
        chi_ed = np.asarray(ed_cached["chi_ed"], dtype=float)
        ed_cache_path = Path(ed_cached["path"])
        ed_cache_hit = True
        print(f"ED cache hit: {ed_cache_path}")
        print("skipping exact diagonalization / ED susceptibility / ED Green function")
    else:
        if args.refresh_ed_cache:
            print("ED cache refresh requested; recomputing exact reference ...")
        else:
            print("ED cache miss; computing exact reference ...")
        exact.diagonalize(float(args.V))
        mu_ed = exact.solve_mu(target, args.T)
        chi_ed, _ = exact.static_susceptibility_matrix(operators, mu_ed, args.T)
        chi_ed = np.asarray(chi_ed, dtype=float)
        Ged, _ = exact.green_iomega(1j * np.asarray(grid.omega), mu_ed, args.T)
        Ged = np.asarray(Ged, dtype=complex)
        ed_cache_path = save_ed_cache(
            args.ed_cache_dir,
            ed_signature,
            np.asarray(grid.omega),
            mu_ed,
            Ged,
            chi_ed,
        )
        ed_cache_hit = False
        print(f"ED cache saved: {ed_cache_path}")

    gw_opts = GWOptions(
        target_filling=target,
        max_iter=args.max_iter,
        tol=args.tol,
        mixing=args.mixing,
        mixing_method=args.mixing_method,
        pulay_history=args.pulay_history,
        pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,
        verbose=args.verbose,
        momentum_backend="direct",
    )
    vopts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        solver="gmres",
        gmres_restart=args.gmres_restart,
        verbose=args.verbose,
        momentum_backend="direct",
    )

    print("=== ED / Fierz-channel GW-cGW benchmark ===")
    print(
        f"torus={args.L1}x{args.L2} ({norb} sites), V={args.V:g}, "
        f"filling={args.filling:g}, T={args.T:g}, nw={args.nw}, nOmega={args.nomega}"
    )
    print(
        f"ED chi: same={chi_ed[0,0]:.9f}, "
        f"opposite={chi_ed[1,1]:.9f}"
    )

    results = {}
    save = {
        "V": args.V,
        "filling": args.filling,
        "T": args.T,
        "L1": args.L1,
        "L2": args.L2,
        "omega": np.asarray(grid.omega),
        "Omega": np.asarray(grid.Omega),
        "mu_ed": mu_ed,
        "G_ed": np.asarray(Ged),
        "chi_ed": chi_ed,
        "ed_cache_hit": bool(ed_cache_hit),
        "ed_cache_path": np.asarray(str(ed_cache_path)),
    }

    for mode_in in args.modes:
        definition = build_channel_definition(
            exact.interaction_pairs, norb, args.V, mode_in
        )
        mode = definition.mode
        print(f"\n--- mode={mode}  nch={len(definition.labels)} ---")
        bg = solve_channel_gw_same_torus(h0, definition, grid, opts=gw_opts)
        if not bg.converged and not args.allow_unconverged:
            raise RuntimeError(
                f"{mode} GW failed: residual={bg.final_error:.3e}"
            )
        gerr, gerr_low = _green_errors(bg.G, Ged, grid, args.low_count)
        print(
            f"GW background: {'OK' if bg.converged else 'FAIL'} "
            f"iter={bg.iterations} r={bg.final_error:.3e} "
            f"mu={bg.mu:+.10f} smin={bg.min_screening_singular_value:.3e}"
        )
        print(f"Gerr={gerr:.6e}, Gerr_low={gerr_low:.6e}")

        chi = np.zeros((len(CHANNELS), len(CHANNELS)), dtype=complex)
        vertex_conv = []
        vertex_res = []
        for b, (name, Ksrc) in enumerate(zip(CHANNELS, operators)):
            vr = solve_channel_vertex_q0(
                bg, definition, Ksrc, grid, opts=vopts
            )
            vertex_conv.append(vr.converged)
            vertex_res.append(vr.final_error)
            if not vr.converged and not args.allow_unconverged:
                raise RuntimeError(
                    f"{mode} cGW vertex {name} failed: residual={vr.final_error:.3e}"
                )
            for a, Kleft in enumerate(operators):
                chi[a, b] = susceptibility_from_vertex_q0(
                    bg.G, Kleft, vr.Gamma, grid
                )

        chi_phys, imag = _physical_chi(chi)
        chi_err = _relerr(chi_phys, chi_ed)
        print(
            f"cGW chi same={chi_phys[0,0]:+.9f} "
            f"(ED {chi_ed[0,0]:+.9f}, "
            f"rel={(chi_phys[0,0]-chi_ed[0,0])/max(abs(chi_ed[0,0]),1e-300):+.2%})"
        )
        print(
            f"cGW chi opposite={chi_phys[1,1]:+.9f} "
            f"(ED {chi_ed[1,1]:+.9f}, "
            f"rel={(chi_phys[1,1]-chi_ed[1,1])/max(abs(chi_ed[1,1]),1e-300):+.2%})"
        )
        print(
            f"chi matrix relerr={chi_err:.6e}, hermitian-imag={imag:.3e}, "
            f"vertex max residual={max(vertex_res):.3e}"
        )

        results[mode] = dict(
            gerr=gerr,
            gerr_low=gerr_low,
            chi=chi_phys,
            chi_err=chi_err,
            mu=bg.mu,
            smin=bg.min_screening_singular_value,
        )
        key = mode.replace("+", "_")
        save[f"G_{key}"] = np.asarray(bg.G)
        save[f"mu_{key}"] = bg.mu
        save[f"gerr_{key}"] = gerr
        save[f"gerr_low_{key}"] = gerr_low
        save[f"chi_{key}"] = chi_phys
        save[f"chi_raw_{key}"] = chi
        save[f"chi_err_{key}"] = chi_err
        save[f"screening_smin_{key}"] = bg.min_screening_singular_value
        save[f"gw_converged_{key}"] = bg.converged
        save[f"gw_residual_{key}"] = bg.final_error
        save[f"vertex_converged_{key}"] = np.asarray(vertex_conv, dtype=bool)
        save[f"vertex_residual_{key}"] = np.asarray(vertex_res, dtype=float)

    print("\n=== compact comparison ===")
    print("mode          Gerr        Gerr_low     chi_same     chi_opp      chi_relerr")
    for mode, r in results.items():
        print(
            f"{mode:8s}  {r['gerr']:10.6f}  {r['gerr_low']:10.6f}  "
            f"{r['chi'][0,0]:11.6f}  {r['chi'][1,1]:11.6f}  {r['chi_err']:10.6f}"
        )

    if len(results) > 1:
        same = np.array([r["chi"][0,0] for r in results.values()])
        opp = np.array([r["chi"][1,1] for r in results.values()])
        print(
            f"\nFierz spread: chi_same range={same.min():.6f}..{same.max():.6f}, "
            f"chi_opp range={opp.min():.6f}..{opp.max():.6f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / (
        f"V{args.V:g}_fill{args.filling:g}_{args.L1}x{args.L2}_fierz_channels.npz"
    )
    np.savez_compressed(outfile, **save)
    print(f"\nsaved {outfile}")


if __name__ == "__main__":
    main()
