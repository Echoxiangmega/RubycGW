#!/usr/bin/env python3
"""Project a Fierz-GW pseudo-arclength branch to exact V and evaluate cGW chi.

The PAC driver intentionally follows the solution manifold and may step past a
requested V.  This post-processor loads the last two PAC states, interpolates a
branch-aware initial guess at the requested V (normally V=1), refines the full
GW fixed point at exactly that V with matrix-free Newton-Krylov, then evaluates
q=0 covariant current susceptibilities.  Matching ED data are loaded from the
persistent cache when available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.ed_benchmark_cache import load_ed_cache, save_ed_cache
from rubycgw.fierz_channel_gw import (
    ChannelGWResult,
    build_channel_definition,
    channel_static_self_energy,
    solve_channel_vertex_q0,
    susceptibility_from_vertex_q0,
)
from rubycgw.fierz_pac import FierzGWResidual, soft_mode_overlap
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import SupercellVertexOptions


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--V", type=float, default=1.0, help="exact target interaction")
    p.add_argument("--newton-tol", type=float, default=1e-10)
    p.add_argument("--newton-max", type=int, default=18)
    p.add_argument("--fd-eps", type=float, default=3e-7)
    p.add_argument("--gmres-rtol", type=float, default=5e-4)
    p.add_argument("--gmres-maxiter", type=int, default=80)
    p.add_argument("--gmres-restart", type=int, default=20)
    p.add_argument("--line-search-min", type=float, default=1.0 / 4096.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--vertex-max-iter", type=int, default=300)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=16)
    p.add_argument("--low-count", type=int, default=8)
    p.add_argument("--skip-ed", action="store_true")
    p.add_argument("--refresh-ed-cache", action="store_true")
    p.add_argument("--ed-cache-dir", type=Path, default=Path("results/ed_cache"))
    p.add_argument("--out", type=Path, default=Path("results/ed_fierz_channel_cgw"))
    p.add_argument("--verbose", action="store_true")
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


def _ed_signature(meta, V):
    return {
        "L1": int(meta["L1"]), "L2": int(meta["L2"]), "V": float(V),
        "filling": float(meta["filling"]), "T": float(meta["T"]),
        "ti": float(meta["ti"]), "t1": float(meta["t1"]), "t2": float(meta["t2"]),
        "nw": int(meta["nw"]), "channels": list(CHANNELS),
    }


def _make_background(problem, definition, x, ev, root):
    sigma_static, sigma_c, mu = problem.codec.decode(x)
    _, tad, exchange = channel_static_self_energy(ev.rho, definition)
    return ChannelGWResult(
        G=np.asarray(ev.G), W=np.asarray(ev.W), P=np.asarray(ev.P),
        Sigma_static=np.asarray(sigma_static),
        Sigma_tadpole=np.asarray(tad), Sigma_exchange=np.asarray(exchange),
        Sigma_c=np.asarray(sigma_c), mu=float(mu), rho=np.asarray(ev.rho),
        density=np.real(np.diag(ev.rho)), converged=bool(root.converged),
        iterations=int(root.newton_iterations), final_error=float(ev.physical_residual),
        mixing_method="newton-krylov", mode=definition.mode,
        min_screening_singular_value=float(ev.smin),
    )


def main():
    args = _args()
    data = np.load(args.checkpoint, allow_pickle=False)
    meta = json.loads(str(np.asarray(data["signature"]).item()))
    required = ("mode", "L1", "L2", "filling", "target", "T", "ti", "t1", "t2", "nw", "nomega", "norb")
    missing = [k for k in required if k not in meta]
    if missing:
        raise RuntimeError(f"PAC checkpoint signature missing keys: {missing}")

    x_prev = np.asarray(data["x_prev"], dtype=float)
    x_curr = np.asarray(data["x_curr"], dtype=float)
    V_prev = float(data["V_prev"])
    V_curr = float(data["V_curr"])
    V_target = float(args.V)
    if abs(V_curr - V_prev) < 1e-14:
        raise RuntimeError("last two checkpoint V values are identical; cannot form target guess")

    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    norb = int(geometry.n_sites)
    if norb != int(meta["norb"]):
        raise RuntimeError("checkpoint orbital count does not match reconstructed geometry")
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(meta["nw"]), nOmega=int(meta["nomega"]), T=float(meta["T"])
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    problem = FierzGWResidual(
        h0, geometry.interaction_pairs, str(meta["mode"]), grid, float(meta["target"])
    )
    if x_prev.size != problem.codec.size or x_curr.size != problem.codec.size:
        raise RuntimeError("checkpoint state size does not match reconstructed PAC problem")

    alpha = (V_target - V_prev) / (V_curr - V_prev)
    x_guess = x_prev + alpha * (x_curr - x_prev)
    bracketed = min(V_prev, V_curr) <= V_target <= max(V_prev, V_curr)
    print("=== Exact-target refinement from Fierz PAC branch ===")
    print(
        f"mode={meta['mode']}, target V={V_target:.12g}, "
        f"last segment=({V_prev:.12g},{V_curr:.12g}), alpha={alpha:.6f}, "
        f"bracketed={'yes' if bracketed else 'no'}"
    )
    if not bracketed:
        print("WARNING: target V is outside the last PAC segment; using secant extrapolation")

    def guarded_residual(x, V):
        ev = problem.evaluate(x, V)
        if ev.smin < float(args.screening_floor):
            raise FloatingPointError("screening matrix below numerical floor")
        return ev.residual

    opts = PACOptions(
        tol=args.newton_tol, max_newton=args.newton_max, fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol, gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart, line_search_min=args.line_search_min,
        verbose=args.verbose,
    )
    root = refine_fixed_parameter(x_guess, V_target, guarded_residual, opts=opts)
    if not root.converged:
        raise RuntimeError(
            f"fixed-V Newton-Krylov failed: |R_scaled|={root.residual_norm:.3e}; "
            "increase --newton-max/--gmres-maxiter or tighten the PAC endpoint around target V"
        )

    ev = problem.evaluate(root.x, V_target)
    definition = build_channel_definition(
        geometry.interaction_pairs, norb, V_target, str(meta["mode"])
    )
    bg = _make_background(problem, definition, root.x, ev, root)
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])
    overlap = [soft_mode_overlap(K, definition, ev.soft_mode) for K in operators]
    print(
        f"exact V={V_target:.12g}: Newton={root.newton_iterations}, GMRES={root.gmres_iterations}, "
        f"|R_scaled|={root.residual_norm:.3e}, phys={ev.physical_residual:.3e}, "
        f"Nerr={ev.filling_error:+.3e}, mu={bg.mu:+.10f}, smin={ev.smin:.3e}"
    )
    print(
        f"soft mode: m={ev.soft_m:+d}, Omega={ev.soft_Omega:+.6g}, "
        f"overlap(same,opp)=({overlap[0]:.6f},{overlap[1]:.6f})"
    )

    vopts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter, tol=args.vertex_tol, solver="gmres",
        gmres_restart=args.vertex_gmres_restart, verbose=args.verbose,
        momentum_backend="direct",
    )
    chi = np.zeros((len(CHANNELS), len(CHANNELS)), dtype=complex)
    vertex_conv = []
    vertex_res = []
    for b, (name, Ksrc) in enumerate(zip(CHANNELS, operators)):
        print(f"cGW vertex: {name} ...")
        vr = solve_channel_vertex_q0(bg, definition, Ksrc, grid, opts=vopts)
        vertex_conv.append(vr.converged)
        vertex_res.append(vr.final_error)
        if not vr.converged:
            raise RuntimeError(f"cGW vertex {name} failed: residual={vr.final_error:.3e}")
        for a, Kleft in enumerate(operators):
            chi[a, b] = susceptibility_from_vertex_q0(bg.G, Kleft, vr.Gamma, grid)
    chi_phys, chi_imag = _physical_chi(chi)
    print(
        f"cGW chi at exact V: same={chi_phys[0,0]:+.9f}, "
        f"opposite={chi_phys[1,1]:+.9f}, hermitian-imag={chi_imag:.3e}, "
        f"vertex max residual={max(vertex_res):.3e}"
    )

    Ged = None
    chi_ed = None
    mu_ed = np.nan
    gerr = np.nan
    gerr_low = np.nan
    chi_err = np.nan
    ed_cache_path = ""
    if not args.skip_ed:
        ed_sig = _ed_signature(meta, V_target)
        cached = None if args.refresh_ed_cache else load_ed_cache(
            args.ed_cache_dir, ed_sig, np.asarray(grid.omega), norb
        )
        if cached is not None:
            mu_ed = float(cached["mu_ed"])
            Ged = np.asarray(cached["G_ed"], dtype=complex)
            chi_ed = np.asarray(cached["chi_ed"], dtype=float)
            ed_cache_path = str(cached["path"])
            print(f"ED cache hit: {ed_cache_path}")
        else:
            print("ED cache miss; computing exact reference at target V ...")
            geometry.diagonalize(V_target)
            mu_ed = geometry.solve_mu(float(meta["target"]), float(meta["T"]))
            chi_ed, _ = geometry.static_susceptibility_matrix(operators, mu_ed, float(meta["T"]))
            chi_ed = np.asarray(chi_ed, dtype=float)
            Ged, _ = geometry.green_iomega(1j * np.asarray(grid.omega), mu_ed, float(meta["T"]))
            Ged = np.asarray(Ged, dtype=complex)
            ed_cache_path = str(save_ed_cache(
                args.ed_cache_dir, ed_sig, np.asarray(grid.omega), mu_ed, Ged, chi_ed
            ))
            print(f"ED cache saved: {ed_cache_path}")
        gerr, gerr_low = _green_errors(bg.G, Ged, grid, args.low_count)
        chi_err = _relerr(chi_phys, chi_ed)
        print(
            f"ED comparison: Gerr={gerr:.6e}, Gerr_low={gerr_low:.6e}, "
            f"chi_ED=({chi_ed[0,0]:.9f},{chi_ed[1,1]:.9f}), chi_relerr={chi_err:.6e}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"V{V_target:g}_fill{float(meta['filling']):g}_{int(meta['L1'])}x{int(meta['L2'])}_{str(meta['mode']).lower()}_from_pac"
    outfile = args.out / f"{tag}.npz"
    sigma_static, sigma_c, mu = problem.codec.decode(root.x)
    np.savez_compressed(
        outfile,
        V=V_target, mode=np.asarray(str(meta["mode"])), filling=float(meta["filling"]),
        T=float(meta["T"]), L1=int(meta["L1"]), L2=int(meta["L2"]),
        omega=np.asarray(grid.omega), Omega=np.asarray(grid.Omega),
        x_target=np.asarray(root.x), Sigma_static=np.asarray(sigma_static),
        Sigma_c=np.asarray(sigma_c), G=np.asarray(bg.G), P=np.asarray(bg.P), W=np.asarray(bg.W),
        rho=np.asarray(bg.rho), mu=float(mu), smin=float(ev.smin),
        soft_m=int(ev.soft_m), soft_Omega=float(ev.soft_Omega), soft_mode=np.asarray(ev.soft_mode),
        soft_overlap_same=float(overlap[0]), soft_overlap_opposite=float(overlap[1]),
        fixed_scaled_residual=float(root.residual_norm), physical_residual=float(ev.physical_residual),
        filling_error=float(ev.filling_error), newton_iterations=int(root.newton_iterations),
        gmres_iterations=int(root.gmres_iterations), chi=np.asarray(chi_phys), chi_raw=np.asarray(chi),
        vertex_converged=np.asarray(vertex_conv, dtype=bool), vertex_residual=np.asarray(vertex_res),
        G_ed=np.asarray(Ged) if Ged is not None else np.empty((0,), dtype=complex),
        chi_ed=np.asarray(chi_ed) if chi_ed is not None else np.empty((0,), dtype=float),
        mu_ed=float(mu_ed), gerr=float(gerr), gerr_low=float(gerr_low), chi_err=float(chi_err),
        ed_cache_path=np.asarray(ed_cache_path), source_checkpoint=np.asarray(str(args.checkpoint)),
    )
    print(f"saved {outfile}")


if __name__ == "__main__":
    main()
