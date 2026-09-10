#!/usr/bin/env python3
"""Refine transverse Fierz-PMS sign changes and evaluate cGW response.

Input is one ``*_transverse.npz`` produced by ``scan_fierz_pms_transverse.py``.
For every sign change of dF/da, interpolate the stored fermionic state and
(s,a), then solve at fixed physical V the augmented square system

    R_GW = 0,
    dF/ds = 0,
    dF/da = 0.

The converged state is an interior stationary point in the full n/B/J Fierz
simplex.  At that same two-dimensional PMS root this script evaluates the GW
Luttinger-Ward Helmholtz free energy and the q=0 covariant-GW current response
for ``z_same`` and ``z_opposite``.  ED reference values are reused from the
upstream target NPZ when available; ED is not recomputed here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_channel_gw import (
    ChannelGWResult,
    channel_static_self_energy,
    solve_channel_vertex_q0,
    susceptibility_from_vertex_q0,
)
from rubycgw.fierz_mixed import build_weighted_nbj_definition
from rubycgw.fierz_pms_full import FullSimplexFierzPMSResidual
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import SupercellVertexOptions


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transverse-npz", type=Path, required=True)
    p.add_argument("--crossing", type=int, default=None, help="refine only this sign-change index")
    p.add_argument("--newton-tol", type=float, default=1e-10)
    p.add_argument("--newton-max", type=int, default=24)
    p.add_argument("--fd-eps", type=float, default=2e-7)
    p.add_argument("--gmres-rtol", type=float, default=3e-4)
    p.add_argument("--gmres-maxiter", type=int, default=120)
    p.add_argument("--gmres-restart", type=int, default=24)
    p.add_argument("--line-search-min", type=float, default=1.0 / 8192.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--fd-h", type=float, default=None)
    p.add_argument("--simplex-margin", type=float, default=1e-8)
    p.add_argument("--vertex-max-iter", type=int, default=300)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=16)
    p.add_argument(
        "--skip-chi", action="store_true",
        help="refine the full PMS root and free energy but skip the cGW vertex solve",
    )
    p.add_argument("--out", type=Path, default=Path("results/fierz_pms_full"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_meta(transverse_npz: Path):
    with np.load(transverse_npz, allow_pickle=False) as d:
        V = float(np.asarray(d["V"]).item())
        root_index = int(np.asarray(d["root"]).item())
        a = np.asarray(d["a"], dtype=float)
        s = np.asarray(d["s"], dtype=float)
        Fa = np.asarray(d["dF_da_per_cell"], dtype=float)
        X = np.asarray(d["X_base"], dtype=float)
        source_target = Path(str(np.asarray(d["source_target_npz"]).item()))
    if not source_target.exists():
        raise RuntimeError(f"source target NPZ not found: {source_target}")

    with np.load(source_target, allow_pickle=False) as d:
        source_checkpoint = Path(str(np.asarray(d["source_checkpoint"]).item()))
        F_ed_pc = float(np.asarray(d["F_ed_per_cell"]).item()) if "F_ed_per_cell" in d else np.nan
        chi_ed = np.asarray(d["chi_ed"], dtype=float) if "chi_ed" in d else np.empty((0,), dtype=float)
    if not source_checkpoint.exists():
        raise RuntimeError(f"source checkpoint not found: {source_checkpoint}")
    with np.load(source_checkpoint, allow_pickle=False) as d:
        signature = str(np.asarray(d["signature"]).item())
    meta = json.loads(signature)
    return V, root_index, a, s, Fa, X, source_target, meta, F_ed_pc, chi_ed


def _crossings(a, Fa):
    out = []
    for i in range(len(a) - 1):
        f0, f1 = float(Fa[i]), float(Fa[i + 1])
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0 or f1 == 0.0 or np.signbit(f0) != np.signbit(f1):
            out.append((i, i + 1))
    return out


def _reconstruct(meta, V, fd_h, simplex_margin):
    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(meta["nw"]), nOmega=int(meta["nomega"]), T=float(meta["T"])
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    npc = int(meta["L1"]) * int(meta["L2"])
    h = float(meta["lambda_fd_h"]) if fd_h is None else float(fd_h)
    problem = FullSimplexFierzPMSResidual(
        h0,
        geometry.interaction_pairs,
        grid,
        float(meta["target"]),
        V=float(V),
        primitive_cells=npc,
        fd_h=h,
        simplex_margin=float(simplex_margin),
        free_energy_scale_floor=float(meta["pms_scale_floor"]),
    )
    return problem, geometry, grid


def _make_background(problem, definition, ev, root, sigma_static, sigma_c, mu):
    """Build the exact ChannelGWResult corresponding to one full-PMS root."""
    base_ev = ev.longitudinal.base_evaluation
    _, tad, exchange = channel_static_self_energy(base_ev.rho, definition)
    return ChannelGWResult(
        G=np.asarray(base_ev.G),
        W=np.asarray(base_ev.W),
        P=np.asarray(base_ev.P),
        Sigma_static=np.asarray(sigma_static),
        Sigma_tadpole=np.asarray(tad),
        Sigma_exchange=np.asarray(exchange),
        Sigma_c=np.asarray(sigma_c),
        mu=float(mu),
        rho=np.asarray(base_ev.rho),
        density=np.real(np.diag(np.asarray(base_ev.rho))),
        converged=bool(root.converged),
        iterations=int(root.newton_iterations),
        final_error=float(ev.physical_residual),
        mixing_method="newton-krylov-full-pms",
        mode=definition.mode,
        min_screening_singular_value=float(ev.smin),
    )


def _physical_chi(chi):
    herm = 0.5 * (np.asarray(chi) + np.asarray(chi).conj().T)
    return np.asarray(herm.real, dtype=float), float(np.max(np.abs(herm.imag)))


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = _args()
    (
        V, root_index, a, s, Fa, X, source_target, meta, F_ed_pc, chi_ed
    ) = _load_meta(args.transverse_npz)
    if X.shape[0] != len(a) or len(s) != len(a) or len(Fa) != len(a):
        raise RuntimeError("transverse arrays have inconsistent lengths")
    brackets = _crossings(a, Fa)
    if not brackets:
        raise RuntimeError("no dF/da sign change found in transverse NPZ")
    if args.crossing is not None:
        if args.crossing < 0 or args.crossing >= len(brackets):
            raise IndexError(f"crossing {args.crossing} outside [0,{len(brackets)-1}]")
        brackets = [brackets[args.crossing]]

    problem, geometry, grid = _reconstruct(meta, V, args.fd_h, args.simplex_margin)
    norb = int(geometry.n_sites)
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])

    opts = PACOptions(
        tol=args.newton_tol,
        max_newton=args.newton_max,
        fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol,
        gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart,
        line_search_min=args.line_search_min,
        verbose=args.verbose,
    )
    vopts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        solver="gmres",
        gmres_restart=args.vertex_gmres_restart,
        verbose=args.verbose,
        momentum_backend="direct",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = []
    print("=== Exact full-simplex Fierz-PMS refinement + cGW response ===")
    print(f"source={args.transverse_npz}")
    print(f"V={V:g}, source root={root_index}, sign-change candidates={len(brackets)}")
    if chi_ed.shape == (2, 2):
        print(f"ED reference chi=({chi_ed[0,0]:.9f},{chi_ed[1,1]:.9f})")
    if np.isfinite(F_ed_pc):
        print(f"ED reference F/cell={F_ed_pc:+.12e}")

    for ic, (i0, i1) in enumerate(brackets):
        a0, a1 = float(a[i0]), float(a[i1])
        f0, f1 = float(Fa[i0]), float(Fa[i1])
        alpha = 0.5 if abs(f1 - f0) < 1e-30 else float(np.clip(-f0 / (f1 - f0), 0.0, 1.0))
        ag = (1.0 - alpha) * a0 + alpha * a1
        sg = (1.0 - alpha) * float(s[i0]) + alpha * float(s[i1])
        xg = (1.0 - alpha) * X[i0] + alpha * X[i1]
        zg = problem.encode(xg, sg, ag)

        print(
            f"candidate {ic}: bracket a={a0:+.9f}->{a1:+.9f}, "
            f"Fa={f0:+.3e}->{f1:+.3e}, interpolated a={ag:+.10f}, s={sg:.10f}"
        )

        def guarded(z, _dummy):
            ev_local = problem.evaluate(z)
            if ev_local.smin < float(args.screening_floor):
                raise FloatingPointError("screening matrix below numerical floor")
            return ev_local.residual

        root = refine_fixed_parameter(zg, 0.0, guarded, opts=opts)
        if not root.converged:
            print(f"  FAILED: |R|={root.residual_norm:.3e}")
            continue

        ev = problem.evaluate(root.x)
        sigma_static, sigma_c, mu, ss, aa, _, y = problem.decode(root.x)
        d = ev.transverse
        w = ev.weights
        thermo = ev.free_energy
        Fpc = float(thermo.free_energy_per_primitive_cell)
        Ferr = Fpc - F_ed_pc if np.isfinite(F_ed_pc) else np.nan

        print(
            f"  OK: a*={aa:+.12f}, s*={ss:.12f}, "
            f"w=(n,B,J)=({w.density:.9f},{w.bond:.9f},{w.current:.9f})"
        )
        print(
            f"      Fs={d.dF_ds_per_cell:+.3e}, Fa={d.dF_da_per_cell:+.3e}, "
            f"gB={d.dF_dlambda_B_per_cell:+.3e}, gJ={d.dF_dlambda_J_per_cell:+.3e}, "
            f"|grad|={d.gradient_norm_per_cell:.3e}"
        )
        print(
            f"      F/cell={Fpc:+.12e}, F-F_ED/cell={Ferr:+.6e}, "
            f"phys={ev.physical_residual:.3e}, smin={ev.smin:.3e}, "
            f"Newton={root.newton_iterations}, GMRES={root.gmres_iterations}"
        )

        definition = build_weighted_nbj_definition(
            geometry.interaction_pairs, norb, V, w
        )
        bg = _make_background(
            problem, definition, ev, root, sigma_static, sigma_c, mu
        )

        chi_raw = np.full((2, 2), np.nan + 0j, dtype=complex)
        chi = np.full((2, 2), np.nan, dtype=float)
        chi_imag = np.nan
        chi_err = np.nan
        vertex_residuals = []
        vertex_converged = False

        if not args.skip_chi:
            vertex_converged = True
            for b, (name, Ksrc) in enumerate(zip(CHANNELS, operators)):
                print(f"      cGW vertex {name} ...")
                vr = solve_channel_vertex_q0(bg, definition, Ksrc, grid, opts=vopts)
                vertex_residuals.append(float(vr.final_error))
                if not vr.converged:
                    vertex_converged = False
                    print(
                        f"      WARNING: {name} vertex failed: "
                        f"residual={vr.final_error:.3e}"
                    )
                    break
                for ia, Kleft in enumerate(operators):
                    chi_raw[ia, b] = susceptibility_from_vertex_q0(
                        bg.G, Kleft, vr.Gamma, grid
                    )

            if vertex_converged:
                chi, chi_imag = _physical_chi(chi_raw)
                if chi_ed.shape == (2, 2):
                    chi_err = _relerr(chi, chi_ed)
                print(
                    f"      chi(same,opp)=({chi[0,0]:+.9f},{chi[1,1]:+.9f}), "
                    f"chi_relerr={chi_err:.6e}, max Im(chi)={chi_imag:.3e}"
                )
        else:
            print("      chi skipped by --skip-chi")

        vertex_max_residual = max(vertex_residuals) if vertex_residuals else np.nan

        rows.append({
            "candidate": ic,
            "source_root": root_index,
            "V": f"{V:.16g}",
            "a_star": f"{aa:.16g}",
            "s_star": f"{ss:.16g}",
            "lambda_n": f"{w.density:.16g}",
            "lambda_B": f"{w.bond:.16g}",
            "lambda_J": f"{w.current:.16g}",
            "mu": f"{mu:.16g}",
            "F_per_cell": f"{Fpc:.16g}",
            "F_ED_per_cell": f"{F_ed_pc:.16g}",
            "F_minus_ED_per_cell": f"{Ferr:.16g}",
            "chi_same": f"{chi[0,0]:.16g}",
            "chi_opposite": f"{chi[1,1]:.16g}",
            "chi_relerr": f"{chi_err:.16g}",
            "chi_max_imag": f"{chi_imag:.16g}",
            "vertex_converged": int(vertex_converged),
            "vertex_max_residual": f"{vertex_max_residual:.16g}",
            "dF_ds_per_cell": f"{d.dF_ds_per_cell:.16g}",
            "dF_da_per_cell": f"{d.dF_da_per_cell:.16g}",
            "gB_per_cell": f"{d.dF_dlambda_B_per_cell:.16g}",
            "gJ_per_cell": f"{d.dF_dlambda_J_per_cell:.16g}",
            "gradient_norm_per_cell": f"{d.gradient_norm_per_cell:.16g}",
            "d2F_ds2_per_cell": f"{ev.longitudinal.d2F_ds2_per_cell:.16g}",
            "d2F_da2_per_cell": f"{d.d2F_da2_per_cell:.16g}",
            "physical_residual": f"{ev.physical_residual:.16g}",
            "smin": f"{ev.smin:.16g}",
            "newton_iterations": root.newton_iterations,
            "gmres_iterations": root.gmres_iterations,
            "residual_norm": f"{root.residual_norm:.16g}",
        })
        saved.append({
            "X_full": np.asarray(root.x),
            "X_longitudinal": np.asarray(y),
            "Sigma_static": np.asarray(sigma_static),
            "Sigma_c": np.asarray(sigma_c),
            "G": np.asarray(bg.G),
            "P": np.asarray(bg.P),
            "W": np.asarray(bg.W),
            "chi": np.asarray(chi),
            "chi_raw": np.asarray(chi_raw),
            "F": float(thermo.helmholtz_free_energy),
            "F_per_cell": Fpc,
        })

    if not rows:
        raise RuntimeError("all full-simplex refinement candidates failed")

    stem = args.transverse_npz.stem.replace("_transverse", "")
    csvfile = args.out / f"{stem}_full_pms.csv"
    npzfile = args.out / f"{stem}_full_pms.npz"
    _write_csv(csvfile, rows)
    np.savez_compressed(
        npzfile,
        V=float(V),
        source_root=int(root_index),
        channels=np.asarray(CHANNELS),
        a_star=np.asarray([float(r["a_star"]) for r in rows]),
        s_star=np.asarray([float(r["s_star"]) for r in rows]),
        lambda_n=np.asarray([float(r["lambda_n"]) for r in rows]),
        lambda_B=np.asarray([float(r["lambda_B"]) for r in rows]),
        lambda_J=np.asarray([float(r["lambda_J"]) for r in rows]),
        mu=np.asarray([float(r["mu"]) for r in rows]),
        F_gw=np.asarray([z["F"] for z in saved]),
        F_gw_per_cell=np.asarray([z["F_per_cell"] for z in saved]),
        F_ed_per_cell=float(F_ed_pc),
        chi=np.stack([z["chi"] for z in saved]),
        chi_raw=np.stack([z["chi_raw"] for z in saved]),
        chi_same=np.asarray([float(r["chi_same"]) for r in rows]),
        chi_opposite=np.asarray([float(r["chi_opposite"]) for r in rows]),
        chi_relerr=np.asarray([float(r["chi_relerr"]) for r in rows]),
        chi_ed=np.asarray(chi_ed),
        vertex_converged=np.asarray([bool(int(r["vertex_converged"])) for r in rows]),
        vertex_max_residual=np.asarray([float(r["vertex_max_residual"]) for r in rows]),
        dF_ds_per_cell=np.asarray([float(r["dF_ds_per_cell"]) for r in rows]),
        dF_da_per_cell=np.asarray([float(r["dF_da_per_cell"]) for r in rows]),
        gradient_norm_per_cell=np.asarray([float(r["gradient_norm_per_cell"]) for r in rows]),
        X_full=np.stack([z["X_full"] for z in saved]),
        X_longitudinal=np.stack([z["X_longitudinal"] for z in saved]),
        Sigma_static=np.stack([z["Sigma_static"] for z in saved]),
        Sigma_c=np.stack([z["Sigma_c"] for z in saved]),
        G=np.stack([z["G"] for z in saved]),
        P=np.stack([z["P"] for z in saved]),
        W=np.stack([z["W"] for z in saved]),
        source_transverse_npz=np.asarray(str(args.transverse_npz)),
        source_target_npz=np.asarray(str(source_target)),
    )
    print(f"saved {csvfile}")
    print(f"saved {npzfile}")


if __name__ == "__main__":
    main()
