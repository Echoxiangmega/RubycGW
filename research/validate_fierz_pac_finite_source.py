#!/usr/bin/env python3
"""Finite-source/FDT validation of the analytic Fierz-channel cGW response.

This script starts from an already refined exact-V PAC target result, applies a
small one-body source

    H(h) = H(0) - h K_source,

and re-solves the *same* Fierz-channel self-consistent GW equations at +h and
-h with matrix-free Newton-Krylov.  The central finite difference

    chi_FD[a,b] = (<K_a>_{+h_b} - <K_a>_{-h_b}) / (2 h_b)

is compared directly with the analytic q=0 cGW susceptibility saved by
``benchmark_fierz_pac_target.py``.  No ED calculation is performed here.

For the TR-odd z_same/z_opposite sources used here, fixed-filling and fixed-mu
linear responses coincide at h=0 in the symmetric state because the mixed
number-current response vanishes by time reversal.  The code nevertheless
keeps the same fixed filling as the PAC calculation and reports dmu/dh as a
numerical check.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_pac import FierzGWResidual
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target-result", type=Path, required=True,
        help="NPZ written by benchmark_fierz_pac_target.py",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help=(
            "PAC checkpoint used to reconstruct ti/t1/t2 and grid metadata. "
            "If omitted, use source_checkpoint stored in the target NPZ."
        ),
    )
    p.add_argument(
        "--h", nargs="+", type=float,
        default=[1e-3, 5e-4, 2e-4, 1e-4],
        help="positive source magnitudes for central finite differences",
    )
    p.add_argument("--newton-tol", type=float, default=1e-11)
    p.add_argument("--newton-max", type=int, default=18)
    p.add_argument("--fd-eps", type=float, default=3e-7)
    p.add_argument("--gmres-rtol", type=float, default=3e-4)
    p.add_argument("--gmres-maxiter", type=int, default=100)
    p.add_argument("--gmres-restart", type=int, default=24)
    p.add_argument("--line-search-min", type=float, default=1.0 / 8192.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument(
        "--out", type=Path, default=Path("results/fierz_finite_source_validation")
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _positive_unique(values):
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("--h values must be finite and strictly positive")
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _expectation(K, rho):
    z = np.einsum(
        "ab,ba->", np.asarray(K, dtype=complex), np.asarray(rho, dtype=complex),
        optimize=True,
    )
    return complex(z)


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _checkpoint_path(args, target):
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if "source_checkpoint" not in target.files:
        raise RuntimeError(
            "target result has no source_checkpoint metadata; pass --checkpoint explicitly"
        )
    p = Path(str(np.asarray(target["source_checkpoint"]).item()))
    if not p.exists():
        raise RuntimeError(
            f"stored source checkpoint does not exist: {p}; pass --checkpoint explicitly"
        )
    return p


def _root_at_source(
    x0, h0_base, Ksrc, h, meta, geometry, grid, V_target, opts, screening_floor
):
    h0h = np.asarray(h0_base, dtype=complex) - float(h) * np.asarray(Ksrc, dtype=complex)
    h0h = 0.5 * (h0h + np.swapaxes(h0h.conj(), -1, -2))
    problem = FierzGWResidual(
        h0h,
        geometry.interaction_pairs,
        str(meta["mode"]),
        grid,
        float(meta["target"]),
    )

    def guarded(x, V):
        ev = problem.evaluate(x, V)
        if ev.smin < float(screening_floor):
            raise FloatingPointError("screening matrix below numerical floor")
        return ev.residual

    root = refine_fixed_parameter(
        np.asarray(x0, dtype=float), float(V_target), guarded, opts=opts
    )
    if not root.converged:
        raise RuntimeError(
            f"finite-source root failed for h={h:+.6g}: "
            f"|R_scaled|={root.residual_norm:.3e}"
        )
    ev = problem.evaluate(root.x, float(V_target))
    return root, ev


def main():
    args = _args()
    hs = _positive_unique(args.h)
    target = np.load(args.target_result, allow_pickle=False)
    needed = ("x_target", "V", "mode", "chi", "filling", "T", "L1", "L2")
    missing = [k for k in needed if k not in target.files]
    if missing:
        raise RuntimeError(f"target result missing keys: {missing}")

    checkpoint_path = _checkpoint_path(args, target)
    checkpoint = np.load(checkpoint_path, allow_pickle=False)
    if "signature" not in checkpoint.files:
        raise RuntimeError("PAC checkpoint has no signature metadata")
    meta = json.loads(str(np.asarray(checkpoint["signature"]).item()))
    required_meta = (
        "mode", "L1", "L2", "filling", "target", "T", "ti", "t1", "t2",
        "nw", "nomega", "norb",
    )
    missing_meta = [k for k in required_meta if k not in meta]
    if missing_meta:
        raise RuntimeError(f"PAC checkpoint signature missing keys: {missing_meta}")

    V_target = float(target["V"])
    mode_target = str(np.asarray(target["mode"]).item()).lower()
    if mode_target != str(meta["mode"]).lower():
        raise RuntimeError("target result mode does not match PAC checkpoint")
    for key in ("L1", "L2"):
        if int(target[key]) != int(meta[key]):
            raise RuntimeError(f"target result {key} does not match PAC checkpoint")
    for key in ("filling", "T"):
        if abs(float(target[key]) - float(meta[key])) > 1e-13:
            raise RuntimeError(f"target result {key} does not match PAC checkpoint")

    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    norb = int(geometry.n_sites)
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(meta["nw"]), nOmega=int(meta["nomega"]),
        T=float(meta["T"]),
    )
    h0_base = np.asarray(geometry.h0, dtype=complex)[None, None]
    x0 = np.asarray(target["x_target"], dtype=float)
    base_problem = FierzGWResidual(
        h0_base, geometry.interaction_pairs, str(meta["mode"]), grid,
        float(meta["target"]),
    )
    if x0.size != base_problem.codec.size:
        raise RuntimeError("target x_target size does not match reconstructed problem")
    ev0 = base_problem.evaluate(x0, V_target)

    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])
    chi_analytic = np.asarray(target["chi"], dtype=float)
    if chi_analytic.shape != (2, 2):
        raise RuntimeError("target analytic chi must have shape (2,2)")
    J0 = np.asarray([_expectation(K, ev0.rho).real for K in operators], dtype=float)

    opts = PACOptions(
        tol=float(args.newton_tol), max_newton=int(args.newton_max),
        fd_eps=float(args.fd_eps), gmres_rtol=float(args.gmres_rtol),
        gmres_maxiter=int(args.gmres_maxiter), gmres_restart=int(args.gmres_restart),
        line_search_min=float(args.line_search_min), verbose=bool(args.verbose),
    )

    print("=== Fierz PAC finite-source / FDT validation ===")
    print(
        f"mode={meta['mode']}, V={V_target:g}, torus={meta['L1']}x{meta['L2']}, "
        f"filling={meta['filling']:g}, T={meta['T']:g}, nw={meta['nw']}, "
        f"nOmega={meta['nomega']}"
    )
    print(
        f"zero-source: phys={ev0.physical_residual:.3e}, Nerr={ev0.filling_error:+.3e}, "
        f"smin={ev0.smin:.3e}, <same,opp>=({J0[0]:+.3e},{J0[1]:+.3e})"
    )
    print(
        "analytic cGW chi:\n"
        f"  [[{chi_analytic[0,0]:+.9f}, {chi_analytic[0,1]:+.9f}],\n"
        f"   [{chi_analytic[1,0]:+.9f}, {chi_analytic[1,1]:+.9f}]]"
    )

    nh = len(hs)
    chi_fd = np.zeros((nh, 2, 2), dtype=float)
    Jplus = np.zeros((nh, 2, 2), dtype=float)
    Jminus = np.zeros((nh, 2, 2), dtype=float)
    mu_plus = np.zeros((nh, 2), dtype=float)
    mu_minus = np.zeros((nh, 2), dtype=float)
    root_scaled = np.zeros((nh, 2, 2), dtype=float)
    physical_residual = np.zeros((nh, 2, 2), dtype=float)
    smin = np.zeros((nh, 2, 2), dtype=float)
    newton_iter = np.zeros((nh, 2, 2), dtype=int)
    gmres_iter = np.zeros((nh, 2, 2), dtype=int)
    rows = []

    for ih, hmag in enumerate(hs):
        print(f"\n--- h={hmag:.6g} ---")
        for b, (src_name, Ksrc) in enumerate(zip(CHANNELS, operators)):
            evals = []
            roots = []
            for isig, sign in enumerate((+1.0, -1.0)):
                h = sign * float(hmag)
                root, ev = _root_at_source(
                    x0, h0_base, Ksrc, h, meta, geometry, grid, V_target,
                    opts, args.screening_floor,
                )
                roots.append(root)
                evals.append(ev)
                vals = np.asarray([_expectation(K, ev.rho).real for K in operators])
                if sign > 0:
                    Jplus[ih, :, b] = vals
                    _, _, mu = base_problem.codec.decode(root.x)
                    mu_plus[ih, b] = mu
                else:
                    Jminus[ih, :, b] = vals
                    _, _, mu = base_problem.codec.decode(root.x)
                    mu_minus[ih, b] = mu
                root_scaled[ih, isig, b] = root.residual_norm
                physical_residual[ih, isig, b] = ev.physical_residual
                smin[ih, isig, b] = ev.smin
                newton_iter[ih, isig, b] = root.newton_iterations
                gmres_iter[ih, isig, b] = root.gmres_iterations

            chi_fd[ih, :, b] = (
                Jplus[ih, :, b] - Jminus[ih, :, b]
            ) / (2.0 * float(hmag))
            dmu = (mu_plus[ih, b] - mu_minus[ih, b]) / (2.0 * float(hmag))
            diag = chi_fd[ih, b, b]
            ana = chi_analytic[b, b]
            rel = (diag - ana) / max(abs(ana), 1e-300)
            print(
                f"source={src_name:10s}: chi_FD_diag={diag:+.9f}, "
                f"analytic={ana:+.9f}, rel={rel:+.3e}, dmu/dh={dmu:+.3e}, "
                f"N/G(+)=({roots[0].newton_iterations}/{roots[0].gmres_iterations}), "
                f"N/G(-)=({roots[1].newton_iterations}/{roots[1].gmres_iterations})"
            )
            rows.append({
                "h": f"{hmag:.16g}", "source": src_name,
                "chi_fd_same": f"{chi_fd[ih,0,b]:.16g}",
                "chi_fd_opposite": f"{chi_fd[ih,1,b]:.16g}",
                "chi_analytic_same": f"{chi_analytic[0,b]:.16g}",
                "chi_analytic_opposite": f"{chi_analytic[1,b]:.16g}",
                "diag_rel_error": f"{rel:.16g}", "dmu_dh": f"{dmu:.16g}",
                "Jplus_same": f"{Jplus[ih,0,b]:.16g}",
                "Jplus_opposite": f"{Jplus[ih,1,b]:.16g}",
                "Jminus_same": f"{Jminus[ih,0,b]:.16g}",
                "Jminus_opposite": f"{Jminus[ih,1,b]:.16g}",
                "phys_plus": f"{evals[0].physical_residual:.16g}",
                "phys_minus": f"{evals[1].physical_residual:.16g}",
                "smin_plus": f"{evals[0].smin:.16g}",
                "smin_minus": f"{evals[1].smin:.16g}",
            })

        materr = _relerr(chi_fd[ih], chi_analytic)
        print(
            f"matrix at h={hmag:.6g}: relerr={materr:.3e}, "
            f"chi_FD=[[{chi_fd[ih,0,0]:+.6f},{chi_fd[ih,0,1]:+.6f}],"
            f"[{chi_fd[ih,1,0]:+.6f},{chi_fd[ih,1,1]:+.6f}]]"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = (
        f"V{V_target:g}_fill{float(meta['filling']):g}_{int(meta['L1'])}x"
        f"{int(meta['L2'])}_{str(meta['mode']).lower()}_finite_source"
    )
    npz_path = args.out / f"{stem}.npz"
    csv_path = args.out / f"{stem}.csv"
    np.savez_compressed(
        npz_path,
        V=V_target, mode=np.asarray(str(meta["mode"])),
        filling=float(meta["filling"]), T=float(meta["T"]),
        L1=int(meta["L1"]), L2=int(meta["L2"]),
        h=np.asarray(hs), chi_analytic=np.asarray(chi_analytic),
        chi_fd=np.asarray(chi_fd), J0=np.asarray(J0),
        Jplus=np.asarray(Jplus), Jminus=np.asarray(Jminus),
        mu_plus=np.asarray(mu_plus), mu_minus=np.asarray(mu_minus),
        root_scaled_residual=np.asarray(root_scaled),
        physical_residual=np.asarray(physical_residual), smin=np.asarray(smin),
        newton_iterations=np.asarray(newton_iter), gmres_iterations=np.asarray(gmres_iter),
        source_target_result=np.asarray(str(args.target_result)),
        source_checkpoint=np.asarray(str(checkpoint_path)),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {npz_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
