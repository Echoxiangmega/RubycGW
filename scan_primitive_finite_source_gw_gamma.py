#!/usr/bin/env python3
"""Finite-source primitive-mesh scan with iterative covariant GW-Gamma_P screening.

The ordinary SC-GW branch is first followed from large to small source.  For each
source field, the covariant orbital-density vertex is then fed back iteratively
through

    chi_nn,cov -> P_Gamma -> W_Gamma -> Sigma_GW -> G.

By default only the static bosonic sector is vertex-corrected (``--m-max 0``),
which is the affordable first test suggested by the one-shot post-GW diagnostic.
Use ``--m-max -1`` for all represented bosonic frequencies.

Optionally (enabled by default) a small-torus exact diagonalization is performed
at the same reference source and filling.  Because a dense primitive k mesh and
an ED torus generally do not share the same momentum points, the Green-function
benchmark compares the directly compatible cell-local object

    G_loc(iw) = (1/Nk) sum_k G(k,iw)

against the cell-averaged diagonal 6x6 block of the exact-torus Green matrix.
The reported ``Gerr`` is the relative Frobenius norm with ED as reference.

This is a Gamma_P screening-feedback diagnostic, not a fully conserving Hedin
GWGamma self-energy-vertex closure; see ``rubycgw.hedin_gamma``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark_finite_source_current import _bilinear_expectation_tail_completed
from rubycgw.ed_green_compare import (
    primitive_local_green,
    relative_green_error,
    solve_exact_finite_source_local_green,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.hedin_gamma import (
    GammaPFeedbackOptions,
    solve_matrix_gw_gamma_feedback,
)
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import compute_polarization_matrix
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--source", choices=["same", "opposite", "z_same", "z_opposite"], default="same")
    p.add_argument("--mesh", default="3x3")
    p.add_argument("--h", nargs="+", type=float, default=[.2, .1, .05, .02, .01])
    p.add_argument("--reference-ncell", type=int, default=2)
    p.add_argument("--T", type=float, default=.08)
    p.add_argument("--ti", type=float, default=.4)
    p.add_argument("--t1", type=float, default=.2)
    p.add_argument("--t2", type=float, default=.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max-iter", type=int, default=1000)
    p.add_argument("--gw-mixing", type=float, default=.18)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=8)
    p.add_argument("--pulay-start", type=int, default=4)
    p.add_argument("--pulay-regularization", type=float, default=1e-9)
    p.add_argument("--backend", choices=["fft", "direct"], default="fft")

    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--vertex-tol", type=float, default=2e-8)
    p.add_argument("--gmres-restart", type=int, default=14)
    p.add_argument("--m-max", type=int, default=0, help="correct |m|<=m_max; -1 means all represented m")

    p.add_argument("--gamma-max-iter", type=int, default=20)
    p.add_argument("--gamma-tol", type=float, default=2e-6)
    p.add_argument("--gamma-mixing", type=float, default=.15)
    p.add_argument("--gamma-w-mixing", type=float, default=None)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fit-points", type=int, default=3)

    p.add_argument("--ed", dest="ed", action="store_true", default=True,
                   help="run the small-torus ED Green benchmark (default)")
    p.add_argument("--no-ed", dest="ed", action="store_false",
                   help="skip the ED Green benchmark")
    p.add_argument("--ed-L1", type=int, default=2)
    p.add_argument("--ed-L2", type=int, default=1)
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-low-nfreq", type=int, default=8,
                   help="number of smallest-|omega| Matsubara points for low-frequency Gerr")

    p.add_argument("--out", type=Path, default=Path("results/primitive_gw_gamma"))
    return p.parse_args()


def _parse_mesh(token: str) -> tuple[int, int]:
    raw = str(token).strip().lower().replace("×", "x")
    parts = raw.split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid mesh {token!r}; expected NxM")
    n1, n2 = int(parts[0]), int(parts[1])
    if n1 < 1 or n2 < 1:
        raise ValueError("mesh dimensions must be positive")
    return n1, n2


def _hvalues(values) -> np.ndarray:
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr < -1e-15):
        raise ValueError("--h must contain finite non-negative values")
    arr[np.abs(arr) < 1e-15] = 0.0
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _gw_retry(h0, Vq, grid, opts, initial=None):
    plans = [
        ("base", opts),
        ("pulay-0.10", replace(opts, mixing=min(float(opts.mixing), .10),
                                mixing_method="pulay", max_iter=max(int(opts.max_iter), 1400))),
        ("pulay-0.05", replace(opts, mixing=min(float(opts.mixing), .05),
                                mixing_method="pulay", max_iter=max(int(opts.max_iter), 1800))),
        ("linear-0.03", replace(opts, mixing=min(float(opts.mixing), .03),
                                 mixing_method="linear", max_iter=max(int(opts.max_iter), 2200))),
    ]
    seed = initial
    best = None
    label_best = "none"
    for ia, (label, trial) in enumerate(plans):
        out = solve_matrix_gw_fast(h0, Vq, grid, opts=trial, initial=seed)
        print(
            f"      GW retry[{ia}] {label:12s}: {'OK' if out.converged else 'FAIL':4s} "
            f"iter={out.iterations:4d} r={out.final_error:.3e} "
            f"smin={out.min_screening_singular_value:.3e} "
            f"q*=({out.min_screening_q1:.4f},{out.min_screening_q2:.4f})"
        )
        if best is None or out.final_error < best.final_error:
            best, label_best = out, label
        if out.converged:
            return out, label
        seed = out
    return best, label_best


def _current_from_state(G, sigma_h, sigma_f, mu, K, h0, grid) -> float:
    h_static = h0 + sigma_h[None, None] + sigma_f
    return float(
        _bilinear_expectation_tail_completed(G, K, grid, h_static, mu).real
    )


def _gw_current(result, K, Vq, h0, grid, backend) -> float:
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        result.G, result.W, Vq, grid, h0, result.mu, result.Sigma_H,
        backend=backend,
    )
    return _current_from_state(
        result.G, result.Sigma_H, sigma_f, result.mu, K, h0, grid
    )


def _finite_mask(P):
    return np.all(np.isfinite(np.asarray(P)), axis=(-2, -1))


def _relative_frobenius(a, b, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    da = np.asarray(a)[mask] - np.asarray(b)[mask]
    bb = np.asarray(b)[mask]
    den = np.linalg.norm(bb)
    return float(np.linalg.norm(da) / den) if den > 0 else float("nan")


def _static_gamma_soft_mode(Pgamma, Vq, grid):
    im = np.flatnonzero(np.asarray(grid.m_values) == 0)
    if not im.size:
        return np.nan, np.nan, np.nan
    p0 = np.asarray(Pgamma[int(im[0])])
    eye = np.eye(p0.shape[-1], dtype=complex)
    best = (np.inf, np.nan, np.nan)
    for iq1 in range(grid.nk1):
        for iq2 in range(grid.nk2):
            p = p0[iq1, iq2]
            if not np.all(np.isfinite(p)):
                continue
            lhs = eye - Vq[iq1, iq2] @ p
            smin = float(np.min(np.linalg.svd(lhs, compute_uv=False)))
            if smin < best[0]:
                best = (smin, iq1 / grid.nk1, iq2 / grid.nk2)
    return best


def _fit(h, J, ok, npoints):
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    idx = np.flatnonzero(np.asarray(ok, dtype=bool) & np.isfinite(J) & (h > 0))
    if idx.size < 2:
        return np.nan, np.nan, 0
    idx = idx[np.argsort(h[idx])][:max(2, int(npoints))]
    slope, intercept = np.polyfit(h[idx], J[idx], 1)
    return float(intercept), float(slope), int(idx.size)


def _low_frequency_indices(omega: np.ndarray, count: int) -> np.ndarray:
    n = min(max(int(count), 1), len(omega))
    return np.asarray(np.argsort(np.abs(np.asarray(omega, dtype=float)))[:n], dtype=int)


def main():
    args = _args()
    nk1, nk2 = _parse_mesh(args.mesh)
    h_ref = _hvalues(args.h)
    m_max = None if int(args.m_max) < 0 else int(args.m_max)
    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    root = float(np.sqrt(args.reference_ncell))
    h_cell = h_ref / root

    if args.ed:
        ed_ncell = int(args.ed_L1) * int(args.ed_L2)
        if ed_ncell != int(args.reference_ncell):
            raise ValueError(
                "ED source/current normalization requires ed-L1*ed-L2 == reference-ncell; "
                f"got {ed_ncell} != {args.reference_ncell}"
            )
        if 6 * ed_ncell > 16:
            raise ValueError("ExactSmallRubyThermal requires at most 16 sites")
        if not (0.0 <= float(args.ed_discard_weight_tol) < 1.0):
            raise ValueError("--ed-discard-weight-tol must lie in [0,1)")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(nk1=nk1, nk2=nk2, nw=args.nw, nOmega=args.nomega, T=args.T)
    h0_base = np.asarray(build_h0(grid.kmesh(), params), dtype=complex)
    Vq = np.asarray(build_interaction(grid.qmesh(), params), dtype=complex)

    gw_opts = GWOptions(
        target_filling=args.filling,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        mixing_method=args.mixing_method,
        pulay_history=args.pulay_history,
        pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    vertex_opts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        solver="gmres",
        gmres_restart=args.gmres_restart,
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    gamma_opts = GammaPFeedbackOptions(
        max_iter=args.gamma_max_iter,
        tol=args.gamma_tol,
        mixing=args.gamma_mixing,
        w_mixing=args.gamma_w_mixing,
        m_max=m_max,
        allow_unconverged_vertex=args.allow_unconverged,
        verbose=True,
    )

    n = len(h_ref)
    Jgw = np.full(n, np.nan)
    Jgamma = np.full(n, np.nan)
    Jed = np.full(n, np.nan)
    mu_ed = np.full(n, np.nan)
    gw_ok = np.zeros(n, dtype=bool)
    gamma_ok = np.zeros(n, dtype=bool)
    gamma_err = np.full(n, np.nan)
    gamma_iter = np.zeros(n, dtype=int)
    relW = np.full(n, np.nan)
    relP = np.full(n, np.nan)
    smin_gamma = np.full(n, np.nan)
    q1_gamma = np.full(n, np.nan)
    q2_gamma = np.full(n, np.nan)
    hed_err = np.full(n, np.nan)
    vertex_err = np.full(n, np.nan)
    Gerr_gw = np.full(n, np.nan)
    Gerr_gamma = np.full(n, np.nan)
    Gerr_gw_low = np.full(n, np.nan)
    Gerr_gamma_low = np.full(n, np.nan)
    G_ed_local = np.full((n, grid.nf, 6, 6), np.nan + 0j, dtype=complex)
    G_gw_local = np.full_like(G_ed_local, np.nan + 0j)
    G_gamma_local = np.full_like(G_ed_local, np.nan + 0j)
    ed_kept_weight = np.full(n, np.nan)

    print("=== primitive finite-source iterative GW-Gamma_P screening ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"mesh={nk1}x{nk2}, h_ref={h_ref.tolist()}, m_max={m_max}")
    print("Gamma enters covariant density screening; Sigma retains the GW form.")
    if args.ed:
        print(
            f"ED Green benchmark: {args.ed_L1}x{args.ed_L2} torus; "
            "Gerr compares cell-local Matsubara G (dense-k finite-size caveat applies)."
        )

    last_gw = None
    last_gamma = None
    rows = []
    low_idx = _low_frequency_indices(grid.omega, args.ed_low_nfreq)

    for ih, (href, hpc) in enumerate(zip(h_ref, h_cell)):
        print(f"\n-- h_ref={href:.8g}, h_cell={hpc:.8g} --")
        h0 = h0_base - float(hpc) * K[None, None]
        h0 = .5 * (h0 + np.swapaxes(h0.conj(), -1, -2))

        gw, retry = _gw_retry(h0, Vq, grid, gw_opts, initial=last_gw)
        gw_ok[ih] = bool(gw.converged)
        Jgw[ih] = root * _gw_current(gw, K, Vq, h0, grid, args.backend)
        G_gw_local[ih] = primitive_local_green(gw.G)
        if gw.converged:
            last_gw = gw
        if not gw.converged and not args.allow_unconverged:
            raise RuntimeError(f"GW did not converge at h_ref={href:g}")

        if args.ed:
            print("      solving matching exact ED Green function ...")
            ed = solve_exact_finite_source_local_green(
                L1=args.ed_L1,
                L2=args.ed_L2,
                params=params,
                V=args.V,
                source_channel=source,
                h_ref=href,
                filling_per_cell=args.filling,
                T=args.T,
                omega=grid.omega,
                discard_weight_tol=args.ed_discard_weight_tol,
            )
            Jed[ih] = ed.J_ref
            mu_ed[ih] = ed.mu
            ed_kept_weight[ih] = ed.kept_weight
            G_ed_local[ih] = ed.G_local
            Gerr_gw[ih] = relative_green_error(G_gw_local[ih], G_ed_local[ih])
            Gerr_gw_low[ih] = relative_green_error(
                G_gw_local[ih, low_idx], G_ed_local[ih, low_idx]
            )
            print(
                f"      ED: J_ref={Jed[ih]:+.9f}, mu={mu_ed[ih]:+.9f}, "
                f"kept_weight={ed_kept_weight[ih]:.12f}, Gerr_GW(local)={Gerr_gw[ih]:.6e}"
            )

        print("      iterating covariant Gamma_P screening feedback ...")
        gamma = solve_matrix_gw_gamma_feedback(
            h0,
            Vq,
            grid,
            gw_opts=gw_opts,
            vertex_opts=vertex_opts,
            feedback_opts=gamma_opts,
            background=gw,
            initial_state=last_gamma,
        )
        gamma_ok[ih] = bool(gamma.converged)
        gamma_err[ih] = float(gamma.final_error)
        gamma_iter[ih] = int(gamma.iterations)
        hed_err[ih] = float(gamma.screening_identity_error)
        Jgamma[ih] = root * _current_from_state(
            gamma.G,
            gamma.Sigma_H,
            gamma.Sigma_F,
            gamma.mu,
            K,
            h0,
            grid,
        )
        G_gamma_local[ih] = primitive_local_green(gamma.G)
        if gamma.converged:
            last_gamma = gamma
        elif not args.allow_unconverged:
            raise RuntimeError(
                f"GW-Gamma_P feedback did not converge at h_ref={href:g}: "
                f"{gamma.final_error:.3e}"
            )

        if args.ed:
            Gerr_gamma[ih] = relative_green_error(G_gamma_local[ih], G_ed_local[ih])
            Gerr_gamma_low[ih] = relative_green_error(
                G_gamma_local[ih, low_idx], G_ed_local[ih, low_idx]
            )

        mask = _finite_mask(gamma.P_gamma)
        relW[ih] = _relative_frobenius(gamma.W, gw.W, mask)
        Pbubble = compute_polarization_matrix(gamma.G, grid, backend=args.backend)
        relP[ih] = _relative_frobenius(gamma.P_gamma, Pbubble, mask)
        smin_gamma[ih], q1_gamma[ih], q2_gamma[ih] = _static_gamma_soft_mode(
            gamma.P_gamma, Vq, grid
        )
        explicit = np.asarray(gamma.density_response.solved_explicitly, dtype=bool)
        if np.any(explicit):
            vertex_err[ih] = float(
                np.nanmax(gamma.density_response.transfer_max_error[explicit])
            )

        print(
            f"    GW:       J_ref={Jgw[ih]:+.9f}, smin={gw.min_screening_singular_value:.4e}, "
            f"q*=({gw.min_screening_q1:.4f},{gw.min_screening_q2:.4f}), retry={retry}"
        )
        print(
            f"    GWGamma_P:{'OK' if gamma.converged else 'FAIL':4s} J_ref={Jgamma[ih]:+.9f}, "
            f"iter={gamma.iterations}, r={gamma.final_error:.3e}, dJ={Jgamma[ih]-Jgw[ih]:+.9f}"
        )
        if args.ed:
            ratio = Gerr_gamma[ih] / Gerr_gw[ih] if Gerr_gw[ih] > 0 else np.nan
            ratio_low = Gerr_gamma_low[ih] / Gerr_gw_low[ih] if Gerr_gw_low[ih] > 0 else np.nan
            print(
                f"    ED:       J_ref={Jed[ih]:+.9f}, dJ_GW={Jgw[ih]-Jed[ih]:+.9f}, "
                f"dJ_Gamma={Jgamma[ih]-Jed[ih]:+.9f}"
            )
            print(
                f"    Gerr local: GW={Gerr_gw[ih]:.6e}, GWGamma_P={Gerr_gamma[ih]:.6e}, "
                f"Gamma/GW={ratio:.4f}"
            )
            print(
                f"    Gerr low(|w|,{len(low_idx)} pts): GW={Gerr_gw_low[ih]:.6e}, "
                f"GWGamma_P={Gerr_gamma_low[ih]:.6e}, Gamma/GW={ratio_low:.4f}"
            )
        print(
            f"    vertex:   maxerr={vertex_err[ih]:.3e}, rel||P_Gamma-P_bub||={relP[ih]:.4e}, "
            f"Hedin identity err={hed_err[ih]:.3e}"
        )
        print(
            f"    screening:rel||W_Gamma-W_GW||={relW[ih]:.4e}, "
            f"smin_Gamma(static)={smin_gamma[ih]:.4e} "
            f"q*=({q1_gamma[ih]:.4f},{q2_gamma[ih]:.4f})"
        )

        rows.append({
            "h_ref": href,
            "h_cell": hpc,
            "J_ed": Jed[ih],
            "J_gw": Jgw[ih],
            "J_gamma": Jgamma[ih],
            "dJ": Jgamma[ih] - Jgw[ih],
            "dJ_gw_vs_ed": Jgw[ih] - Jed[ih],
            "dJ_gamma_vs_ed": Jgamma[ih] - Jed[ih],
            "mu_ed": mu_ed[ih],
            "Gerr_gw_local": Gerr_gw[ih],
            "Gerr_gamma_local": Gerr_gamma[ih],
            "Gerr_gw_low": Gerr_gw_low[ih],
            "Gerr_gamma_low": Gerr_gamma_low[ih],
            "gw_ok": int(gw_ok[ih]),
            "gamma_ok": int(gamma_ok[ih]),
            "gamma_iter": gamma_iter[ih],
            "gamma_error": gamma_err[ih],
            "vertex_error": vertex_err[ih],
            "rel_W_gamma_vs_gw": relW[ih],
            "rel_P_gamma_vs_bubble": relP[ih],
            "smin_gamma_static": smin_gamma[ih],
            "q1_gamma_static": q1_gamma[ih],
            "q2_gamma_static": q2_gamma[ih],
            "hedin_identity_error": hed_err[ih],
        })

    J0g, bg, ng = _fit(h_ref, Jgw, gw_ok, args.fit_points)
    J0x, bx, nx = _fit(h_ref, Jgamma, gamma_ok, args.fit_points)
    print("\n=== small-h fits ===")
    print(f"GW:        J0={J0g:+.9f}, slope={bg:+.9f}, n={ng}")
    print(f"GWGamma_P: J0={J0x:+.9f}, slope={bx:+.9f}, n={nx}")
    print(f"dJ0={J0x-J0g:+.9f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"V{args.V:g}_fill{args.filling:g}_{source}_{nk1}x{nk2}_m{('all' if m_max is None else m_max)}"
    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    npz_path = args.out / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        h_ref=h_ref,
        h_cell=h_cell,
        J_ed=Jed,
        J_gw=Jgw,
        J_gamma=Jgamma,
        mu_ed=mu_ed,
        Gerr_gw_local=Gerr_gw,
        Gerr_gamma_local=Gerr_gamma,
        Gerr_gw_low=Gerr_gw_low,
        Gerr_gamma_low=Gerr_gamma_low,
        G_ed_local=G_ed_local,
        G_gw_local=G_gw_local,
        G_gamma_local=G_gamma_local,
        ed_kept_weight=ed_kept_weight,
        ed_low_frequency_indices=low_idx,
        omega=grid.omega,
        gw_ok=gw_ok,
        gamma_ok=gamma_ok,
        gamma_error=gamma_err,
        gamma_iter=gamma_iter,
        relW=relW,
        relP=relP,
        smin_gamma=smin_gamma,
        q1_gamma=q1_gamma,
        q2_gamma=q2_gamma,
        hedin_identity_error=hed_err,
        J0_gw=J0g,
        slope_gw=bg,
        J0_gamma=J0x,
        slope_gamma=bx,
        ed_L1=int(args.ed_L1),
        ed_L2=int(args.ed_L2),
        ed_enabled=bool(args.ed),
    )
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
