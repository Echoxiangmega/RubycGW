#!/usr/bin/env python3
"""Weak-coupling O(V^2) response ledger on the exact 12-site Ruby torus.

Static Gamma-point current diagnostic comparing the exact grand-canonical ED
vertex correction with production tail-consistent cGW.  Through second order,

    Delta chi_cGW = H(1) + F(1) + F(2) + MT(1) + AL(1) + O(V^3),

where F(2) is the second application of the affine production Fock map.  The
script also solves fully resummed F-only and full cGW responses as closure
checks.  The exact bubble is reference-tail completed so V=0 has no finite-box
offset.  This is response-level weak-coupling bookkeeping, not a claim that the
individual pieces are separately observable or valid at V~1.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.ed_cgw_benchmark import bubble_iomega
from rubycgw.fock_diagnostic import fock_bond_diagnostic
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.production_cgw import _dynamic_parts, solve_vertex_q0_tail
from rubycgw.response_tail import (
    build_tail_hf_context,
    build_tail_reference,
    reference_tail_remainder,
)
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    _x_field,
    susceptibility_matrix_q0,
)
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    hartree_self_energy_matrix,
)
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson

DEFAULT_VALUES = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
DEFAULT_CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--values", nargs="+", type=float, default=DEFAULT_VALUES)
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--filling", type=float, default=3.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--fit-vmax", type=float, default=0.05)
    p.add_argument("--thermal-weight-tol", type=float, default=1e-12)
    p.add_argument("--gw-tol", type=float, default=1e-9)
    p.add_argument("--gw-max-iter", type=int, default=1200)
    p.add_argument("--vertex-tol", type=float, default=1e-10)
    p.add_argument("--vertex-max-iter", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("results/ed12_order2_ledger"))
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def _interaction_matrix(exact, V):
    out = np.zeros((exact.n_sites, exact.n_sites), dtype=complex)
    for i, j in exact.interaction_pairs:
        out[i, j] = float(V)
        out[j, i] = float(V)
    return out[None, None]


def _exact_density(exact, mu, T):
    """Exact grand-canonical site density used only for the ED tail reference."""
    probs, _, _ = exact._normalized_probabilities(mu, T)
    density = np.zeros(exact.n_sites, dtype=float)
    for sec, p in zip(exact.sectors, probs):
        if not len(p) or float(np.max(p)) == 0.0:
            continue
        basis_prob = np.abs(sec.eigenvectors) ** 2 @ p
        for i in range(exact.n_sites):
            occ = ((sec.basis >> np.uint64(i)) & np.uint64(1)).astype(float)
            density[i] += float(np.dot(occ, basis_prob).real)
    return density


def _tail_completed_static_bubble(G, operators, grid, h0, mu, sigma_h):
    ops = np.asarray(operators, dtype=complex)
    ib0 = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
    box = np.asarray(bubble_iomega(G, ops, grid)[ib0], dtype=complex)
    ref = build_tail_reference(h0, float(mu), np.asarray(sigma_h, dtype=complex), grid)
    tail = np.empty_like(box)
    remainders = [
        reference_tail_remainder(ref, right, grid, q_index=(0, 0), m_ext=0)[0, 0]
        for right in ops
    ]
    for a, left in enumerate(ops):
        for b, rem in enumerate(remainders):
            tail[a, b] = -np.trace(left @ rem)
    return box + tail, box, tail


def _field_chi(G, K, field, grid):
    return susceptibility_matrix_q0(G, np.asarray([K]), [field], grid)[0, 0]


def _full_opts(args):
    return SupercellVertexOptions(
        max_iter=int(args.vertex_max_iter), tol=float(args.vertex_tol), mixing=0.2,
        solver="gmres", gmres_restart=40,
        include_hartree=True, include_fock=True, include_mt=True, include_al=True,
        verbose=False, momentum_backend="direct",
    )


def _fit_abc(V, y, vmax):
    V = np.asarray(V, float); y = np.asarray(y, float)
    mask = (V > 0.0) & (V <= float(vmax) + 1e-15)
    x, z = V[mask], y[mask]
    if len(x) < 3:
        raise ValueError("need at least three positive points for aV+bV^2+cV^3")
    A = np.column_stack([x, x*x, x*x*x])
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    return dict(a=float(coeff[0]), b=float(coeff[1]), c=float(coeff[2]), n=int(len(x)))


def _fit_bc(V, y, vmax):
    V = np.asarray(V, float); y = np.asarray(y, float)
    mask = (V > 0.0) & (V <= float(vmax) + 1e-15)
    x, z = V[mask], y[mask]
    if len(x) < 2:
        raise ValueError("need at least two positive points for bV^2+cV^3")
    A = np.column_stack([x*x, x*x*x])
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    return dict(b=float(coeff[0]), c=float(coeff[1]), n=int(len(x)))


def _run_point(V, seed, args, grid):
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=float(V))
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    exact.diagonalize(float(V))
    target_N = float(args.filling) * int(args.L1) * int(args.L2)
    mu_ed = exact.solve_mu(target_N, args.T)
    ops = np.asarray([
        exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in args.channels
    ])
    chi_ed_mat, _ = exact.static_susceptibility_matrix(ops, mu_ed, args.T)
    chi_ed = np.diag(chi_ed_mat).astype(float)

    G_ed, _ = exact.green_iomega(
        1j*np.asarray(grid.omega), mu_ed, args.T,
        discard_weight_tol=float(args.thermal_weight_tol),
    )
    G_ed = G_ed[:, None, None]
    Vq = _interaction_matrix(exact, V)
    h0 = np.asarray(exact.h0, dtype=complex)[None, None]
    density_ed = _exact_density(exact, mu_ed, args.T)
    sigma_h_ed = hartree_self_energy_matrix(density_ed, Vq[0, 0])
    B_ed, _, _ = _tail_completed_static_bubble(
        G_ed, ops, grid, h0, mu_ed, sigma_h_ed
    )
    bubble_ed = np.real(np.diag(B_ed))
    exact_vertex = chi_ed - bubble_ed

    gw_opts = GWOptions(
        mu=0.0 if seed is None else float(seed.mu),
        target_filling=target_N,
        max_iter=int(args.gw_max_iter), tol=float(args.gw_tol), mixing=0.2,
        mixing_method="linear", mu_tol=min(1e-10, float(args.gw_tol)*0.1),
        mu_max_iter=120, verbose=False, momentum_backend="direct",
    )
    gw = solve_matrix_gw_anderson(
        h0, Vq, grid, opts=gw_opts, initial=seed,
        anderson=AndersonOptions(),
    )
    if not gw.converged:
        raise RuntimeError(f"SC-GW failed at V={V:g}: residual={gw.final_error:.3e}")

    ib0 = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
    bubble_gw = np.real(np.diag(bubble_iomega(gw.G, ops, grid)[ib0]))
    reference = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)
    P = compute_polarization_matrix(gw.G, grid, backend="direct")
    W = compute_screened_interaction_matrix(P, Vq)
    opts = _full_opts(args)

    nc = len(args.channels)
    H1=np.zeros(nc); F1=np.zeros(nc); F2=np.zeros(nc)
    MT1=np.zeros(nc); AL1=np.zeros(nc); F_resum=np.zeros(nc)
    full_total=np.zeros(nc); full_vertex=np.zeros(nc); mismatch=np.zeros(nc)
    for ic, (ch, K) in enumerate(zip(args.channels, ops)):
        ctx = build_tail_hf_context(
            reference, K, Vq, grid, q_index=(0, 0), m_ext=0,
            backend="direct", include_hartree=True, include_fock=True,
        )
        Kfield = np.broadcast_to(K, gw.G.shape).copy()
        X0 = _x_field(gw.G, Kfield)
        gh, gf = ctx.total_vertex_parts(X0, gw.G)
        gmt, gal1, gal2 = _dynamic_parts(gw.G, W, Vq, X0, grid, opts)
        H1[ic] = float(_field_chi(gw.G, K, gh, grid).real)
        f_direct = float(_field_chi(gw.G, K, gf, grid).real)
        MT1[ic] = float(_field_chi(gw.G, K, gmt, grid).real)
        AL1[ic] = float(_field_chi(gw.G, K, gal1+gal2, grid).real)

        fd = fock_bond_diagnostic(gw.G, Vq, K, 0, grid, reference)
        F1[ic] = float(np.real(fd["first"]))
        F2[ic] = float(np.real(fd["second"]))
        F_resum[ic] = float(np.real(fd["response"][-1] - fd["bubble"]))
        mismatch[ic] = abs(f_direct-F1[ic])
        if abs(float(np.real(fd["bubble"])) - bubble_gw[ic]) > 5e-9:
            raise RuntimeError(f"Fock bond bubble mismatch for {ch} at V={V:g}")

        sol = solve_vertex_q0_tail(gw.G, W, Vq, K, grid, reference, opts=opts)
        if not sol.converged:
            raise RuntimeError(
                f"full cGW failed for {ch} at V={V:g}: {sol.final_error:.3e}"
            )
        full_total[ic] = float(
            susceptibility_matrix_q0(
                gw.G, np.asarray([K]), [sol.Gamma], grid
            )[0, 0].real
        )
        full_vertex[ic] = full_total[ic] - bubble_gw[ic]

    subtotal2 = H1 + F1 + F2 + MT1 + AL1
    closure = full_vertex - subtotal2
    F3plus = F_resum - F1 - F2
    return gw, dict(
        V=float(V), mu_ed=float(mu_ed), mu_gw=float(gw.mu),
        gw_residual=float(gw.final_error), chi_ed=chi_ed,
        bubble_ed=bubble_ed, exact_vertex=exact_vertex,
        bubble_gw=bubble_gw, H1=H1, F1=F1, F2=F2, MT1=MT1, AL1=AL1,
        F_resummed=F_resum, F3plus=F3plus,
        full_total=full_total, full_vertex=full_vertex,
        subtotal2=subtotal2, closure=closure, fock_one_mismatch=mismatch,
    )


def _write_csv(path, values, channels, arrays):
    names = [
        "chi_ed", "bubble_ed", "exact_vertex", "bubble_gw", "H1", "F1",
        "F2", "MT1", "AL1", "F_resummed", "F3plus", "full_total",
        "full_vertex", "subtotal2", "closure",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["V", "channel", *names])
        for iv, V in enumerate(values):
            for ic, ch in enumerate(channels):
                w.writerow([
                    f"{V:.12g}", ch,
                    *[f"{arrays[n][iv, ic]:.16e}" for n in names],
                ])


def _plot(out, V, channels, A, dpi):
    fig, axes = plt.subplots(
        len(channels), 2, figsize=(13.5, 4.2*len(channels)), squeeze=False
    )
    for ic, ch in enumerate(channels):
        ax = axes[ic, 0]
        ax.plot(V, A["exact_vertex"][:,ic], "o-", label="exact vertex")
        ax.plot(V, A["full_vertex"][:,ic], "s--", label="full cGW vertex")
        ax.plot(V, A["F1"][:,ic], "^-", label="F(1)")
        ax.plot(V, A["F1"][:,ic]+A["F2"][:,ic], "v-", label="F(1)+F(2)")
        ax.set_title(ch); ax.set_xlabel("V"); ax.set_ylabel("static vertex correction")
        ax.grid(alpha=.25); ax.legend(fontsize=8)

        ax = axes[ic, 1]
        mask = V > 0
        for key, lab in (
            ("F2", "F(2)"), ("MT1", "MT(1)"), ("AL1", "AL(1)"),
            ("closure", "closure / higher order"),
        ):
            ax.plot(V[mask], A[key][mask,ic]/V[mask]**2, "o-", label=lab)
        miss = A["exact_vertex"] - A["full_vertex"]
        ax.plot(
            V[mask], miss[mask,ic]/V[mask]**2,
            "k--", linewidth=2, label="exact-cGW",
        )
        ax.axhline(0, color="0.4", linewidth=.8)
        ax.set_xlabel("V"); ax.set_ylabel(r"contribution / $V^2$")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(r"12-site static current $O(V^2)$ response ledger", y=.995)
    fig.tight_layout(); fig.savefig(out, dpi=int(dpi)); plt.close(fig)


def main():
    args = _args()
    values = np.array(sorted(set(float(x) for x in args.values)), dtype=float)
    if values[0] < -1e-15:
        raise ValueError("V values must be nonnegative")
    if 0.0 not in values:
        values = np.r_[0.0, values]
    if any(ch not in DEFAULT_CHANNELS for ch in args.channels):
        raise ValueError(
            "ledger is restricted to static z_same/z_opposite current channels"
        )
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(args.nw), nOmega=int(args.nomega), T=float(args.T)
    )
    seed = None; rows = []
    for i, V in enumerate(values):
        print(f"\n=== O(V^2) ledger point {i+1}/{len(values)}: V={V:g} ===", flush=True)
        seed, row = _run_point(V, seed, args, grid)
        rows.append(row)
        print(f"GW residual={row['gw_residual']:.3e}, mu={row['mu_gw']:.9f}")
        for ic, ch in enumerate(args.channels):
            print(
                f"{ch:12s} exactV={row['exact_vertex'][ic]:+.8e} "
                f"F1={row['F1'][ic]:+.8e} F2={row['F2'][ic]:+.8e} "
                f"MT1={row['MT1'][ic]:+.8e} AL1={row['AL1'][ic]:+.8e} "
                f"fullV={row['full_vertex'][ic]:+.8e} "
                f"closure={row['closure'][ic]:+.3e}"
            )

    keys = [k for k, v in rows[0].items() if isinstance(v, np.ndarray)]
    A = {k: np.stack([r[k] for r in rows]) for k in keys}
    scalars = {
        k: np.array([r[k] for r in rows])
        for k in ("mu_ed", "mu_gw", "gw_residual")
    }
    fits = {}
    linear_keys = ("exact_vertex", "full_vertex", "H1", "F1")
    quadratic_keys = ("F2", "MT1", "AL1", "closure", "F3plus")
    for ic, ch in enumerate(args.channels):
        cf = {}
        for k in linear_keys:
            cf[k] = _fit_abc(values, A[k][:,ic], args.fit_vmax)
        for k in quadratic_keys:
            cf[k] = _fit_bc(values, A[k][:,ic], args.fit_vmax)
        b_sub = (
            cf["H1"]["b"] + cf["F1"]["b"] + cf["F2"]["b"]
            + cf["MT1"]["b"] + cf["AL1"]["b"]
        )
        cf["b_subtotal"] = b_sub
        cf["b_missing_exact_minus_cgw"] = (
            cf["exact_vertex"]["b"] - cf["full_vertex"]["b"]
        )
        cf["b_full_minus_subtotal"] = cf["full_vertex"]["b"] - b_sub
        fits[ch] = cf

    print("\n=== weak-coupling coefficient ledger: Delta chi = a V + b V^2 + ... ===")
    for ch in args.channels:
        f = fits[ch]
        print(f"\n[{ch}] fit V<= {args.fit_vmax:g}")
        print(
            f"  exact vertex     a={f['exact_vertex']['a']:+.9e}  "
            f"b={f['exact_vertex']['b']:+.9e}"
        )
        print(
            f"  full cGW vertex  a={f['full_vertex']['a']:+.9e}  "
            f"b={f['full_vertex']['b']:+.9e}"
        )
        print("  additive cGW b ledger:")
        print(f"    H(1) dressing  {f['H1']['b']:+.9e}")
        print(f"    F(1) dressing  {f['F1']['b']:+.9e}")
        print(f"    F(2) feedback  {f['F2']['b']:+.9e}")
        print(f"    MT(1)          {f['MT1']['b']:+.9e}")
        print(f"    AL(1)          {f['AL1']['b']:+.9e}")
        print(f"    subtotal        {f['b_subtotal']:+.9e}")
        print(
            f"    full-subtotal   {f['b_full_minus_subtotal']:+.3e}  "
            "(fit/numerical O(V^3) leakage)"
        )
        print(f"  exact-full cGW    {f['b_missing_exact_minus_cgw']:+.9e}")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir/"ledger_scan.npz", V=values, channels=np.asarray(args.channels),
        fit_vmax=float(args.fit_vmax), **A, **scalars,
    )
    _write_csv(outdir/"ledger_scan.csv", values, args.channels, A)
    with (outdir/"ledger_fits.json").open("w", encoding="utf-8") as f:
        json.dump(fits, f, indent=2)
    _plot(outdir/"order2_ledger.png", values, args.channels, A, args.dpi)
    print(f"\nwrote {outdir/'ledger_scan.npz'}")
    print(f"wrote {outdir/'ledger_scan.csv'}")
    print(f"wrote {outdir/'ledger_fits.json'}")
    print(f"wrote {outdir/'order2_ledger.png'}")


if __name__ == "__main__":
    main()
