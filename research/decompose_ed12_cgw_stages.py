#!/usr/bin/env python3
"""Decompose the 12-site four-layer benchmark into cGW vertex stages.

This script reuses an existing ``diagnose_ed12_four_layer.py`` NPZ.  It does
*not* rerun exact diagonalization or the exact Lehmann Green function.  The
saved GW Green function is kept fixed, its production Hartree reference and W
are reconstructed, and the dynamic tail-consistent vertex is solved at every
represented external bosonic Matsubara frequency for

    GG -> H+F -> H+F+MT -> full(+AL).

For each channel we compare the stage vertex contribution

    C_stage - C_bubble[G_GW]

against the exact vertex contribution

    C_ED - C_bubble[G_ED].

This isolates which cGW stage produces the excessive current enhancement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.dynamic_cgw import DynamicVertexOptions, susceptibility_matrix_iomega
from rubycgw.ed_cgw_benchmark import bubble_iomega, bosonic_iomega_to_tau
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.production_dynamic_cgw import solve_vertex_iomega_tail
from rubycgw.response_tail import build_tail_reference
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    density_from_G_matrix,
    hartree_self_energy_matrix,
)


_STAGE_CONFIG = {
    "hf": dict(include_hartree=True, include_fock=True, include_mt=False, include_al=False),
    "mt": dict(include_hartree=True, include_fock=True, include_mt=True, include_al=False),
    "full": dict(include_hartree=True, include_fock=True, include_mt=True, include_al=True),
}
_STAGE_LABEL = {"gg": "GG", "hf": "H+F", "mt": "H+F+MT", "full": "full(+AL)"}
_CHANNEL_TEX = {
    "x_even": r"x_{\rm even}",
    "x_odd": r"x_{\rm odd}",
    "y_even": r"y_{\rm even}",
    "y_odd": r"y_{\rm odd}",
    "z_same": r"z_{\rm same}",
    "z_opposite": r"z_{\rm opposite}",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark", type=Path, help="existing 12-site four-layer NPZ")
    p.add_argument("--stages", nargs="+", choices=("hf", "mt", "full"), default=("hf", "mt", "full"))
    p.add_argument("--vertex-tol", type=float, default=1e-10)
    p.add_argument("--vertex-max-iter", type=int, default=400)
    p.add_argument("--hartree-tol", type=float, default=1e-12)
    p.add_argument("--hartree-max-iter", type=int, default=1000)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _first(d, *names):
    for name in names:
        if name in d.files:
            return d[name]
    raise KeyError(f"none of {names} found in NPZ; available keys: {d.files}")


def _scalar(d, *names):
    return float(np.asarray(_first(d, *names)).item())


def _strings(d, *names):
    return tuple(str(x) for x in np.asarray(_first(d, *names)).tolist())


def _interaction_matrix(exact: ExactSmallRubyThermal, V: float) -> np.ndarray:
    mat = np.zeros((exact.n_sites, exact.n_sites), dtype=complex)
    for i, j in exact.interaction_pairs:
        mat[i, j] = float(V)
        mat[j, i] = float(V)
    return mat[None, None]


def _recover_hartree(G, h0, Vq, grid, mu, tol, max_iter):
    """Recover the production Hartree field from the saved converged G."""
    norb = int(G.shape[-1])
    sigma = np.zeros((norb, norb), dtype=complex)
    err = float("inf")
    density = None
    for it in range(1, int(max_iter) + 1):
        density = density_from_G_matrix(G, grid, h0=h0, mu=float(mu), sigma_h=sigma)
        out = hartree_self_energy_matrix(density, Vq[0, 0])
        err = float(np.max(np.abs(out - sigma)))
        if err < float(tol):
            sigma = out
            break
        sigma = 0.5 * sigma + 0.5 * out
    else:
        raise RuntimeError(f"Hartree-reference recovery failed: residual={err:.3e}")
    density = density_from_G_matrix(G, grid, h0=h0, mu=float(mu), sigma_h=sigma)
    return sigma, np.asarray(density), int(it), float(err)


def _positive_to_full(pos: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    out = np.zeros((grid.nb, pos.shape[1]), dtype=complex)
    for im, mraw in enumerate(grid.m_values):
        m = int(mraw)
        out[im] = pos[m] if m >= 0 else np.conj(pos[-m])
    return out


def _solve_stages(G, W, Vq, vertices, grid, reference, stages, tol, max_iter):
    nc = len(vertices)
    mmax = int(grid.nOmega)
    data = {s: np.zeros((mmax + 1, nc), dtype=complex) for s in stages}
    iterations = {s: np.zeros((mmax + 1, nc), dtype=int) for s in stages}
    residuals = {s: np.zeros((mmax + 1, nc), dtype=float) for s in stages}

    ordered = [s for s in ("hf", "mt", "full") if s in stages]
    for ic, K in enumerate(vertices):
        prev_m = {s: None for s in ordered}
        for m in range(mmax + 1):
            current = None
            for stage in ordered:
                cfg = _STAGE_CONFIG[stage]
                initial = current if current is not None else prev_m[stage]
                print(
                    f"stage {_STAGE_LABEL[stage]:8s} channel {ic+1}/{nc}, "
                    f"m={m}/{mmax}"
                )
                opts = DynamicVertexOptions(
                    max_iter=int(max_iter),
                    tol=float(tol),
                    mixing=0.2,
                    solver="gmres",
                    gmres_restart=40,
                    verbose=False,
                    momentum_backend="direct",
                    **cfg,
                )
                res = solve_vertex_iomega_tail(
                    G, W, Vq, K, m, grid, reference,
                    opts=opts,
                    initial_gamma=initial,
                )
                if not res.converged:
                    raise RuntimeError(
                        f"{stage} cGW failed for channel={ic}, m={m}: "
                        f"residual={res.final_error:.3e}"
                    )
                data[stage][m, ic] = susceptibility_matrix_iomega(
                    G, np.asarray([K]), [res.Gamma], m, grid
                )[0, 0]
                iterations[stage][m, ic] = int(res.iterations)
                residuals[stage][m, ic] = float(res.final_error)
                current = res.Gamma
                prev_m[stage] = res.Gamma

    return (
        {s: _positive_to_full(data[s], grid) for s in stages},
        iterations,
        residuals,
    )


def _plot(tau, beta, channels, exactC, bubbleED, bubbleGW, stage_tau, out, dpi, qlabel):
    x = np.asarray(tau) / float(beta)
    nc = len(channels)
    fig, axes = plt.subplots(nc, 2, figsize=(13.5, max(3.4 * nc, 4.0)), squeeze=False)
    stage_order = [s for s in ("hf", "mt", "full") if s in stage_tau]
    for ic, ch in enumerate(channels):
        tex = _CHANNEL_TEX.get(ch, ch.replace("_", r"\_"))
        ax = axes[ic, 0]
        ax.plot(x, np.real(exactC[:, ic]), label="exact ED")
        ax.plot(x, np.real(bubbleGW[:, ic]), linestyle="--", label="GG")
        for stage in stage_order:
            ax.plot(x, np.real(stage_tau[stage][:, ic]), label=_STAGE_LABEL[stage])
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(rf"$C_{{{tex},{tex}}}(\tau)$")
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax = axes[ic, 1]
        exact_vertex = exactC[:, ic] - bubbleED[:, ic]
        ax.plot(x, np.real(exact_vertex), linewidth=2.2, label="exact vertex")
        for stage in stage_order:
            ax.plot(
                x,
                np.real(stage_tau[stage][:, ic] - bubbleGW[:, ic]),
                label=f"{_STAGE_LABEL[stage]} vertex",
            )
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("vertex contribution")
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(f"12-site cGW stage decomposition at {qlabel}", y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    d = np.load(args.benchmark, allow_pickle=False)

    V = _scalar(d, "V")
    filling = _scalar(d, "filling")
    T = _scalar(d, "temperature", "T")
    beta = _scalar(d, "beta")
    L1 = int(round(_scalar(d, "L1")))
    L2 = int(round(_scalar(d, "L2")))
    channels = _strings(d, "channels")
    q = np.asarray(_first(d, "q"), dtype=float).reshape(2)
    qlabel = str(np.asarray(_first(d, "q_label", "qlabel")).item())
    tau = np.asarray(_first(d, "tau_grid", "tau"), dtype=float)
    mu_gw = _scalar(d, "gw_mu", "mu_gw")

    exactC = np.asarray(_first(d, "exact_correlation_tau", "exact_C_tau"), dtype=complex)
    bubbleED = np.asarray(_first(d, "bubble_exact_tau", "bubble_ed_C_tau"), dtype=complex)
    bubbleGW_saved = np.asarray(_first(d, "bubble_gw_tau", "bubble_gw_C_tau"), dtype=complex)
    full_saved = np.asarray(_first(d, "full_cgw_tau", "full_cgw_C_tau"), dtype=complex)

    Graw = np.asarray(_first(d, "gw_G_iomega", "G_gw"), dtype=complex)
    if Graw.ndim == 3:
        G = Graw[:, None, None, :, :]
    elif Graw.ndim == 5:
        G = Graw
    else:
        raise ValueError(f"unexpected saved GW G shape {Graw.shape}")
    nw = G.shape[0] // 2
    if "m_values" in d.files:
        mvals = np.asarray(d["m_values"], dtype=int)
        nomega = int(np.max(np.abs(mvals)))
    else:
        full_iw = np.asarray(_first(d, "full_cgw_iomega"))
        nomega = (full_iw.shape[0] - 1) // 2
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=int(nw), nOmega=int(nomega), T=float(T))

    ti = _scalar(d, "ti") if "ti" in d.files else 0.4
    t1 = _scalar(d, "t1") if "t1" in d.files else 0.2
    t2 = _scalar(d, "t2") if "t2" in d.files else 0.2
    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    exact = ExactSmallRubyThermal(L1, L2, params)
    h0 = np.asarray(exact.h0, dtype=complex)[None, None]
    Vq = _interaction_matrix(exact, V)
    vertices = [np.asarray(exact.pseudospin_operator(ch, q), dtype=complex) for ch in channels]

    sigma_h, density, hit, herr = _recover_hartree(
        G, h0, Vq, grid, mu_gw, args.hartree_tol, args.hartree_max_iter
    )
    print(
        f"recovered production Hartree reference: it={hit}, residual={herr:.3e}, "
        f"N={np.sum(density):.10f} (target={filling*L1*L2:.10f})"
    )
    P = compute_polarization_matrix(G, grid, backend="direct")
    W = compute_screened_interaction_matrix(P, Vq)
    reference = build_tail_reference(h0, mu_gw, sigma_h, grid)

    vertices_arr = np.asarray(vertices)
    bubbleGW_iw = np.diagonal(bubble_iomega(G, vertices_arr, grid), axis1=-2, axis2=-1)
    bubbleGW = bosonic_iomega_to_tau(bubbleGW_iw, grid, tau)
    gg_rel = float(
        np.linalg.norm((bubbleGW - bubbleGW_saved).ravel())
        / max(np.linalg.norm(bubbleGW_saved.ravel()), 1e-300)
    )
    print(f"saved-vs-reconstructed GG C(tau) relative difference = {gg_rel:.3e}")

    stages = tuple(dict.fromkeys(args.stages))
    stage_iw, iterations, residuals = _solve_stages(
        G, W, Vq, vertices, grid, reference, stages,
        args.vertex_tol, args.vertex_max_iter,
    )
    stage_tau = {s: bosonic_iomega_to_tau(stage_iw[s], grid, tau) for s in stages}

    if "full" in stage_tau:
        full_rel = float(
            np.linalg.norm((stage_tau["full"] - full_saved).ravel())
            / max(np.linalg.norm(full_saved.ravel()), 1e-300)
        )
        print(f"saved-vs-recomputed full C(tau) relative difference = {full_rel:.3e}")

    mid = int(np.argmin(np.abs(tau - beta / 2.0)))
    izero = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
    print("\n=== midpoint vertex contribution C(beta/2)-GG baseline ===")
    header = "channel        exact vertex"
    for s in stages:
        header += f"   {_STAGE_LABEL[s]:>12s}"
    print(header)
    for ic, ch in enumerate(channels):
        vex = float((exactC[mid, ic] - bubbleED[mid, ic]).real)
        line = f"{ch:13s} {vex:+.8e}"
        for s in stages:
            val = float((stage_tau[s][mid, ic] - bubbleGW[mid, ic]).real)
            line += f"  {val:+.8e}"
        print(line)

    exact_static = np.asarray(_first(d, "exact_static_susceptibility", "exact_chi0"), dtype=complex)
    bubbleED_iw = np.asarray(_first(d, "bubble_exact_iomega", "bubble_ed_iomega"), dtype=complex)
    print("\n=== static vertex contribution chi(0)-GG baseline ===")
    print(header)
    for ic, ch in enumerate(channels):
        vex = float((exact_static[ic] - bubbleED_iw[izero, ic]).real)
        line = f"{ch:13s} {vex:+.8e}"
        for s in stages:
            val = float((stage_iw[s][izero, ic] - bubbleGW_iw[izero, ic]).real)
            line += f"  {val:+.8e}"
        print(line)

    out = args.out if args.out is not None else args.benchmark.with_name(args.benchmark.stem + "_stages.npz")
    out = Path(out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": np.asarray(str(args.benchmark)),
        "channels": np.asarray(channels),
        "q": np.asarray(q),
        "q_label": np.asarray(qlabel),
        "tau": np.asarray(tau),
        "beta": float(beta),
        "m_values": np.asarray(grid.m_values),
        "Omega": np.asarray(grid.Omega),
        "bubble_gw_tau": bubbleGW,
        "bubble_gw_iomega": bubbleGW_iw,
        "exact_vertex_tau": exactC - bubbleED,
        "exact_vertex_static": exact_static - bubbleED_iw[izero],
        "reconstructed_hartree": sigma_h,
        "reconstructed_density": density,
        "reconstructed_gg_relative_difference": float(gg_rel),
    }
    for s in stages:
        payload[f"{s}_iomega"] = stage_iw[s]
        payload[f"{s}_tau"] = stage_tau[s]
        payload[f"{s}_vertex_tau"] = stage_tau[s] - bubbleGW
        payload[f"{s}_vertex_static"] = stage_iw[s][izero] - bubbleGW_iw[izero]
        payload[f"{s}_iterations"] = iterations[s]
        payload[f"{s}_residuals"] = residuals[s]
    np.savez_compressed(out, **payload)

    png = out.with_suffix(".png")
    _plot(tau, beta, channels, exactC, bubbleED, bubbleGW, stage_tau, png, args.dpi, qlabel)
    print(f"saved: {out}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
