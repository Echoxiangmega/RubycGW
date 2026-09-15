#!/usr/bin/env python3
"""Detailed 12-site cGW diagnostic: H vs F and one-shot vs resummed kernels.

Reuses ``ed12_four_layer.npz``.  No exact diagonalization or exact Lehmann G is
recomputed.  On the saved SC-GW background this script evaluates

    GG -> H -> F -> H+F -> H+F+MT -> full(+AL)

as separately resummed approximations, and also applies the full tail-consistent
kernel only once to the bare source K.  The one-shot response is decomposed into
individual H, F, MT and AL contributions.  This distinguishes a large bare
kernel action from amplification generated mainly by repeated resummation.

Important: the resummed H-only/F-only curves are separate approximate theories,
not additive diagram contributions.  The one-shot H/F/MT/AL table is the proper
linear decomposition of the first kernel application.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import decompose_ed12_cgw_stages as base
from rubycgw.dynamic_cgw import (
    DynamicVertexOptions,
    _bare_vertex_field_iomega,
    _mask_external_window,
    _x_field_iomega,
    susceptibility_matrix_iomega,
)
from rubycgw.ed_cgw_benchmark import bubble_iomega, bosonic_iomega_to_tau
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.production_dynamic_cgw import _dynamic_parts, solve_vertex_iomega_tail
from rubycgw.response_tail import build_tail_hf_context, build_tail_reference
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_gw import compute_polarization_matrix, compute_screened_interaction_matrix


_STAGE_CONFIG = {
    "h": dict(include_hartree=True, include_fock=False, include_mt=False, include_al=False),
    "f": dict(include_hartree=False, include_fock=True, include_mt=False, include_al=False),
    "hf": dict(include_hartree=True, include_fock=True, include_mt=False, include_al=False),
    "mt": dict(include_hartree=True, include_fock=True, include_mt=True, include_al=False),
    "full": dict(include_hartree=True, include_fock=True, include_mt=True, include_al=True),
}
_STAGE_ORDER = ("h", "f", "hf", "mt", "full")
_STAGE_LABEL = {
    "h": "H only",
    "f": "F only",
    "hf": "H+F",
    "mt": "H+F+MT",
    "full": "full(+AL)",
}
_PART_LABEL = ("H(1)", "F(1)", "MT(1)", "AL(1)", "total(1)")
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
    p.add_argument("benchmark", type=Path)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=_STAGE_ORDER,
        default=_STAGE_ORDER,
        help="separately resummed stages to solve",
    )
    p.add_argument("--vertex-tol", type=float, default=1e-10)
    p.add_argument("--vertex-max-iter", type=int, default=400)
    p.add_argument("--hartree-tol", type=float, default=1e-12)
    p.add_argument("--hartree-max-iter", type=int, default=1000)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--out", type=Path, default=Path("ed12_hf_detail.npz"))
    return p.parse_args()


def _positive_to_full(pos: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    out = np.zeros((grid.nb,) + pos.shape[1:], dtype=complex)
    for im, mraw in enumerate(grid.m_values):
        m = int(mraw)
        out[im] = pos[m] if m >= 0 else np.conj(pos[-m])
    return out


def _opts(cfg, args):
    return DynamicVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        mixing=0.2,
        solver="gmres",
        gmres_restart=40,
        verbose=False,
        momentum_backend="direct",
        **cfg,
    )


def _chi_of_field(G, K, field, m, grid):
    return susceptibility_matrix_iomega(
        G, np.asarray([K]), [field], int(m), grid
    )[0, 0]


def _one_shot_parts(G, W, Vq, K, m, grid, reference, args):
    """Return response contributions from one application of the full kernel to K."""
    opts = _opts(_STAGE_CONFIG["full"], args)
    Kfield = _bare_vertex_field_iomega(K, G, int(m), grid)
    ctx = build_tail_hf_context(
        reference,
        K,
        Vq,
        grid,
        q_index=(0, 0),
        m_ext=int(m),
        backend="direct",
        include_hartree=True,
        include_fock=True,
    )
    X0 = _x_field_iomega(G, Kfield, int(m), grid)
    gh, gf = ctx.total_vertex_parts(X0, G)
    gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X0, int(m), grid, opts)
    gh, gf, gmt, gal1, gal2 = tuple(
        _mask_external_window(x, grid, int(m)) for x in (gh, gf, gmt, gal1, gal2)
    )
    gal = gal1 + gal2
    total = gh + gf + gmt + gal
    return np.asarray(
        [
            _chi_of_field(G, K, gh, m, grid),
            _chi_of_field(G, K, gf, m, grid),
            _chi_of_field(G, K, gmt, m, grid),
            _chi_of_field(G, K, gal, m, grid),
            _chi_of_field(G, K, total, m, grid),
        ],
        dtype=complex,
    )


def _plot(tau, beta, channels, exactC, bubbleED, bubbleGW, stage_tau, part_tau, out, dpi, qlabel):
    x = np.asarray(tau) / float(beta)
    nc = len(channels)
    fig, axes = plt.subplots(nc, 2, figsize=(14.5, max(3.5 * nc, 4.0)), squeeze=False)
    stages = [s for s in _STAGE_ORDER if s in stage_tau]
    for ic, ch in enumerate(channels):
        tex = _CHANNEL_TEX.get(ch, ch.replace("_", r"\_"))
        ax = axes[ic, 0]
        ax.plot(x, np.real(exactC[:, ic]), linewidth=2.0, label="exact ED")
        ax.plot(x, np.real(bubbleGW[:, ic]), linestyle="--", label="GG")
        for s in stages:
            ax.plot(x, np.real(stage_tau[s][:, ic]), label=_STAGE_LABEL[s])
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(rf"$C_{{{tex},{tex}}}(\tau)$")
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[ic, 1]
        ax.plot(
            x,
            np.real(exactC[:, ic] - bubbleED[:, ic]),
            linewidth=2.2,
            label="exact vertex",
        )
        for ip, lab in enumerate(_PART_LABEL[:-1]):
            ax.plot(x, np.real(part_tau[:, ic, ip]), label=lab)
        ax.plot(
            x,
            np.real(part_tau[:, ic, 4]),
            linestyle="--",
            linewidth=2.0,
            label="total first application",
        )
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("one-shot vertex contribution")
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"12-site H/F detail and first-kernel diagnostic at {qlabel}", y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    d = np.load(args.benchmark, allow_pickle=False)

    V = base._scalar(d, "V")
    filling = base._scalar(d, "filling")
    T = base._scalar(d, "temperature", "T")
    beta = base._scalar(d, "beta")
    L1 = int(round(base._scalar(d, "L1")))
    L2 = int(round(base._scalar(d, "L2")))
    channels = base._strings(d, "channels")
    q = np.asarray(base._first(d, "q"), dtype=float).reshape(2)
    qlabel = str(np.asarray(base._first(d, "q_label", "qlabel")).item())
    tau = np.asarray(base._first(d, "tau_grid", "tau"), dtype=float)
    mu_gw = base._scalar(d, "gw_mu", "mu_gw")

    exactC = np.asarray(base._first(d, "exact_correlation_tau", "exact_C_tau"), dtype=complex)
    bubbleED = np.asarray(base._first(d, "bubble_exact_tau", "bubble_ed_C_tau"), dtype=complex)
    bubbleGW_saved = np.asarray(base._first(d, "bubble_gw_tau", "bubble_gw_C_tau"), dtype=complex)
    exact_static = np.asarray(
        base._first(d, "exact_static_susceptibility", "exact_chi0"), dtype=complex
    )
    bubbleED_iw = np.asarray(
        base._first(d, "bubble_exact_iomega", "bubble_ed_iomega"), dtype=complex
    )

    Graw = np.asarray(base._first(d, "gw_G_iomega", "G_gw"), dtype=complex)
    if Graw.ndim == 3:
        G = Graw[:, None, None, :, :]
    elif Graw.ndim == 5:
        G = Graw
    else:
        raise ValueError(f"unexpected saved GW G shape {Graw.shape}")
    nw = G.shape[0] // 2
    mvals = np.asarray(base._first(d, "m_values"), dtype=int)
    nomega = int(np.max(np.abs(mvals)))
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=int(nw), nOmega=nomega, T=float(T))

    ti = base._scalar(d, "ti") if "ti" in d.files else 0.4
    t1 = base._scalar(d, "t1") if "t1" in d.files else 0.2
    t2 = base._scalar(d, "t2") if "t2" in d.files else 0.2
    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    exact = ExactSmallRubyThermal(L1, L2, params)
    h0 = np.asarray(exact.h0, dtype=complex)[None, None]
    Vq = base._interaction_matrix(exact, V)
    vertices = [np.asarray(exact.pseudospin_operator(ch, q), dtype=complex) for ch in channels]

    sigma_h, density, hit, herr = base._recover_hartree(
        G,
        h0,
        Vq,
        grid,
        mu_gw,
        args.hartree_tol,
        args.hartree_max_iter,
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

    stages = tuple(s for s in _STAGE_ORDER if s in tuple(dict.fromkeys(args.stages)))
    nc = len(channels)
    mmax = int(grid.nOmega)
    stage_pos = {s: np.zeros((mmax + 1, nc), dtype=complex) for s in stages}
    stage_iter = {s: np.zeros((mmax + 1, nc), dtype=int) for s in stages}
    stage_res = {s: np.zeros((mmax + 1, nc), dtype=float) for s in stages}
    part_pos = np.zeros((mmax + 1, nc, 5), dtype=complex)

    for ic, K in enumerate(vertices):
        previous = {s: None for s in stages}
        for m in range(mmax + 1):
            print(f"one-shot parts channel {ic+1}/{nc}, m={m}/{mmax}")
            part_pos[m, ic] = _one_shot_parts(G, W, Vq, K, m, grid, reference, args)
            for s in stages:
                print(f"resummed {_STAGE_LABEL[s]:8s} channel {ic+1}/{nc}, m={m}/{mmax}")
                res = solve_vertex_iomega_tail(
                    G,
                    W,
                    Vq,
                    K,
                    m,
                    grid,
                    reference,
                    opts=_opts(_STAGE_CONFIG[s], args),
                    initial_gamma=previous[s],
                )
                if not res.converged:
                    raise RuntimeError(
                        f"{s} cGW failed channel={ic}, m={m}: residual={res.final_error:.3e}"
                    )
                stage_pos[s][m, ic] = _chi_of_field(G, K, res.Gamma, m, grid)
                stage_iter[s][m, ic] = int(res.iterations)
                stage_res[s][m, ic] = float(res.final_error)
                previous[s] = res.Gamma

    stage_iw = {s: _positive_to_full(stage_pos[s], grid) for s in stages}
    stage_tau = {s: bosonic_iomega_to_tau(stage_iw[s], grid, tau) for s in stages}
    part_iw = _positive_to_full(part_pos, grid)
    part_tau = np.empty((len(tau), nc, 5), dtype=complex)
    for ip in range(5):
        part_tau[:, :, ip] = bosonic_iomega_to_tau(part_iw[:, :, ip], grid, tau)

    mid = int(np.argmin(np.abs(tau - beta / 2.0)))
    izero = int(np.where(np.asarray(grid.m_values) == 0)[0][0])

    print("\n=== resummed midpoint vertex contribution C(beta/2)-GG ===")
    header = "channel        exact vertex" + "".join(f"  {_STAGE_LABEL[s]:>12s}" for s in stages)
    print(header)
    for ic, ch in enumerate(channels):
        line = f"{ch:13s} {float((exactC[mid,ic]-bubbleED[mid,ic]).real):+.8e}"
        for s in stages:
            line += f"  {float((stage_tau[s][mid,ic]-bubbleGW[mid,ic]).real):+.8e}"
        print(line)

    print("\n=== resummed static vertex contribution chi(0)-GG ===")
    print(header)
    for ic, ch in enumerate(channels):
        line = f"{ch:13s} {float((exact_static[ic]-bubbleED_iw[izero,ic]).real):+.8e}"
        for s in stages:
            line += f"  {float((stage_iw[s][izero,ic]-bubbleGW_iw[izero,ic]).real):+.8e}"
        print(line)

    print("\n=== one-shot midpoint contribution from L[K] ===")
    print("channel        exact vertex        H(1)          F(1)         MT(1)         AL(1)      total(1)")
    for ic, ch in enumerate(channels):
        vals = [float(part_tau[mid, ic, ip].real) for ip in range(5)]
        print(
            f"{ch:13s} {float((exactC[mid,ic]-bubbleED[mid,ic]).real):+.8e}  "
            + "  ".join(f"{v:+.8e}" for v in vals)
        )

    print("\n=== one-shot static contribution from L[K] ===")
    print("channel        exact vertex        H(1)          F(1)         MT(1)         AL(1)      total(1)")
    for ic, ch in enumerate(channels):
        vals = [float(part_iw[izero, ic, ip].real) for ip in range(5)]
        print(
            f"{ch:13s} {float((exact_static[ic]-bubbleED_iw[izero,ic]).real):+.8e}  "
            + "  ".join(f"{v:+.8e}" for v in vals)
        )

    # TR diagnostic: current channels should have negligible Hartree response.
    print("\n=== Hartree TR diagnostic (one-shot static H contribution) ===")
    for ic, ch in enumerate(channels):
        print(f"{ch:13s} H(1)={part_iw[izero,ic,0].real:+.8e}")

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        source_benchmark=np.asarray(str(args.benchmark)),
        channels=np.asarray(channels),
        q=np.asarray(q),
        q_label=np.asarray(qlabel),
        tau=tau,
        beta=float(beta),
        m_values=np.asarray(grid.m_values),
        bubble_gw_iomega=bubbleGW_iw,
        bubble_gw_tau=bubbleGW,
        exact_C_tau=exactC,
        bubble_ed_C_tau=bubbleED,
        exact_static=exact_static,
        bubble_ed_iomega=bubbleED_iw,
        one_shot_part_names=np.asarray(_PART_LABEL),
        one_shot_part_iomega=part_iw,
        one_shot_part_tau=part_tau,
    )
    for s in stages:
        payload[f"stage_{s}_iomega"] = stage_iw[s]
        payload[f"stage_{s}_tau"] = stage_tau[s]
        payload[f"stage_{s}_iterations"] = stage_iter[s]
        payload[f"stage_{s}_residuals"] = stage_res[s]
    np.savez_compressed(out, **payload)

    png = out.with_suffix(".png")
    _plot(
        tau,
        beta,
        channels,
        exactC,
        bubbleED,
        bubbleGW,
        stage_tau,
        part_tau,
        png,
        args.dpi,
        qlabel,
    )
    print(f"saved: {out}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
