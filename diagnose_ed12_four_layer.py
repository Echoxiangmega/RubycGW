#!/usr/bin/env python3
"""Four-layer 12-site diagnostic: exact ED, exact-G bubble, GW bubble, full cGW.

The default geometry is the ordinary 2x1 primitive-cell Ruby torus (12 sites).
Unlike the 18-site thermal-typicality benchmark, every particle-number sector is
fully diagonalized, so the grand-canonical one-particle Green function is obtained
from an exact Lehmann representation up to a user-controlled discarded Boltzmann
weight.

For each requested Hermitian folded momentum/channel this script compares

  1. C_exact(tau)                         exact interacting two-body correlation,
  2. C_bubble[G_exact](tau)               exact one-particle background, no vertex,
  3. C_bubble[G_GW](tau)                  GW one-particle background, no vertex,
  4. C_cGW[G_GW](tau)                     full tail-consistent H/F/MT/AL response.

The useful decomposition is then

  background error    = bubble[G_exact] - bubble[G_GW],
  exact vertex effect = C_exact          - bubble[G_exact],
  cGW vertex effect   = C_cGW            - bubble[G_GW].

If bubble[G_exact] and bubble[G_GW] are close while the exact and cGW vertex
effects differ strongly, the main problem lies in the two-particle kernel rather
than the one-particle GW background.

Examples
--------
Default Gamma comparison at n=3, V=1, T=0.08::

    python diagnose_ed12_four_layer.py ^
      --V 1 --filling 3 --T 0.08 ^
      --channels x_even z_same z_opposite ^
      --out ed12_four_layer.npz

The 2x1 torus also supports the folded primitive momentum M1=(1/2,0)::

    python diagnose_ed12_four_layer.py --q m1 --out ed12_M1_four_layer.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.dynamic_cgw import DynamicVertexOptions, susceptibility_matrix_iomega
from rubycgw.ed_cgw_benchmark import bubble_iomega, bosonic_iomega_to_tau
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.production_dynamic_cgw import solve_vertex_iomega_tail
from rubycgw.response_tail import build_tail_reference
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson


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
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=3.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--q", choices=("gamma", "m1"), default="gamma")
    p.add_argument(
        "--channels",
        nargs="+",
        default=("x_even", "z_same", "z_opposite"),
    )
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--tau-points", type=int, default=101)
    p.add_argument("--thermal-weight-tol", type=float, default=1e-12)
    p.add_argument("--gw-tol", type=float, default=1e-9)
    p.add_argument("--gw-max-iter", type=int, default=1200)
    p.add_argument("--vertex-tol", type=float, default=1e-10)
    p.add_argument("--vertex-max-iter", type=int, default=400)
    p.add_argument("--vertex-workers", type=int, default=1)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--out", type=Path, default=Path("ed12_four_layer.npz"))
    return p.parse_args()


def _q_value(args):
    if args.q == "gamma":
        return np.array([0.0, 0.0]), "Gamma"
    if args.q == "m1":
        return np.array([0.5, 0.0]), "M1=(1/2,0)"
    raise ValueError(args.q)


def _interaction_matrix_cluster(exact: ExactSmallRubyThermal, params: RubyParameters):
    norb = exact.n_sites
    V = np.zeros((norb, norb), dtype=complex)
    for i, j in exact.interaction_pairs:
        V[i, j] = float(params.V)
        V[j, i] = float(params.V)
    return V[None, None]


def _h0_cluster_field(exact: ExactSmallRubyThermal):
    return np.asarray(exact.h0, dtype=complex)[None, None]


def _cluster_vertex(exact: ExactSmallRubyThermal, channel: str, q: np.ndarray):
    return np.asarray(exact.operator_matrix(channel, q), dtype=complex)


def _gw_continuation(exact, params, grid, target_N, args):
    h0 = _h0_cluster_field(exact)
    seed = None
    schedule = [v for v in (0.1, 0.25, 0.5, 0.75, float(params.V)) if v <= float(params.V) + 1e-12]
    schedule = sorted(set(round(float(v), 12) for v in schedule))
    for V in schedule:
        p = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=V)
        Vq = _interaction_matrix_cluster(exact, p)
        opts = GWOptions(
            mu=0.0 if seed is None else float(seed.mu),
            target_filling=float(target_N),
            max_iter=int(args.gw_max_iter),
            tol=float(args.gw_tol),
            mixing=0.2,
            mixing_method="linear",
            mu_tol=min(1e-10, float(args.gw_tol) * 0.1),
            mu_max_iter=120,
            verbose=False,
            momentum_backend="direct",
        )
        seed = solve_matrix_gw_anderson(h0, Vq, grid, opts=opts, initial=seed, anderson=AndersonOptions())
        print(
            f"SC-GW continuation V={V:g}\n"
            f"  converged={seed.converged}, it={seed.iterations}, residual={seed.final_error:.3e}, "
            f"mu={seed.mu:.10f}, N={np.sum(seed.density):.10f}"
        )
        if not seed.converged:
            raise RuntimeError(f"SC-GW failed at V={V:g}: residual={seed.final_error:.3e}")
    return seed, _interaction_matrix_cluster(exact, params), h0


def _relative_g_error(Gref, Gtest, grid):
    ref = np.asarray(Gref, dtype=complex)
    test = np.asarray(Gtest, dtype=complex)
    denom = max(float(np.linalg.norm(ref.ravel())), 1e-300)
    full = float(np.linalg.norm((test - ref).ravel()) / denom)
    order = np.argsort(np.abs(grid.omega))
    nlow = min(8, len(order))
    idx = order[:nlow]
    dlow = max(float(np.linalg.norm(ref[idx].ravel())), 1e-300)
    low = float(np.linalg.norm((test[idx] - ref[idx]).ravel()) / dlow)
    return full, low


def _cgw_dynamic(gw, Vq, h0, vertices, grid, args):
    reference = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)
    nc = len(vertices)
    mmax = int(grid.nOmega)
    pos = {}
    iterations = np.zeros((mmax + 1, nc), dtype=int)
    residuals = np.zeros((mmax + 1, nc), dtype=float)
    for ic, K in enumerate(vertices):
        previous = None
        for m in range(mmax + 1):
            print(f"full cGW channel {ic+1}/{nc}, m={m}/{mmax}")
            opts = DynamicVertexOptions(
                max_iter=int(args.vertex_max_iter),
                tol=float(args.vertex_tol),
                mixing=0.2,
                solver="gmres",
                gmres_restart=40,
                include_hartree=True,
                include_fock=True,
                include_mt=True,
                include_al=True,
                verbose=False,
                momentum_backend="direct",
            )
            res = solve_vertex_iomega_tail(
                gw.G,
                gw.W,
                Vq,
                K,
                m,
                grid,
                reference,
                opts=opts,
                initial_gamma=previous,
            )
            if not res.converged:
                raise RuntimeError(
                    f"full cGW failed for channel={ic}, m={m}: residual={res.final_error:.3e}"
                )
            pos.setdefault(m, np.zeros(nc, dtype=complex))[ic] = susceptibility_matrix_iomega(
                gw.G,
                np.asarray([K]),
                [res.Gamma],
                m,
                grid,
            )[0, 0]
            iterations[m, ic] = int(res.iterations)
            residuals[m, ic] = float(res.final_error)
            previous = res.Gamma

    full = np.zeros((grid.nb, nc), dtype=complex)
    for im, mraw in enumerate(grid.m_values):
        m = int(mraw)
        full[im] = pos[m] if m >= 0 else np.conj(pos[-m])
    return full, iterations, residuals


def _plot(tau, beta, channels, exactC, bubbleED, bubbleGW, fullC, out, dpi, qlabel):
    x = np.asarray(tau) / float(beta)
    nc = len(channels)
    fig, axes = plt.subplots(nc, 2, figsize=(13.5, max(3.4 * nc, 4.0)), squeeze=False)
    for ic, ch in enumerate(channels):
        label = _CHANNEL_TEX.get(str(ch), str(ch).replace("_", r"\_"))
        ax = axes[ic, 0]
        ax.plot(x, np.real(exactC[:, ic]), label="exact ED")
        ax.plot(x, np.real(bubbleED[:, ic]), linestyle="--", label=r"bubble[$G_{ED}$]")
        ax.plot(x, np.real(bubbleGW[:, ic]), linestyle="-.", label=r"bubble[$G_{GW}$]")
        ax.plot(x, np.real(fullC[:, ic]), linestyle=":", linewidth=2.2, label="full cGW")
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(rf"$C_{{{label},{label}}}(\tau)$")
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax = axes[ic, 1]
        ax.plot(x, np.real(bubbleED[:, ic] - bubbleGW[:, ic]), label="background: bubble ED-GW")
        ax.plot(x, np.real(exactC[:, ic] - bubbleED[:, ic]), label="exact vertex: ED-bubbleED")
        ax.plot(x, np.real(fullC[:, ic] - bubbleGW[:, ic]), label="cGW vertex: full-bubbleGW")
        ax.plot(x, np.real(fullC[:, ic] - exactC[:, ic]), linestyle="--", label="total cGW-ED")
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.set_ylabel("difference")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"12-site 2x1 torus four-layer diagnostic at {qlabel}", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    if args.T <= 0.0:
        raise ValueError("T must be positive")
    if args.tau_points < 3 or args.tau_points % 2 != 1:
        raise ValueError("tau-points must be odd and >=3")
    if args.L1 * args.L2 * 6 != 12:
        raise ValueError("this diagnostic is intentionally restricted to exactly 12 sites")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    target_N = float(args.filling) * exact.n_cells
    if abs(target_N - round(target_N)) > 1e-12:
        raise ValueError("filling*number_of_cells must be integer for the target n=3 diagnostic")
    q, qlabel = _q_value(args)
    channels = tuple(str(x) for x in args.channels)

    grid = MatsubaraGrid(nk1=1, nk2=1, nw=int(args.nw), nOmega=int(args.nomega), T=float(args.T))
    beta = 1.0 / float(args.T)
    tau = np.linspace(0.0, beta, int(args.tau_points))

    print("=" * 88)
    print("12-site exact-G four-layer benchmark")
    print(
        f"geometry={args.L1}x{args.L2} primitive cells, Nsite={exact.n_sites}, "
        f"V={args.V:g}, filling={args.filling:g}, target N={target_N:g}, T={args.T:g}"
    )
    print(f"q={qlabel}; channels={', '.join(channels)}")
    print(f"Matsubara: nw={grid.nw}, nOmega={grid.nOmega}")
    print("=" * 88)

    t0 = time.perf_counter()
    exact.diagonalize_all()
    print(f"full sector ED diagonalization: {time.perf_counter()-t0:.2f} s")
    mu_ed = exact.solve_mu_for_mean_number(target_N, float(args.T))
    weights = exact.thermal_weights(mu_ed, float(args.T), tol=float(args.thermal_weight_tol))
    print(
        f"exact GC: mu={mu_ed:.10f}, <N>={weights.mean_number:.10f}, "
        f"Var(N)={weights.var_number:.6e}, discarded thermal weight={weights.discarded_weight:.3e}"
    )
    print("exact particle-sector weights:")
    for N, pN in weights.sector_probabilities.items():
        if pN > 1e-8:
            print(f"  N={N:2d}: p={pN:.8e}")

    t0 = time.perf_counter()
    G_ed = exact.green_iomega(mu_ed, float(args.T), grid.omega, weights)
    print(
        f"exact Lehmann G(iw): {time.perf_counter()-t0:.2f} s, "
        f"discarded weight={weights.discarded_weight:.3e}"
    )

    exactC = np.zeros((len(tau), len(channels)), dtype=complex)
    chi_exact = np.zeros(len(channels), dtype=complex)
    means = np.zeros(len(channels), dtype=complex)
    for ic, ch in enumerate(channels):
        t0 = time.perf_counter()
        exactC[:, ic], means[ic] = exact.correlation_tau(
            ch, q, mu_ed, float(args.T), tau, weights
        )
        chi_exact[ic], _ = exact.static_susceptibility(
            ch, q, mu_ed, float(args.T), weights
        )
        print(
            f"exact C {ch:12s}: chi0={chi_exact[ic].real:+.8e}, "
            f"<O>={means[ic].real:+.3e}{means[ic].imag:+.3e}j, "
            f"time={time.perf_counter()-t0:.2f}s"
        )

    gw, Vq, h0 = _gw_continuation(exact, params, grid, target_N, args)
    Gerr_full, Gerr_low = _relative_g_error(G_ed, gw.G, grid)
    print(
        f"single-particle comparison: mu_ED={mu_ed:.10f}, mu_GW={gw.mu:.10f}, "
        f"rel||G_GW-G_ED||={Gerr_full:.6f}, low-|w| rel={Gerr_low:.6f}"
    )

    vertices = np.asarray([_cluster_vertex(exact, ch, q) for ch in channels], dtype=complex)
    chi_bubble_ed = bubble_iomega(G_ed, vertices, grid)[:, range(len(channels)), range(len(channels))]
    chi_bubble_gw = bubble_iomega(gw.G, vertices, grid)[:, range(len(channels)), range(len(channels))]
    full_iomega, vertex_iterations, vertex_residuals = _cgw_dynamic(
        gw, Vq, h0, vertices, grid, args
    )

    bubbleED = bosonic_iomega_to_tau(chi_bubble_ed, grid, tau)
    bubbleGW = bosonic_iomega_to_tau(chi_bubble_gw, grid, tau)
    fullC = bosonic_iomega_to_tau(full_iomega, grid, tau)

    mid = len(tau) // 2
    izero = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
    print("\n=== four-layer midpoint C(beta/2) ===")
    print("channel        exact ED        bubble[G_ED]    bubble[G_GW]    full cGW")
    for ic, ch in enumerate(channels):
        bg = (bubbleED[mid, ic] - bubbleGW[mid, ic]).real
        vex = (exactC[mid, ic] - bubbleED[mid, ic]).real
        vcg = (fullC[mid, ic] - bubbleGW[mid, ic]).real
        print(
            f"{ch:13s} {exactC[mid,ic].real:+.8e}  {bubbleED[mid,ic].real:+.8e}  "
            f"{bubbleGW[mid,ic].real:+.8e}  {fullC[mid,ic].real:+.8e}\n"
            f"              background={bg:+.4e}, exact_vertex={vex:+.4e}, cGW_vertex={vcg:+.4e}"
        )

    print("\n=== four-layer static chi(iOmega=0) ===")
    print("channel        exact ED        bubble[G_ED]    bubble[G_GW]    full cGW")
    for ic, ch in enumerate(channels):
        print(
            f"{ch:13s} {chi_exact[ic].real:+.8e}  {chi_bubble_ed[izero,ic].real:+.8e}  "
            f"{chi_bubble_gw[izero,ic].real:+.8e}  {full_iomega[izero,ic].real:+.8e}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    np.savez_compressed(
        out,
        L1=int(args.L1),
        L2=int(args.L2),
        V=float(args.V),
        filling=float(args.filling),
        target_N=float(target_N),
        T=float(args.T),
        beta=float(beta),
        q=np.asarray(q, dtype=float),
        qlabel=np.asarray(qlabel),
        channels=np.asarray(channels),
        tau=tau,
        omega=np.asarray(grid.omega),
        Omega=np.asarray(grid.Omega),
        m_values=np.asarray(grid.m_values),
        mu_ed=float(mu_ed),
        mu_gw=float(gw.mu),
        exact_sector_N=np.asarray(sorted(weights.sector_probabilities), dtype=int),
        exact_sector_p=np.asarray([weights.sector_probabilities[n] for n in sorted(weights.sector_probabilities)]),
        exact_mean_N=float(weights.mean_number),
        exact_var_N=float(weights.var_number),
        discarded_thermal_weight=float(weights.discarded_weight),
        G_ed=G_ed,
        G_gw=gw.G,
        G_rel_error=float(Gerr_full),
        G_lowfreq_rel_error=float(Gerr_low),
        exact_C_tau=exactC,
        bubble_ed_iomega=chi_bubble_ed,
        bubble_gw_iomega=chi_bubble_gw,
        full_cgw_iomega=full_iomega,
        bubble_ed_C_tau=bubbleED,
        bubble_gw_C_tau=bubbleGW,
        full_cgw_C_tau=fullC,
        exact_chi0=chi_exact,
        exact_means=means,
        vertex_iterations=vertex_iterations,
        vertex_residuals=vertex_residuals,
        gw_final_error=float(gw.final_error),
        gw_iterations=int(gw.iterations),
    )
    _plot(
        tau, beta, channels, exactC, bubbleED, bubbleGW, fullC, png, args.dpi, qlabel
    )
    print(f"saved: {out}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
