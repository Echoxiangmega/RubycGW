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
    "x_even": r"$x_{\rm even}$",
    "x_odd": r"$x_{\rm odd}$",
    "y_even": r"$y_{\rm even}$",
    "y_odd": r"$y_{\rm odd}$",
    "z_same": r"$z_{\rm same}$",
    "z_opposite": r"$z_{\rm opposite}$",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=3.0, help="Particles per six-site primitive cell.")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument(
        "--channels",
        nargs="+",
        default=["x_even", "z_same", "z_opposite"],
    )
    p.add_argument("--q", choices=["gamma", "m1"], default="gamma")
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--tau-points", type=int, default=101)
    p.add_argument("--thermal-discard", type=float, default=1e-12)
    p.add_argument("--ed-mu-tol", type=float, default=1e-12)

    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=1e-9)
    p.add_argument("--gw-mu-tol", type=float, default=5e-12)
    p.add_argument("--gw-mu-max-iter", type=int, default=80)
    p.add_argument("--gw-verbose", action="store_true")
    p.add_argument(
        "--gw-ramp",
        nargs="*",
        type=float,
        default=None,
        help="Optional positive continuation V values. Default uses a conservative ramp up to target V.",
    )

    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-workers", type=int, default=1, help="Reserved for future parallelization; current driver is serial for deterministic memory use.")
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--out", default="ed12_four_layer.npz")
    p.add_argument("--dpi", type=int, default=190)
    return p.parse_args()


def _ramp_values(target: float, raw) -> list[float]:
    target = float(target)
    if target < 0.0:
        raise ValueError("V must be non-negative")
    if raw is None:
        base = [0.10, 0.25, 0.50, 0.75]
    else:
        base = [float(x) for x in raw]
    vals = [x for x in base if 0.0 < x < target - 1e-12]
    if target > 0.0:
        vals.append(target)
    else:
        vals = [0.0]
    return sorted(set(vals))


def _q_value(args):
    if args.q == "gamma":
        return np.asarray([0.0, 0.0]), "Gamma"
    if args.L1 % 2 != 0:
        raise ValueError("M1=(1/2,0) requires even L1")
    return np.asarray([0.5, 0.0]), "M1=(1/2,0)"


def _solve_gw(exact, params, grid, target_N, args):
    h0 = exact.h0[None, None, :, :]
    Vunit = exact.Vunit[None, None, :, :]
    initial = None
    last = None
    for Vramp in _ramp_values(params.V, args.gw_ramp):
        print(f"SC-GW continuation V={Vramp:g}")
        opts = GWOptions(
            mu=0.0 if initial is None else float(initial.mu),
            target_filling=float(target_N),
            max_iter=int(args.gw_max_iter),
            tol=float(args.gw_tol),
            mu_tol=float(args.gw_mu_tol),
            mu_max_iter=int(args.gw_mu_max_iter),
            verbose=bool(args.gw_verbose),
            momentum_backend=str(args.momentum_backend),
        )
        last = solve_matrix_gw_anderson(
            h0,
            float(Vramp) * Vunit,
            grid,
            opts=opts,
            initial=initial,
            anderson=AndersonOptions(),
        )
        print(
            f"  converged={last.converged}, it={last.iterations}, "
            f"residual={last.final_error:.3e}, mu={last.mu:.10f}, "
            f"N={np.sum(last.density):.10f}"
        )
        if not last.converged:
            raise RuntimeError(
                f"12-site SC-GW failed at V={Vramp:g}: residual={last.final_error:.3e}"
            )
        initial = last
    return last, h0, float(params.V) * Vunit


def _dynamic_full_diag(gw, Vq, h0, vertices, grid, args):
    ref = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)
    opts = DynamicVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        solver="gmres",
        gmres_restart=int(args.vertex_gmres_restart),
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=bool(args.vertex_verbose),
        momentum_backend=str(args.momentum_backend),
    )
    nc = len(vertices)
    pos = np.zeros((grid.nOmega + 1, nc), dtype=complex)
    iterations = np.zeros_like(pos.real, dtype=int)
    residuals = np.zeros_like(pos.real)
    for ic, K in enumerate(vertices):
        previous = None
        for m in range(grid.nOmega + 1):
            print(f"full cGW channel {ic+1}/{nc}, m={m}/{grid.nOmega}")
            res = solve_vertex_iomega_tail(
                gw.G,
                gw.W,
                Vq,
                K,
                m,
                grid,
                ref,
                opts=opts,
                initial_gamma=previous,
            )
            if not res.converged:
                raise RuntimeError(
                    f"dynamic cGW failed channel={ic}, m={m}: residual={res.final_error:.3e}"
                )
            pos[m, ic] = susceptibility_matrix_iomega(
                gw.G,
                K[None, :, :],
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
        label = _CHANNEL_TEX.get(str(ch), str(ch))
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
    beta = 1.0 / float(args.T)
    tau = np.linspace(0.0, beta, int(args.tau_points))
    grid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=int(args.nw),
        nOmega=int(args.nomega),
        T=float(args.T),
    )

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
    exact.diagonalize(args.V)
    print(f"full sector ED diagonalization: {time.perf_counter()-t0:.2f} s")
    mu_ed = exact.solve_mu(target_N, args.T, tol=args.ed_mu_tol)
    selection = exact.thermal_selection(
        mu_ed,
        args.T,
        discard_weight_tol=args.thermal_discard,
    )
    print(
        f"exact GC: mu={mu_ed:.10f}, <N>={selection.average_particles:.10f}, "
        f"Var(N)={selection.variance_particles:.6e}, discarded thermal weight="
        f"{selection.discarded_weight:.3e}"
    )
    print("exact particle-sector weights:")
    for N, pN in enumerate(selection.sector_probabilities):
        if pN > 1e-8:
            print(f"  N={N:2d}: p={pN:.8e}")

    # Exact one-particle Green function.
    t0 = time.perf_counter()
    Ged, selection_G = exact.green_iomega(
        1j * np.asarray(grid.omega),
        mu_ed,
        args.T,
        discard_weight_tol=args.thermal_discard,
    )
    print(
        f"exact Lehmann G(iw): {time.perf_counter()-t0:.2f} s, "
        f"discarded weight={selection_G.discarded_weight:.3e}"
    )
    Ged5 = Ged[:, None, None, :, :]

    # Exact two-body C(tau) and static susceptibility.
    vertices = np.stack([exact.pseudospin_operator(ch, q=q) for ch in channels])
    exact_tau = np.zeros((len(tau), len(channels)), dtype=complex)
    exact_static = np.zeros(len(channels), dtype=float)
    exact_means = np.zeros(len(channels), dtype=complex)
    for ic, (ch, K) in enumerate(zip(channels, vertices)):
        t0 = time.perf_counter()
        C, chi, mean, _ = exact.correlation_tau(
            K,
            tau,
            mu_ed,
            args.T,
            discard_weight_tol=args.thermal_discard,
        )
        exact_tau[:, ic] = C
        exact_static[ic] = chi
        exact_means[ic] = mean
        print(
            f"exact C {ch:12s}: chi0={chi:+.8e}, <O>={mean.real:+.3e}{mean.imag:+.3e}j, "
            f"time={time.perf_counter()-t0:.2f}s"
        )

    # Same 12-site SC-GW background at the same average total N.
    gw, h0gw, Vq = _solve_gw(exact, params, grid, target_N, args)
    Ggw = np.asarray(gw.G, dtype=complex)
    relG = float(np.linalg.norm((Ggw - Ged5).ravel()) / max(np.linalg.norm(Ged5.ravel()), 1e-300))
    near = np.argsort(np.abs(grid.omega))[:4]
    relG_low = float(
        np.linalg.norm((Ggw[near] - Ged5[near]).ravel())
        / max(np.linalg.norm(Ged5[near].ravel()), 1e-300)
    )
    print(
        f"single-particle comparison: mu_ED={mu_ed:.10f}, mu_GW={gw.mu:.10f}, "
        f"rel||G_GW-G_ED||={relG:.6f}, low-|w| rel={relG_low:.6f}"
    )

    # Two bubbles on their respective one-particle backgrounds.
    bubble_ed_iw = bubble_iomega(Ged5, vertices, grid)
    bubble_gw_iw = bubble_iomega(Ggw, vertices, grid)
    bubble_ed_diag_iw = np.diagonal(bubble_ed_iw, axis1=-2, axis2=-1)
    bubble_gw_diag_iw = np.diagonal(bubble_gw_iw, axis1=-2, axis2=-1)
    bubble_ed_tau = bosonic_iomega_to_tau(bubble_ed_diag_iw, grid, tau)
    bubble_gw_tau = bosonic_iomega_to_tau(bubble_gw_diag_iw, grid, tau)

    # Full dynamic tail-consistent cGW on G_GW.
    full_iw, vertex_iterations, vertex_residuals = _dynamic_full_diag(
        gw, Vq, h0gw, vertices, grid, args
    )
    full_tau = bosonic_iomega_to_tau(full_iw, grid, tau)

    izero = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
    mid = int(np.argmin(np.abs(tau - beta / 2.0)))
    print("\n=== four-layer midpoint C(beta/2) ===")
    print("channel        exact ED        bubble[G_ED]    bubble[G_GW]    full cGW")
    for ic, ch in enumerate(channels):
        print(
            f"{ch:12s}  {exact_tau[mid,ic].real:+.8e}  "
            f"{bubble_ed_tau[mid,ic].real:+.8e}  "
            f"{bubble_gw_tau[mid,ic].real:+.8e}  "
            f"{full_tau[mid,ic].real:+.8e}"
        )
        print(
            " " * 14
            + f"background={float((bubble_ed_tau[mid,ic]-bubble_gw_tau[mid,ic]).real):+.4e}, "
            + f"exact_vertex={float((exact_tau[mid,ic]-bubble_ed_tau[mid,ic]).real):+.4e}, "
            + f"cGW_vertex={float((full_tau[mid,ic]-bubble_gw_tau[mid,ic]).real):+.4e}"
        )

    print("\n=== four-layer static chi(iOmega=0) ===")
    print("channel        exact ED        bubble[G_ED]    bubble[G_GW]    full cGW")
    for ic, ch in enumerate(channels):
        print(
            f"{ch:12s}  {exact_static[ic]:+.8e}  "
            f"{bubble_ed_diag_iw[izero,ic].real:+.8e}  "
            f"{bubble_gw_diag_iw[izero,ic].real:+.8e}  "
            f"{full_iw[izero,ic].real:+.8e}"
        )

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        V=float(args.V),
        filling=float(args.filling),
        temperature=float(args.T),
        beta=float(beta),
        L1=int(args.L1),
        L2=int(args.L2),
        n_sites=int(exact.n_sites),
        target_particles=float(target_N),
        q=np.asarray(q),
        q_label=np.asarray(qlabel),
        channels=np.asarray(channels),
        tau_grid=tau,
        exact_mu=float(mu_ed),
        gw_mu=float(gw.mu),
        exact_average_particles=float(selection.average_particles),
        exact_particle_variance=float(selection.variance_particles),
        exact_sector_probabilities=np.asarray(selection.sector_probabilities),
        exact_discarded_weight=float(selection.discarded_weight),
        exact_G_iomega=Ged,
        gw_G_iomega=np.asarray(gw.G[:, 0, 0]),
        G_relative_error=float(relG),
        G_low_frequency_relative_error=float(relG_low),
        exact_correlation_tau=exact_tau,
        exact_static_susceptibility=exact_static,
        exact_operator_means=exact_means,
        bubble_exact_iomega=bubble_ed_diag_iw,
        bubble_gw_iomega=bubble_gw_diag_iw,
        bubble_exact_tau=bubble_ed_tau,
        bubble_gw_tau=bubble_gw_tau,
        full_cgw_iomega=full_iw,
        full_cgw_tau=full_tau,
        cgw_vertex_iterations=vertex_iterations,
        cgw_vertex_residuals=vertex_residuals,
        m_values=np.asarray(grid.m_values),
        Omega=np.asarray(grid.Omega),
        omega=np.asarray(grid.omega),
        thermal_discard_tolerance=float(args.thermal_discard),
        note=np.asarray(
            "grand-canonical exact ED and SC-GW independently matched to the same average total particle number"
        ),
    )
    plot = out.with_name(out.stem + ".png")
    _plot(
        tau,
        beta,
        channels,
        exact_tau,
        bubble_ed_tau,
        bubble_gw_tau,
        full_tau,
        plot,
        args.dpi,
        qlabel,
    )
    print("saved:", out)
    print("saved:", plot)


if __name__ == "__main__":
    main()
