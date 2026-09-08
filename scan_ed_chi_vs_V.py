#!/usr/bin/env python3
"""Exact finite-temperature ED scan of q=0 pseudospin susceptibilities versus V.

This script intentionally matches the ED definition used by
``benchmark_ed_gw_cgw_sox.py``:

* ``ExactSmallRubyThermal`` on the same finite periodic Ruby torus;
* grand-canonical finite-temperature susceptibility;
* fixed average filling set through the exact chemical potential;
* q=0 pseudospin operators.

In addition to chi(V), the scan records diagnostics useful for distinguishing
large finite-size fluctuations from an incipient spontaneous loop-current state:

* equal-time connected structure factors S_mumu = <O_mu^2>-<O_mu>^2;
* the lowest many-body excitation energies in the fixed-N sector corresponding
  to the requested filling;
* |<0|O_mu|n>| for the same low-lying fixed-N states.

The default 2x1 torus has 12 sites, so all particle-number sectors together
contain only 2**12 = 4096 states and can be fully diagonalized exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


_DEFAULT_CHANNELS = ("x_even", "z_same", "z_opposite")
_CHANNEL_TEX = {
    "x_even": r"$x_{\rm even}$",
    "z_same": r"$z_{\rm same}$",
    "z_opposite": r"$z_{\rm opposite}$",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--Vmin", type=float, default=0.0)
    p.add_argument("--Vmax", type=float, default=2.0)
    p.add_argument("--dV", type=float, default=0.05)
    p.add_argument(
        "--V",
        nargs="*",
        type=float,
        default=None,
        help="explicit V values; overrides --Vmin/--Vmax/--dV",
    )
    p.add_argument("--channels", nargs="+", default=list(_DEFAULT_CHANNELS))
    p.add_argument("--filling", type=float, default=3.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument(
        "--n-levels",
        type=int,
        default=6,
        help="number of fixed-N many-body levels saved for spectral diagnostics",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/ed_chi_vs_V"),
        help="output directory",
    )
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _v_grid(args) -> np.ndarray:
    if args.V:
        return np.asarray(sorted(set(float(x) for x in args.V)), dtype=float)
    if args.dV <= 0.0:
        raise ValueError("--dV must be positive")
    if args.Vmax < args.Vmin:
        raise ValueError("--Vmax must be >= --Vmin")
    n = int(np.floor((args.Vmax - args.Vmin) / args.dV + 1e-12))
    vals = args.Vmin + args.dV * np.arange(n + 1, dtype=float)
    if vals.size == 0 or vals[-1] < args.Vmax - 1e-12:
        vals = np.append(vals, float(args.Vmax))
    return vals


def _thermal_structure_diag(exact, operators, mu, T):
    """Exact grand-canonical connected equal-time <O^2>-<O>^2."""
    probs, _, _ = exact._normalized_probabilities(mu, T)
    nc = len(operators)
    means = np.zeros(nc, dtype=complex)
    second = np.zeros(nc, dtype=float)

    for N, (sec, p) in enumerate(zip(exact.sectors, probs)):
        if len(sec.energies) == 0 or not np.any(p > 0.0):
            continue
        U = np.asarray(sec.eigenvectors, dtype=complex)
        for ic, K in enumerate(operators):
            Opsi = exact._apply_onebody_batch(N, K, U)
            M = U.conj().T @ Opsi
            diag = np.diag(M)
            means[ic] += np.dot(p, diag)
            # For eigenstate |n>, <n|O^2|n> = sum_m |<m|O|n>|^2.
            second[ic] += float(np.dot(p, np.sum(np.abs(M) ** 2, axis=0)).real)

    connected = second - np.abs(means) ** 2
    connected = np.maximum(np.real(connected), 0.0)
    return connected, means


def _fixed_n_spectral_diagnostics(exact, operators, N, n_levels):
    """Low-energy gaps and |<0|O|n>| in one fixed-particle-number sector."""
    sec = exact.sectors[int(N)]
    nlev = min(int(n_levels), len(sec.energies))
    gaps = np.full(int(n_levels), np.nan, dtype=float)
    matrix_elements = np.full((len(operators), int(n_levels)), np.nan, dtype=float)
    if nlev == 0:
        return gaps, matrix_elements

    gaps[:nlev] = np.asarray(sec.energies[:nlev] - sec.energies[0], dtype=float)
    U = np.asarray(sec.eigenvectors, dtype=complex)
    psi0 = U[:, 0]
    for ic, K in enumerate(operators):
        Opsi0 = exact._apply_onebody_batch(int(N), K, psi0)[:, 0]
        overlaps = U[:, :nlev].conj().T @ Opsi0
        matrix_elements[ic, :nlev] = np.abs(overlaps)
    return gaps, matrix_elements


def _plot_chi(V, channels, chi_diag, T, path, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for ic, ch in enumerate(channels):
        ax.plot(V, chi_diag[:, ic], marker="o", ms=3.2, label=_CHANNEL_TEX.get(ch, ch))
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"static susceptibility $\chi$")
    ax.set_title(rf"Exact ED, $q=0$, $T={T:g}$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_diagnostics(
    V,
    channels,
    chi_diag,
    structure_diag,
    energy_gaps,
    matrix_elements,
    T,
    spectral_N,
    path,
    dpi,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True)
    ax = axes[0, 0]
    for ic, ch in enumerate(channels):
        ax.plot(V, chi_diag[:, ic], marker="o", ms=3, label=_CHANNEL_TEX.get(ch, ch))
    ax.set_ylabel(r"$\chi_{\mu\mu}$")
    ax.set_title(r"Static susceptibility")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for ic, ch in enumerate(channels):
        ax.plot(V, structure_diag[:, ic], marker="o", ms=3, label=_CHANNEL_TEX.get(ch, ch))
    ax.set_ylabel(r"$S_{\mu\mu}=\langle O_\mu^2\rangle_c$")
    ax.set_title(r"Equal-time connected fluctuations")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if energy_gaps.shape[1] > 1:
        for n in range(1, energy_gaps.shape[1]):
            if np.any(np.isfinite(energy_gaps[:, n])):
                ax.plot(V, energy_gaps[:, n], marker="o", ms=3, label=fr"$E_{n}-E_0$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel("many-body gap")
    ax.set_title(fr"Fixed-$N$ spectrum, $N={spectral_N}$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if matrix_elements.shape[2] > 1:
        for ic, ch in enumerate(channels):
            ax.plot(
                V,
                matrix_elements[:, ic, 1],
                marker="o",
                ms=3,
                label=_CHANNEL_TEX.get(ch, ch),
            )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$|\langle0|O_\mu|1\rangle|$")
    ax.set_title(r"Ground-to-first-excited matrix element")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(rf"Exact ED order diagnostics, $q=0$, $T={T:g}$", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    if args.T <= 0.0:
        raise ValueError("--T must be positive")
    if args.n_levels < 2:
        raise ValueError("--n-levels must be at least 2")
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal requires at most 16 sites")

    Vvalues = _v_grid(args)
    channels = [str(ch) for ch in args.channels]
    ncell = int(args.L1) * int(args.L2)
    target_particles = float(args.filling) * ncell
    spectral_N = int(round(target_particles))
    spectral_exact = abs(target_particles - spectral_N) <= 1e-10

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    operators = np.stack([exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in channels])

    nV = len(Vvalues)
    nc = len(channels)
    chi = np.full((nV, nc, nc), np.nan, dtype=float)
    chi_diag = np.full((nV, nc), np.nan, dtype=float)
    structure_diag = np.full((nV, nc), np.nan, dtype=float)
    thermal_means = np.full((nV, nc), np.nan + 0j, dtype=complex)
    energy_gaps = np.full((nV, args.n_levels), np.nan, dtype=float)
    matrix_elements = np.full((nV, nc, args.n_levels), np.nan, dtype=float)
    mu = np.full(nV, np.nan, dtype=float)

    print("=== exact finite-T ED q=0 susceptibility/order scan ===")
    print(
        f"cluster={args.L1}x{args.L2} primitive cells, sites={exact.n_sites}, "
        f"target particles={target_particles:g}"
    )
    print(
        f"T={args.T:g}, filling={args.filling:g}, "
        f"ti={args.ti:g}, t1={args.t1:g}, t2={args.t2:g}"
    )
    print("channels=" + ", ".join(channels))
    if spectral_exact:
        print(f"low-energy spectral diagnostics use fixed-N sector N={spectral_N}")
    else:
        print(
            "WARNING: filling*ncell is noninteger; low-energy fixed-N diagnostics "
            f"use the nearest sector N={spectral_N} while chi/S remain grand canonical"
        )

    for iv, V in enumerate(Vvalues):
        exact.diagonalize(float(V))
        mu[iv] = exact.solve_mu(target_particles, args.T)
        chi[iv], _ = exact.static_susceptibility_matrix(operators, mu[iv], args.T)
        chi_diag[iv] = np.real(np.diag(chi[iv]))
        structure_diag[iv], thermal_means[iv] = _thermal_structure_diag(
            exact, operators, mu[iv], args.T
        )
        energy_gaps[iv], matrix_elements[iv] = _fixed_n_spectral_diagnostics(
            exact, operators, spectral_N, args.n_levels
        )

        chi_text = "  ".join(
            f"{ch}={chi_diag[iv, ic]:.8f}" for ic, ch in enumerate(channels)
        )
        s_text = "  ".join(
            f"S_{ch}={structure_diag[iv, ic]:.8f}" for ic, ch in enumerate(channels)
        )
        m01_text = "  ".join(
            f"|0{ch}1|={matrix_elements[iv, ic, 1]:.6f}"
            for ic, ch in enumerate(channels)
        )
        print(f"V={V:8.5f}  mu={mu[iv]:+.10f}  {chi_text}")
        print(
            f"            gap01={energy_gaps[iv, 1]:.8e}  "
            f"{s_text}  {m01_text}"
        )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "ed_chi_scan.npz"
    chi_plot_path = outdir / "ed_chi_vs_V.png"
    diagnostics_plot_path = outdir / "ed_order_diagnostics_vs_V.png"

    np.savez_compressed(
        npz_path,
        V=Vvalues,
        channels=np.asarray(channels),
        chi=chi,
        chi_diag=chi_diag,
        structure_diag=structure_diag,
        thermal_means=thermal_means,
        mu=mu,
        spectral_sector_N=int(spectral_N),
        spectral_sector_matches_target=bool(spectral_exact),
        energy_gaps=energy_gaps,
        matrix_elements_from_ground=matrix_elements,
        L1=int(args.L1),
        L2=int(args.L2),
        n_sites=int(exact.n_sites),
        filling=float(args.filling),
        target_particles=float(target_particles),
        T=float(args.T),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        q=np.asarray([0.0, 0.0]),
        ensemble=np.asarray("grand_canonical_exact_fixed_average_filling"),
        spectral_ensemble=np.asarray("canonical_fixed_N"),
    )
    _plot_chi(Vvalues, channels, chi_diag, args.T, chi_plot_path, args.dpi)
    _plot_diagnostics(
        Vvalues,
        channels,
        chi_diag,
        structure_diag,
        energy_gaps,
        matrix_elements,
        args.T,
        spectral_N,
        diagnostics_plot_path,
        args.dpi,
    )

    print("saved:", npz_path)
    print("saved:", chi_plot_path)
    print("saved:", diagnostics_plot_path)


if __name__ == "__main__":
    main()
