#!/usr/bin/env python3
"""Exact finite-temperature ED scan of q=0 pseudospin susceptibilities versus V.

This script intentionally matches the ED definition used by
``benchmark_ed_gw_cgw_sox.py``:

* ``ExactSmallRubyThermal`` on the same finite periodic Ruby torus;
* grand-canonical finite-temperature susceptibility;
* fixed average filling set through the exact chemical potential;
* q=0 pseudospin operators.

In addition to chi(V), the scan records basis-independent low-energy diagnostics
useful for distinguishing ordinary finite-size fluctuations from an incipient
spontaneous ordered state:

* equal-time connected structure factors S_mumu = <O_mu^2>-<O_mu>^2;
* the lowest many-body excitation energies in the fixed-N sector;
* the exact ground-state multiplicity and gap above that manifold;
* a user-controlled quasi-degenerate low-energy manifold;
* projected operator spectra of P O_mu P inside the ground/low-energy manifold;
* the basis-independent coupling between the exact ground manifold and the
  rest of the quasi-degenerate low-energy manifold.

For a loop-current doublet, the projected current operator should develop
approximately opposite eigenvalues (+J,-J) inside the low-energy manifold.
This remains meaningful when a numerical eigensolver rotates the basis inside
an exactly or nearly degenerate subspace.

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
        "--ground-degeneracy-tol",
        type=float,
        default=1e-10,
        help=(
            "absolute energy tolerance used only to identify an exactly degenerate "
            "ground-state manifold"
        ),
    )
    p.add_argument(
        "--low-energy-window",
        type=float,
        default=1e-3,
        help=(
            "absolute E-E0 window defining a quasi-degenerate low-energy manifold; "
            "this is a finite-size diagnostic, not an exact degeneracy criterion"
        ),
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
            diag = np.sum(U.conj() * Opsi, axis=0)
            second_state = np.sum(np.abs(Opsi) ** 2, axis=0)
            means[ic] += np.dot(p, diag)
            second[ic] += float(np.dot(p, second_state).real)

    connected = second - np.abs(means) ** 2
    connected = np.maximum(np.real(connected), 0.0)
    return connected, means


def _projected_operator_stats(exact, operators, N, indices, store_eigs):
    """Basis-independent statistics of P O P for one selected eigen-subspace."""
    sec = exact.sectors[int(N)]
    idx = np.asarray(indices, dtype=int)
    nc = len(operators)
    eigvals = np.full((nc, int(store_eigs)), np.nan, dtype=float)
    rms = np.full(nc, np.nan, dtype=float)
    span = np.full(nc, np.nan, dtype=float)
    if idx.size == 0:
        return eigvals, rms, span

    Usub = np.asarray(sec.eigenvectors[:, idx], dtype=complex)
    d = int(idx.size)
    for ic, K in enumerate(operators):
        Osub = Usub.conj().T @ exact._apply_onebody_batch(int(N), K, Usub)
        Osub = 0.5 * (Osub + Osub.conj().T)
        vals = np.linalg.eigvalsh(Osub).real
        nstore = min(len(vals), int(store_eigs))
        eigvals[ic, :nstore] = vals[:nstore]
        mean = float(np.trace(Osub).real / d)
        centered = Osub - mean * np.eye(d, dtype=complex)
        rms[ic] = float(
            np.sqrt(max(0.0, np.trace(centered.conj().T @ centered).real / d))
        )
        span[ic] = float(vals[-1] - vals[0])
    return eigvals, rms, span


def _fixed_n_manifold_diagnostics(
    exact,
    operators,
    N,
    n_levels,
    ground_degeneracy_tol,
    low_energy_window,
):
    """Many-body gaps plus basis-independent ground/low-manifold diagnostics."""
    sec = exact.sectors[int(N)]
    energies = np.asarray(sec.energies, dtype=float)
    nlev = min(int(n_levels), len(energies))
    gaps = np.full(int(n_levels), np.nan, dtype=float)
    if nlev == 0:
        raise RuntimeError("empty fixed-N sector")
    full_gaps = energies - energies[0]
    gaps[:nlev] = full_gaps[:nlev]

    deg_tol = float(ground_degeneracy_tol)
    low_window = float(low_energy_window)
    if deg_tol < 0.0:
        raise ValueError("--ground-degeneracy-tol must be non-negative")
    if low_window < 0.0:
        raise ValueError("--low-energy-window must be non-negative")
    low_window = max(low_window, deg_tol)

    ground_dim = int(np.count_nonzero(full_gaps <= deg_tol))
    low_dim = int(np.count_nonzero(full_gaps <= low_window))
    ground_dim = max(1, ground_dim)
    low_dim = max(ground_dim, low_dim)

    gap_above_ground = (
        float(full_gaps[ground_dim]) if ground_dim < len(full_gaps) else np.nan
    )
    gap_above_low = float(full_gaps[low_dim]) if low_dim < len(full_gaps) else np.nan

    ground_eigs, ground_rms, ground_span = _projected_operator_stats(
        exact,
        operators,
        int(N),
        np.arange(ground_dim, dtype=int),
        int(n_levels),
    )
    low_eigs, low_rms, low_span = _projected_operator_stats(
        exact,
        operators,
        int(N),
        np.arange(low_dim, dtype=int),
        int(n_levels),
    )

    # Basis-independent generalization of |<0|O|1>| for a quasi-degenerate
    # partner manifold:
    #
    #   A^2 = (1/d_gs) || P_(low\gs) O P_gs ||_F^2 .
    #
    # If the exact GS is nondegenerate and the low manifold has two states,
    # this reduces exactly to |<1|O|0>|^2.
    gs_to_low_amplitude = np.zeros(len(operators), dtype=float)
    if low_dim > ground_dim:
        Ugs = np.asarray(sec.eigenvectors[:, :ground_dim], dtype=complex)
        Upartner = np.asarray(
            sec.eigenvectors[:, ground_dim:low_dim], dtype=complex
        )
        for ic, K in enumerate(operators):
            Ogs = exact._apply_onebody_batch(int(N), K, Ugs)
            block = Upartner.conj().T @ Ogs
            weight = float(np.sum(np.abs(block) ** 2) / ground_dim)
            gs_to_low_amplitude[ic] = np.sqrt(max(0.0, weight))

    return {
        "energy_gaps": gaps,
        "ground_dim": ground_dim,
        "low_dim": low_dim,
        "gap_above_ground": gap_above_ground,
        "gap_above_low": gap_above_low,
        "ground_projected_eigenvalues": ground_eigs,
        "low_projected_eigenvalues": low_eigs,
        "ground_order_rms": ground_rms,
        "low_order_rms": low_rms,
        "ground_order_span": ground_span,
        "low_order_span": low_span,
        "ground_to_low_amplitude": gs_to_low_amplitude,
    }


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
    ground_dim,
    low_dim,
    low_order_rms,
    ground_to_low_amplitude,
    T,
    spectral_N,
    low_energy_window,
    path,
    dpi,
):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.5), sharex=True)

    ax = axes[0, 0]
    for ic, ch in enumerate(channels):
        ax.plot(V, chi_diag[:, ic], marker="o", ms=3, label=_CHANNEL_TEX.get(ch, ch))
    ax.set_ylabel(r"$\chi_{\mu\mu}$")
    ax.set_title("Static susceptibility")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for ic, ch in enumerate(channels):
        ax.plot(
            V,
            structure_diag[:, ic],
            marker="o",
            ms=3,
            label=_CHANNEL_TEX.get(ch, ch),
        )
    ax.set_ylabel(r"$S_{\mu\mu}=\langle O_\mu^2\rangle_c$")
    ax.set_title("Equal-time connected fluctuations")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if energy_gaps.shape[1] > 1:
        for n in range(1, energy_gaps.shape[1]):
            if np.any(np.isfinite(energy_gaps[:, n])):
                ax.plot(
                    V,
                    energy_gaps[:, n],
                    marker="o",
                    ms=3,
                    label=fr"$E_{n}-E_0$",
                )
    ax.set_ylabel("many-body gap")
    ax.set_title(fr"Fixed-$N$ spectrum, $N={spectral_N}$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.step(V, ground_dim, where="mid", label="exact GS dimension")
    ax.step(
        V,
        low_dim,
        where="mid",
        label=rf"$E-E_0\leq {low_energy_window:g}$",
    )
    ax.set_ylabel("subspace dimension")
    ax.set_title("Ground / quasi-degenerate manifold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for ic, ch in enumerate(channels):
        ax.plot(
            V,
            low_order_rms[:, ic],
            marker="o",
            ms=3,
            label=_CHANNEL_TEX.get(ch, ch),
        )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"RMS eigenspread of $P_{\rm low}O_\mu P_{\rm low}$")
    ax.set_title("Order amplitude inside low-energy manifold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    for ic, ch in enumerate(channels):
        ax.plot(
            V,
            ground_to_low_amplitude[:, ic],
            marker="o",
            ms=3,
            label=_CHANNEL_TEX.get(ch, ch),
        )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\sqrt{\|P_{\rm p}O_\mu P_{\rm GS}\|_F^2/d_{\rm GS}}$")
    ax.set_title("GS-to-quasi-degenerate-partner coupling")
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
    if args.ground_degeneracy_tol < 0.0:
        raise ValueError("--ground-degeneracy-tol must be non-negative")
    if args.low_energy_window < 0.0:
        raise ValueError("--low-energy-window must be non-negative")
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
    operators = np.stack(
        [exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in channels]
    )

    nV = len(Vvalues)
    nc = len(channels)
    chi = np.full((nV, nc, nc), np.nan, dtype=float)
    chi_diag = np.full((nV, nc), np.nan, dtype=float)
    structure_diag = np.full((nV, nc), np.nan, dtype=float)
    thermal_means = np.full((nV, nc), np.nan + 0j, dtype=complex)
    energy_gaps = np.full((nV, args.n_levels), np.nan, dtype=float)
    ground_dim = np.zeros(nV, dtype=int)
    low_dim = np.zeros(nV, dtype=int)
    gap_above_ground = np.full(nV, np.nan, dtype=float)
    gap_above_low = np.full(nV, np.nan, dtype=float)
    ground_projected_eigenvalues = np.full(
        (nV, nc, args.n_levels), np.nan, dtype=float
    )
    low_projected_eigenvalues = np.full(
        (nV, nc, args.n_levels), np.nan, dtype=float
    )
    ground_order_rms = np.full((nV, nc), np.nan, dtype=float)
    low_order_rms = np.full((nV, nc), np.nan, dtype=float)
    ground_order_span = np.full((nV, nc), np.nan, dtype=float)
    low_order_span = np.full((nV, nc), np.nan, dtype=float)
    ground_to_low_amplitude = np.full((nV, nc), np.nan, dtype=float)
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
    print(
        f"exact GS tolerance={args.ground_degeneracy_tol:g}, "
        f"quasi-degenerate window={args.low_energy_window:g}"
    )
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

        spec = _fixed_n_manifold_diagnostics(
            exact,
            operators,
            spectral_N,
            args.n_levels,
            args.ground_degeneracy_tol,
            args.low_energy_window,
        )
        energy_gaps[iv] = spec["energy_gaps"]
        ground_dim[iv] = spec["ground_dim"]
        low_dim[iv] = spec["low_dim"]
        gap_above_ground[iv] = spec["gap_above_ground"]
        gap_above_low[iv] = spec["gap_above_low"]
        ground_projected_eigenvalues[iv] = spec["ground_projected_eigenvalues"]
        low_projected_eigenvalues[iv] = spec["low_projected_eigenvalues"]
        ground_order_rms[iv] = spec["ground_order_rms"]
        low_order_rms[iv] = spec["low_order_rms"]
        ground_order_span[iv] = spec["ground_order_span"]
        low_order_span[iv] = spec["low_order_span"]
        ground_to_low_amplitude[iv] = spec["ground_to_low_amplitude"]

        chi_text = "  ".join(
            f"{ch}={chi_diag[iv, ic]:.8f}" for ic, ch in enumerate(channels)
        )
        s_text = "  ".join(
            f"S_{ch}={structure_diag[iv, ic]:.8f}"
            for ic, ch in enumerate(channels)
        )
        print(f"V={V:8.5f}  mu={mu[iv]:+.10f}  {chi_text}")
        print(
            f"            gap01={energy_gaps[iv, 1]:.8e}  "
            f"GSdeg={ground_dim[iv]}  gap>GS={gap_above_ground[iv]:.8e}  "
            f"lowdim={low_dim[iv]}  gap>low={gap_above_low[iv]:.8e}"
        )
        print(f"            {s_text}")
        for ic, ch in enumerate(channels):
            vals = low_projected_eigenvalues[iv, ic, : low_dim[iv]]
            vals_text = ",".join(f"{x:+.4e}" for x in vals[np.isfinite(vals)])
            print(
                f"            {ch:12s} low_rms={low_order_rms[iv, ic]:.6e}  "
                f"GS->low={ground_to_low_amplitude[iv, ic]:.6e}  "
                f"eig(P_low O P_low)=[{vals_text}]"
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
        ground_degeneracy_tolerance=float(args.ground_degeneracy_tol),
        low_energy_window=float(args.low_energy_window),
        ground_multiplicity=ground_dim,
        low_energy_multiplicity=low_dim,
        gap_above_ground_manifold=gap_above_ground,
        gap_above_low_energy_manifold=gap_above_low,
        ground_projected_operator_eigenvalues=ground_projected_eigenvalues,
        low_projected_operator_eigenvalues=low_projected_eigenvalues,
        ground_order_rms=ground_order_rms,
        low_order_rms=low_order_rms,
        ground_order_span=ground_order_span,
        low_order_span=low_order_span,
        ground_to_low_amplitude=ground_to_low_amplitude,
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
        ground_dim,
        low_dim,
        low_order_rms,
        ground_to_low_amplitude,
        args.T,
        spectral_N,
        args.low_energy_window,
        diagnostics_plot_path,
        args.dpi,
    )

    print("saved:", npz_path)
    print("saved:", chi_plot_path)
    print("saved:", diagnostics_plot_path)


if __name__ == "__main__":
    main()
