#!/usr/bin/env python3
"""Finite-source current-order benchmark: exact ED vs SC-GW vs SC-GW+SOX.

This benchmark deliberately does *not* solve a cGW response vertex.  Instead it
compares the one-body order parameter itself on the same explicitly
symmetry-broken Hamiltonian,

    H(h) = H(0) - h K_source,

where ``K_source`` is one of the two q=0 loop-current pseudospins:

    same      -> z_same
    opposite  -> z_opposite.

For every (source,V) pair the source field is followed from large positive h
down to h=0.  The previous converged solution is used as the initial condition
for the next smaller h, so the final h=0 point tests whether a selected current
branch survives after the explicit source is removed.

All three methods use the same finite periodic Ruby torus, the same one-body
source operator and the same fixed average filling:

* ED: exact grand-canonical finite-T equilibrium on the small torus;
* GW: fully self-consistent GW at each h;
* GW+SOX: fully self-consistent GW plus the bare second-order exchange skeleton.

The measured current/order parameter is the expectation value of the same
normalized one-body pseudospin operator K.  For GW/GW+SOX it is obtained from
G with a static-reference Matsubara tail completion.  No cGW vertex is needed
because <K> is a one-body observable.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox import solve_matrix_gw_sox
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


_SOURCE_TO_CHANNEL = {
    "same": "z_same",
    "opposite": "z_opposite",
}
_SOURCE_TEX = {
    "same": r"same current ($z_{\rm same}$)",
    "opposite": r"opposite current ($z_{\rm opposite}$)",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument(
        "--V",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="interaction strengths; each V gets an independent h continuation",
    )
    p.add_argument(
        "--source",
        nargs="+",
        choices=sorted(_SOURCE_TO_CHANNEL),
        default=["same", "opposite"],
        help="current source mode(s)",
    )
    p.add_argument(
        "--h",
        nargs="+",
        type=float,
        default=[0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
        help=(
            "non-negative source strengths; internally sorted from large to small "
            "to follow the selected branch toward h=0"
        ),
    )
    p.add_argument("--filling", type=float, default=3.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max-iter", type=int, default=180)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.22)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--sox-nquad", type=int, default=128)
    p.add_argument("--backend", choices=["fft", "direct"], default="direct")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/finite_source_current_benchmark"),
    )
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _require(ok: bool, message: str, allow: bool):
    if ok:
        return
    if allow:
        print("WARNING:", message)
        return
    raise RuntimeError(message)


def _unique_descending_nonnegative(values) -> np.ndarray:
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size == 0:
        raise ValueError("need at least one --h value")
    if np.any(~np.isfinite(arr)):
        raise ValueError("all --h values must be finite")
    if np.any(arr < -1e-15):
        raise ValueError("this branch-selection benchmark expects non-negative --h values")
    arr[np.abs(arr) < 1e-15] = 0.0
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _fermi(e_minus_mu: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(e_minus_mu, dtype=float) / float(T)
    out = np.empty_like(x)
    high = x > 40.0
    low = x < -40.0
    mid = ~(high | low)
    out[high] = 0.0
    out[low] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def _exact_onebody_expectation(exact, K, mu, T) -> complex:
    """Exact grand-canonical <K> for a one-body Hermitian operator."""
    probs, _, _ = exact._normalized_probabilities(float(mu), float(T))
    value = 0.0j
    for N, (sec, p) in enumerate(zip(exact.sectors, probs)):
        if len(sec.energies) == 0 or not np.any(p > 0.0):
            continue
        U = np.asarray(sec.eigenvectors, dtype=complex)
        Opsi = exact._apply_onebody_batch(int(N), K, U)
        diag = np.sum(U.conj() * Opsi, axis=0)
        value += np.dot(p, diag)
    return complex(value)


def _bilinear_expectation_tail_completed(G, K, grid, h_static, mu) -> complex:
    """Return <K> with a static-reference Matsubara tail completion.

    We write

        <K> = <K>_ref + T/Nk sum Tr[K (G-G_ref)],

    with G_ref=(iw+mu-h_static)^(-1).  The reference contribution is evaluated
    analytically from the Fermi function, while G-G_ref has a faster-decaying
    high-frequency tail.  ``h_static`` should contain h0 + Hartree + static Fock.
    """
    Garr = np.asarray(G, dtype=complex)
    Karr = np.asarray(K, dtype=complex)
    href = np.asarray(h_static, dtype=complex)
    if Garr.ndim != 5:
        raise ValueError("G must have shape (nf,nk1,nk2,norb,norb)")
    if href.shape != Garr.shape[1:3] + Garr.shape[-2:]:
        raise ValueError("h_static shape mismatch")
    if Karr.shape != Garr.shape[-2:]:
        raise ValueError("K shape mismatch")

    href = 0.5 * (href + np.swapaxes(href.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(href)
    occ = _fermi(evals - float(mu), float(grid.T))
    rho_ref = np.einsum(
        "xyai,xyi,xybi->xyab",
        evecs,
        occ,
        evecs.conj(),
        optimize=True,
    )
    ref_value = (1.0 / float(grid.nk)) * np.einsum(
        "ab,xyba->", Karr, rho_ref, optimize=True
    )

    denom = (
        1j * np.asarray(grid.omega)[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    Gref = np.einsum(
        "xyai,nxyi,xybi->nxyab",
        evecs,
        1.0 / denom,
        evecs.conj(),
        optimize=True,
    )
    correction = (float(grid.T) / float(grid.nk)) * np.einsum(
        "ab,nxyba->", Karr, Garr - Gref, optimize=True
    )
    return complex(ref_value + correction)


def _source_exact(base_params, L1, L2, V, h, K, target, T):
    exact = ExactSmallRubyThermal(int(L1), int(L2), base_params)
    exact.h0 = np.asarray(exact.h0, dtype=complex) - float(h) * np.asarray(K, dtype=complex)
    exact.h0 = 0.5 * (exact.h0 + exact.h0.conj().T)
    exact.diagonalize(float(V))
    mu = exact.solve_mu(float(target), float(T))
    value = _exact_onebody_expectation(exact, K, mu, T)
    return exact, float(mu), complex(value)


def _other_channel(source: str) -> str:
    return "opposite" if source == "same" else "same"


def _plot_summary(path, sources, Vvalues, hvalues, J_ed, J_gw, J_sox, dpi):
    ns = len(sources)
    nV = len(Vvalues)
    fig, axes = plt.subplots(
        ns,
        nV,
        figsize=(5.4 * nV, 4.1 * ns),
        squeeze=False,
        sharex=True,
    )
    order = np.argsort(hvalues)
    hs = np.asarray(hvalues)[order]
    for isrc, src in enumerate(sources):
        for iv, V in enumerate(Vvalues):
            ax = axes[isrc, iv]
            ax.plot(hs, J_ed[isrc, iv, order], marker="o", label="ED")
            ax.plot(hs, J_gw[isrc, iv, order], marker="s", label="GW")
            ax.plot(hs, J_sox[isrc, iv, order], marker="^", label="GW+SOX")
            ax.axhline(0.0, linewidth=0.8)
            ax.set_title(f"{_SOURCE_TEX[src]}, V={V:g}")
            ax.set_xlabel(r"source strength $h$")
            ax.set_ylabel(r"$\langle K_{\rm source}\rangle$")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Finite-source current benchmark", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    if args.T <= 0.0:
        raise ValueError("--T must be positive")
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal requires at most 16 sites")

    hvalues = _unique_descending_nonnegative(args.h)
    Vvalues = np.asarray([float(x) for x in args.V], dtype=float)
    if np.any(~np.isfinite(Vvalues)):
        raise ValueError("all --V values must be finite")
    sources = list(dict.fromkeys(str(x) for x in args.source))

    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell
    params0 = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    base_exact = ExactSmallRubyThermal(args.L1, args.L2, params0)
    base_h0 = np.asarray(base_exact.h0, dtype=complex)
    Vunit = np.asarray(base_exact.Vunit, dtype=complex)

    channel_K = {
        src: np.asarray(
            base_exact.pseudospin_operator(_SOURCE_TO_CHANNEL[src], (0.0, 0.0)),
            dtype=complex,
        )
        for src in sources
    }
    # Always monitor both current channels, even when only one is sourced.
    monitor_sources = ["same", "opposite"]
    monitor_K = {
        src: np.asarray(
            base_exact.pseudospin_operator(_SOURCE_TO_CHANNEL[src], (0.0, 0.0)),
            dtype=complex,
        )
        for src in monitor_sources
    }

    grid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=int(args.nw),
        nOmega=int(args.nomega),
        T=float(args.T),
    )
    gw_opts = GWOptions(
        target_filling=float(target),
        max_iter=int(args.gw_max_iter),
        tol=float(args.gw_tol),
        mixing=float(args.mixing),
        mixing_method=str(args.mixing_method),
        verbose=bool(args.verbose),
        momentum_backend=str(args.backend),
    )
    sox_opts = SOXOptions(n_quad=int(args.sox_nquad), tail_complete=True)

    ns = len(sources)
    nV = len(Vvalues)
    nh = len(hvalues)
    nm = len(monitor_sources)
    shape = (ns, nV, nh)
    obs_shape = (ns, nV, nh, nm)

    mu_ed = np.full(shape, np.nan)
    mu_gw = np.full(shape, np.nan)
    mu_sox = np.full(shape, np.nan)
    obs_ed = np.full(obs_shape, np.nan)
    obs_gw = np.full(obs_shape, np.nan)
    obs_sox = np.full(obs_shape, np.nan)
    gw_converged = np.zeros(shape, dtype=bool)
    sox_converged = np.zeros(shape, dtype=bool)
    gw_iterations = np.zeros(shape, dtype=int)
    sox_iterations = np.zeros(shape, dtype=int)
    gw_residual = np.full(shape, np.nan)
    sox_residual = np.full(shape, np.nan)
    max_sigma_sox = np.full(shape, np.nan)

    print("=== finite-source current benchmark: ED / GW / GW+SOX ===")
    print(
        f"cluster={args.L1}x{args.L2}, sites={base_exact.n_sites}, "
        f"T={args.T:g}, filling={args.filling:g}, target particles={target:g}"
    )
    print(f"V={Vvalues.tolist()}")
    print(f"h continuation={hvalues.tolist()}")
    print(f"sources={sources}")

    for isrc, src in enumerate(sources):
        Ksrc = channel_K[src]
        print(f"\n##### source={src} ({_SOURCE_TO_CHANNEL[src]}) #####")

        for iv, V in enumerate(Vvalues):
            print(f"\n=== source={src}, V={V:g} ===")
            Vq = (float(V) * Vunit)[None, None]

            # Each (source,V) branch is selected independently at the largest h.
            gw_initial = None
            sox_initial = None

            for ih, h in enumerate(hvalues):
                print(f"\n  -- h={h:.8g} --")
                h0h_matrix = base_h0 - float(h) * Ksrc
                h0h_matrix = 0.5 * (h0h_matrix + h0h_matrix.conj().T)
                h0h = h0h_matrix[None, None]

                # ---------------- exact ED ----------------
                exact_h, mu_e, _ = _source_exact(
                    params0,
                    args.L1,
                    args.L2,
                    float(V),
                    float(h),
                    Ksrc,
                    target,
                    args.T,
                )
                mu_ed[isrc, iv, ih] = mu_e
                for imon, msrc in enumerate(monitor_sources):
                    obs_ed[isrc, iv, ih, imon] = float(
                        _exact_onebody_expectation(
                            exact_h, monitor_K[msrc], mu_e, args.T
                        ).real
                    )

                # ---------------- ordinary self-consistent GW ----------------
                gw = solve_matrix_gw_fast(
                    h0h,
                    Vq,
                    grid,
                    opts=gw_opts,
                    initial=gw_initial,
                )
                gw_initial = gw
                gw_converged[isrc, iv, ih] = bool(gw.converged)
                gw_iterations[isrc, iv, ih] = int(gw.iterations)
                gw_residual[isrc, iv, ih] = float(gw.final_error)
                mu_gw[isrc, iv, ih] = float(gw.mu)
                _require(
                    bool(gw.converged),
                    f"GW did not converge for source={src}, V={V:g}, h={h:g}: "
                    f"residual={gw.final_error:.3e}",
                    args.allow_unconverged,
                )
                _, sigma_f_gw, _, _ = compute_sigma_gw_split_components(
                    gw.G,
                    gw.W,
                    Vq,
                    grid,
                    h0h,
                    gw.mu,
                    gw.Sigma_H,
                    backend=args.backend,
                )
                h_static_gw = h0h + gw.Sigma_H[None, None] + sigma_f_gw
                for imon, msrc in enumerate(monitor_sources):
                    obs_gw[isrc, iv, ih, imon] = float(
                        _bilinear_expectation_tail_completed(
                            gw.G,
                            monitor_K[msrc],
                            grid,
                            h_static_gw,
                            gw.mu,
                        ).real
                    )

                # ---------------- self-consistent GW+SOX ----------------
                gwsox = solve_matrix_gw_sox(
                    h0h,
                    Vq,
                    grid,
                    opts=gw_opts,
                    sox_opts=sox_opts,
                    initial=sox_initial if sox_initial is not None else gw,
                )
                sox_initial = gwsox
                sox_converged[isrc, iv, ih] = bool(gwsox.converged)
                sox_iterations[isrc, iv, ih] = int(gwsox.iterations)
                sox_residual[isrc, iv, ih] = float(gwsox.final_error)
                mu_sox[isrc, iv, ih] = float(gwsox.mu)
                max_sigma_sox[isrc, iv, ih] = float(np.max(np.abs(gwsox.Sigma_SOX)))
                _require(
                    bool(gwsox.converged),
                    f"GW+SOX did not converge for source={src}, V={V:g}, h={h:g}: "
                    f"residual={gwsox.final_error:.3e}",
                    args.allow_unconverged,
                )
                h_static_sox = h0h + gwsox.Sigma_H[None, None] + gwsox.Sigma_F
                for imon, msrc in enumerate(monitor_sources):
                    obs_sox[isrc, iv, ih, imon] = float(
                        _bilinear_expectation_tail_completed(
                            gwsox.G,
                            monitor_K[msrc],
                            grid,
                            h_static_sox,
                            gwsox.mu,
                        ).real
                    )

                src_idx = monitor_sources.index(src)
                print(
                    f"     J_{src}: "
                    f"ED={obs_ed[isrc,iv,ih,src_idx]:+.9f}  "
                    f"GW={obs_gw[isrc,iv,ih,src_idx]:+.9f}  "
                    f"GW+SOX={obs_sox[isrc,iv,ih,src_idx]:+.9f}"
                )
                other = _other_channel(src)
                other_idx = monitor_sources.index(other)
                print(
                    f"     cross J_{other}: "
                    f"ED={obs_ed[isrc,iv,ih,other_idx]:+.3e}  "
                    f"GW={obs_gw[isrc,iv,ih,other_idx]:+.3e}  "
                    f"GW+SOX={obs_sox[isrc,iv,ih,other_idx]:+.3e}"
                )

    source_channel_indices = np.asarray(
        [monitor_sources.index(src) for src in sources], dtype=int
    )
    J_ed = np.empty(shape, dtype=float)
    J_gw = np.empty(shape, dtype=float)
    J_sox = np.empty(shape, dtype=float)
    for isrc, idx in enumerate(source_channel_indices):
        J_ed[isrc] = obs_ed[isrc, :, :, idx]
        J_gw[isrc] = obs_gw[isrc, :, :, idx]
        J_sox[isrc] = obs_sox[isrc, :, :, idx]

    # Also save an intensive convention useful for later finite-size scaling.
    norm = np.sqrt(float(ncell))
    J_ed_intensive = J_ed / norm
    J_gw_intensive = J_gw / norm
    J_sox_intensive = J_sox / norm

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "finite_source_current_benchmark.npz"
    csv_path = outdir / "finite_source_current_benchmark.csv"
    plot_path = outdir / "finite_source_current_benchmark.png"

    np.savez_compressed(
        npz_path,
        sources=np.asarray(sources),
        source_channels=np.asarray([_SOURCE_TO_CHANNEL[x] for x in sources]),
        monitor_sources=np.asarray(monitor_sources),
        monitor_channels=np.asarray([_SOURCE_TO_CHANNEL[x] for x in monitor_sources]),
        V=Vvalues,
        h=hvalues,
        J_ed=J_ed,
        J_gw=J_gw,
        J_gw_sox=J_sox,
        J_ed_intensive=J_ed_intensive,
        J_gw_intensive=J_gw_intensive,
        J_gw_sox_intensive=J_sox_intensive,
        observables_ed=obs_ed,
        observables_gw=obs_gw,
        observables_gw_sox=obs_sox,
        mu_ed=mu_ed,
        mu_gw=mu_gw,
        mu_gw_sox=mu_sox,
        gw_converged=gw_converged,
        gw_sox_converged=sox_converged,
        gw_iterations=gw_iterations,
        gw_sox_iterations=sox_iterations,
        gw_residual=gw_residual,
        gw_sox_residual=sox_residual,
        max_sigma_sox=max_sigma_sox,
        L1=int(args.L1),
        L2=int(args.L2),
        n_cells=int(ncell),
        n_sites=int(base_exact.n_sites),
        filling=float(args.filling),
        target_particles=float(target),
        T=float(args.T),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        source_convention=np.asarray("H(h)=H0-h*K_source"),
        continuation=np.asarray("independent for each (source,V), h descending to zero"),
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "source",
                "channel",
                "V",
                "h",
                "J_ED",
                "J_GW",
                "J_GW_SOX",
                "J_ED_intensive",
                "J_GW_intensive",
                "J_GW_SOX_intensive",
                "mu_ED",
                "mu_GW",
                "mu_GW_SOX",
                "GW_converged",
                "GW_SOX_converged",
                "GW_residual",
                "GW_SOX_residual",
            ]
        )
        for isrc, src in enumerate(sources):
            for iv, V in enumerate(Vvalues):
                for ih, h in enumerate(hvalues):
                    writer.writerow(
                        [
                            src,
                            _SOURCE_TO_CHANNEL[src],
                            f"{V:.16g}",
                            f"{h:.16g}",
                            f"{J_ed[isrc,iv,ih]:.16g}",
                            f"{J_gw[isrc,iv,ih]:.16g}",
                            f"{J_sox[isrc,iv,ih]:.16g}",
                            f"{J_ed_intensive[isrc,iv,ih]:.16g}",
                            f"{J_gw_intensive[isrc,iv,ih]:.16g}",
                            f"{J_sox_intensive[isrc,iv,ih]:.16g}",
                            f"{mu_ed[isrc,iv,ih]:.16g}",
                            f"{mu_gw[isrc,iv,ih]:.16g}",
                            f"{mu_sox[isrc,iv,ih]:.16g}",
                            bool(gw_converged[isrc,iv,ih]),
                            bool(sox_converged[isrc,iv,ih]),
                            f"{gw_residual[isrc,iv,ih]:.6e}",
                            f"{sox_residual[isrc,iv,ih]:.6e}",
                        ]
                    )

    if not args.no_plots:
        _plot_summary(
            plot_path,
            sources,
            Vvalues,
            hvalues,
            J_ed,
            J_gw,
            J_sox,
            args.dpi,
        )

    print("\nsaved:", npz_path)
    print("saved:", csv_path)
    if not args.no_plots:
        print("saved:", plot_path)


if __name__ == "__main__":
    main()
