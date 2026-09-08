#!/usr/bin/env python3
"""Zero-temperature ED scan for rectangular periodic Ruby clusters.

The default geometry is the 2x2 primitive-cell torus (24 microscopic sites),
which contains Gamma and all three M points.  This is complementary to the
specialized 18-site index-three torus, which contains Gamma and +/-Q but no M.

For 24 sites the calculation is matrix-free in the many-body Hamiltonian.  At
primitive filling n=3 the Hilbert dimension is C(24,12)=2,704,156, so a broad V
scan can still be expensive; begin with one or a few ``--V-list`` points.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.ed_cluster import (
    ED_CLUSTER_CHANNELS,
    RubyEDClusterSolver,
    rectangular_momenta,
)
from rubycgw.ed_cluster_chi import zero_temperature_cluster_susceptibility
from rubycgw.ed18_plot import leading_subspace_weights
from rubycgw.model import RubyParameters


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
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=2)
    p.add_argument("--Vmin", type=float, default=0.0)
    p.add_argument("--Vmax", type=float, default=2.0)
    p.add_argument("--dV", type=float, default=0.1)
    p.add_argument("--V-list", nargs="*", type=float, default=None)
    p.add_argument("--filling", type=float, default=3.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--n-eigs", type=int, default=6)
    p.add_argument("--eig-tol", type=float, default=1e-9)
    p.add_argument("--deg-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--precompute-transitions", action="store_true")
    p.add_argument("--with-chi", action="store_true")
    p.add_argument("--chi-tol", type=float, default=1e-8)
    p.add_argument("--chi-maxiter", type=int, default=20000)
    p.add_argument("--chi-singular-tol", type=float, default=1e-10)
    p.add_argument("--out", default="ed24_n3_scan.npz")
    p.add_argument("--plot", default=None)
    p.add_argument("--modes-plot", default=None)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _v_grid(args):
    if args.V_list:
        return np.asarray(sorted(set(float(x) for x in args.V_list)), dtype=float)
    if args.dV <= 0:
        raise ValueError("--dV must be positive")
    n = int(np.floor((args.Vmax - args.Vmin) / args.dV + 0.5))
    vals = args.Vmin + args.dV * np.arange(n + 1, dtype=float)
    return vals[vals <= args.Vmax + 1e-12]


def _mode_weights(evals, evecs):
    w, d = leading_subspace_weights(np.asarray(evals)[None, ...], np.asarray(evecs)[None, ...])
    return w[0], d[0]


def _save_summary(path, V, energies, gaps, q_names, Sevals, chi_evals, dpi):
    nrows = 4 if chi_evals is not None else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(9.5, 2.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(V, energies[:, 0], marker="o")
    axes[0].set_ylabel(r"$E_0$")
    axes[0].grid(alpha=0.2)

    for j in range(1, min(energies.shape[1], 6)):
        axes[1].plot(V, energies[:, j] - energies[:, 0], marker="o", ms=3, label=fr"$E_{j}-E_0$")
    axes[1].plot(V, gaps, linewidth=2.0, label="gap above GS manifold")
    axes[1].set_ylabel("energy gap")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    for iq, name in enumerate(q_names):
        axes[2].plot(V, Sevals[:, iq, 0], marker="o", label=str(name))
    axes[2].set_ylabel(r"$S_{\max}(q)$")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8, ncol=2)

    if chi_evals is not None:
        for iq, name in enumerate(q_names):
            axes[3].plot(V, chi_evals[:, iq, 0], marker="o", label=str(name))
        axes[3].set_ylabel(r"$\chi^{reg}_{\max}(q)$")
        axes[3].grid(alpha=0.2)
        axes[3].legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel(r"$V$")
    fig.suptitle("rectangular Ruby ED cluster", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _save_modes(path, V, q_names, channels, S, Sevals, Sevecs, chi, chi_evals, chi_evecs, dpi):
    nq = len(q_names)
    nrows = 4 if chi is not None else 2
    fig, axes = plt.subplots(nrows, nq, figsize=(4.0 * nq, 3.0 * nrows), squeeze=False)
    labels = [_CHANNEL_TEX.get(str(x), str(x)) for x in channels]
    for iq, name in enumerate(q_names):
        ax = axes[0, iq]
        for ic, lab in enumerate(labels):
            ax.plot(V, np.real(S[:, iq, ic, ic]), label=lab)
        ax.set_title(str(name))
        ax.set_ylabel(r"$S_{\mu\mu}$")
        ax.grid(alpha=0.2)
        if iq == nq - 1:
            ax.legend(fontsize=7)

        sw, sd = leading_subspace_weights(Sevals[:, iq], Sevecs[:, iq])
        ax = axes[1, iq]
        for ic, lab in enumerate(labels):
            ax.plot(V, sw[:, ic], label=lab)
        ax.set_ylabel("leading S weight")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        if iq == nq - 1:
            ax.legend(fontsize=7)
        dims = sorted(set(int(x) for x in np.asarray(sd).reshape(-1)))
        ax.text(0.02, 0.95, f"d={dims}", transform=ax.transAxes, va="top", fontsize=8)

        if chi is not None:
            ax = axes[2, iq]
            for ic, lab in enumerate(labels):
                ax.plot(V, np.real(chi[:, iq, ic, ic]), label=lab)
            ax.set_ylabel(r"$\chi^{reg}_{\mu\mu}$")
            ax.grid(alpha=0.2)
            if iq == nq - 1:
                ax.legend(fontsize=7)

            cw, cd = leading_subspace_weights(chi_evals[:, iq], chi_evecs[:, iq])
            ax = axes[3, iq]
            for ic, lab in enumerate(labels):
                ax.plot(V, cw[:, ic], label=lab)
            ax.set_ylabel("leading chi weight")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
            if iq == nq - 1:
                ax.legend(fontsize=7)
            dims = sorted(set(int(x) for x in np.asarray(cd).reshape(-1)))
            ax.text(0.02, 0.95, f"d={dims}", transform=ax.transAxes, va="top", fontsize=8)

        axes[-1, iq].set_xlabel(r"$V$")
    fig.suptitle("channel-resolved rectangular-cluster ED", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    Vvals = _v_grid(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    t0 = time.perf_counter()
    solver = RubyEDClusterSolver(args.L1, args.L2, params, primitive_filling=args.filling)
    qvecs, q_names = rectangular_momenta(args.L1, args.L2)
    mem = solver.memory_estimate()

    print("=== rectangular Ruby ED ===")
    print(
        f"cluster={args.L1}x{args.L2} primitive cells, sites={solver.n_sites}, "
        f"particles={solver.n_particles}, dimension={solver.dimension:,}"
    )
    print(f"primitive filling={solver.primitive_filling:g}")
    print("allowed q: " + ", ".join(f"{n}={tuple(q)}" for n, q in zip(q_names, qvecs)))
    print(
        "estimated core memory before Lanczos vectors: "
        f"{mem['core_estimate_GiB']:.3f} GiB "
        f"(transition cache {mem['hopping_transition_cache_GiB']:.3f} GiB)"
    )
    if solver.n_sites == 24 and solver.n_particles == 12:
        print("NOTE: 24-site n=3 is a 2.70M-dimensional calculation; start with one V point.")
    if args.precompute_transitions:
        print("precomputing hopping transition cache ...")
        tp = time.perf_counter()
        solver.precompute_hopping_transitions()
        print(f"transition cache ready in {time.perf_counter()-tp:.1f} s")

    nv = len(Vvals)
    nq = len(qvecs)
    nc = len(ED_CLUSTER_CHANNELS)
    ne = int(args.n_eigs)
    energies = np.full((nv, ne), np.nan, dtype=float)
    ground_deg = np.zeros(nv, dtype=int)
    gaps = np.full(nv, np.nan, dtype=float)
    dEdV = np.full(nv, np.nan, dtype=float)
    S = np.zeros((nv, nq, nc, nc), dtype=complex)
    Sevals = np.zeros((nv, nq, nc), dtype=float)
    Sevecs = np.zeros((nv, nq, nc, nc), dtype=complex)
    means = np.zeros((nv, nq, nc), dtype=complex)

    if args.with_chi:
        chi = np.zeros((nv, nq, nc, nc), dtype=complex)
        chi_evals = np.zeros((nv, nq, nc), dtype=float)
        chi_evecs = np.zeros((nv, nq, nc, nc), dtype=complex)
        chi_sing = np.zeros((nv, nq, nc, nc), dtype=complex)
        chi_sing_evals = np.zeros((nv, nq, nc), dtype=float)
        chi_is_singular = np.zeros((nv, nq), dtype=bool)
        chi_residual_max = np.zeros((nv, nq), dtype=float)
    else:
        chi = chi_evals = chi_evecs = None

    previous = None
    for iv, V in enumerate(Vvals):
        ts = time.perf_counter()
        spec = solver.solve(
            float(V),
            n_eigs=ne,
            tol=args.eig_tol,
            maxiter=args.maxiter,
            v0=previous,
            degeneracy_tol=args.deg_tol,
        )
        previous = spec.eigenvectors[:, 0].real.copy()
        energies[iv, : len(spec.energies)] = spec.energies
        ground_deg[iv] = spec.ground_multiplicity
        gaps[iv] = spec.gap_above_manifold
        dEdV[iv] = spec.interaction_expectation
        print(
            f"V={V:7.4f} E0={spec.energies[0]:+.10f} deg={spec.ground_multiplicity} "
            f"gap={spec.gap_above_manifold:.4e} <D>={spec.interaction_expectation:.6f}"
        )

        for iq, (qname, qvec) in enumerate(zip(q_names, qvecs)):
            sf = solver.structure_factor(spec, qvec)
            S[iv, iq] = sf.matrix
            Sevals[iv, iq] = sf.eigenvalues
            Sevecs[iv, iq] = sf.eigenvectors
            means[iv, iq] = sf.mean
            sw, sd = _mode_weights(sf.eigenvalues, sf.eigenvectors)
            wc = float(sw[4] + sw[5])
            print(f"    {qname:6s}: Smax={sf.eigenvalues[0]:.6f} d={int(sd)} current_weight={wc:.3f}")

            if args.with_chi:
                cres = zero_temperature_cluster_susceptibility(
                    solver,
                    spec,
                    qvec,
                    solve_tol=args.chi_tol,
                    maxiter=args.chi_maxiter,
                    singular_tol=args.chi_singular_tol,
                )
                chi[iv, iq] = cres.regular_matrix
                chi_evals[iv, iq] = cres.regular_eigenvalues
                chi_evecs[iv, iq] = cres.regular_eigenvectors
                chi_sing[iv, iq] = cres.ground_singular_matrix
                chi_sing_evals[iv, iq] = cres.ground_singular_eigenvalues
                chi_is_singular[iv, iq] = cres.is_singular
                chi_residual_max[iv, iq] = float(np.max(cres.solver_residuals))
                cw, cd = _mode_weights(cres.regular_eigenvalues, cres.regular_eigenvectors)
                wc_chi = float(cw[4] + cw[5])
                print(
                    f"             chi_reg={cres.regular_eigenvalues[0]:.6g} d={int(cd)} "
                    f"current_weight={wc_chi:.3f} "
                    f"C_GS={cres.ground_singular_eigenvalues[0]:.3e} "
                    f"res={chi_residual_max[iv, iq]:.2e}"
                )
        print(f"    point wall time: {time.perf_counter()-ts:.1f} s")

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        V=Vvals,
        L1=int(args.L1),
        L2=int(args.L2),
        n_cells=solver.n_cells,
        n_sites=solver.n_sites,
        n_particles=solver.n_particles,
        dimension=solver.dimension,
        filling=float(solver.primitive_filling),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        channels=np.asarray(ED_CLUSTER_CHANNELS),
        q_names=np.asarray(q_names),
        q_vectors=qvecs,
        energies=energies,
        ground_multiplicity=ground_deg,
        gap_above_manifold=gaps,
        interaction_expectation=dEdV,
        structure_matrix=S,
        structure_eigenvalues=Sevals,
        structure_eigenvectors=Sevecs,
        structure_means=means,
        memory_core_estimate_GiB=float(mem["core_estimate_GiB"]),
        cluster_warning=np.asarray(
            "Rectangular finite PBC torus; exact only for this cluster. "
            "For 2x2 the accessible primitive momenta are Gamma and three M points; Q=(1/3,1/3) is absent."
        ),
    )
    if args.with_chi:
        payload.update(
            susceptibility_regular=chi,
            susceptibility_regular_eigenvalues=chi_evals,
            susceptibility_regular_eigenvectors=chi_evecs,
            susceptibility_ground_singular=chi_sing,
            susceptibility_ground_singular_eigenvalues=chi_sing_evals,
            susceptibility_is_singular=chi_is_singular,
            susceptibility_residual_max=chi_residual_max,
        )
    np.savez_compressed(out, **payload)

    summary = Path(args.plot) if args.plot else out.with_name(out.stem + ".png")
    modes = Path(args.modes_plot) if args.modes_plot else out.with_name(out.stem + "_modes.png")
    _save_summary(summary, Vvals, energies, gaps, q_names, Sevals, chi_evals, args.dpi)
    _save_modes(modes, Vvals, q_names, ED_CLUSTER_CHANNELS, S, Sevals, Sevecs, chi, chi_evals, chi_evecs, args.dpi)
    print(f"saved {out}")
    print(f"saved {summary}")
    print(f"saved {modes}")
    print(f"total wall time: {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
