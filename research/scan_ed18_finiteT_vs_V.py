#!/usr/bin/env python3
"""Finite-temperature 18-site Ruby ED/Krylov scan versus V.

This driver is the finite-T counterpart of ``scan_ed18_vs_V.py``.  It works in
one fixed-particle-number Hilbert space (canonical ensemble) and evaluates both

  * equal-time pseudospin structure factors S(q,T), and
  * connected static Kubo susceptibilities chi(q,T)

at Gamma and Q=(1/3,1/3).  The default temperature is T=0.08 to match the
current cGW benchmark calculations.

Thermal traces are estimated with common random-phase vectors and sparse Krylov
imaginary-time propagation.  Reusing the same trace vectors at every V makes
stochastic errors highly correlated and keeps V-dependent curves smooth.

As for all 18-site ED results in this repository, the index-three torus does
NOT contain the period-two M point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.ed18 import ED18_CHANNELS, ED18Solver, Q_GAMMA, Q_PERIOD3, phase_fix_vector
from rubycgw.ed18_plot import leading_subspace_weights
from rubycgw.ed18_thermal import random_phase_trace_vectors, thermal_response
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
    p.add_argument("--Vmin", type=float, default=0.0)
    p.add_argument("--Vmax", type=float, default=2.0)
    p.add_argument("--dV", type=float, default=0.1)
    p.add_argument("--V-list", nargs="*", type=float, default=None)
    p.add_argument("--filling", type=float, default=3.0)
    p.add_argument("--temperature", "--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--thermal-samples", type=int, default=8)
    p.add_argument(
        "--tau-points",
        type=int,
        default=17,
        help="Odd Simpson-grid size for the imaginary-time Kubo integral.",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--eig-tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--out", default="ed18_n3_T008.npz")
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


def _mode_text(channels, vec, threshold=0.05):
    vec = phase_fix_vector(vec)
    weights = np.abs(vec) ** 2
    pieces = []
    for i in np.argsort(weights)[::-1]:
        if weights[i] < threshold:
            continue
        z = vec[i]
        pieces.append(
            f"{channels[i]}:{weights[i]:.3f}"
            f"({z.real:+.3f}{z.imag:+.3f}j)"
        )
    return " ".join(pieces) if pieces else "mixed"


def _plot_response(
    V,
    evals,
    evecs,
    matrices,
    channels,
    quantity,
    temperature,
    out,
    dpi,
):
    labels = [_CHANNEL_TEX.get(str(ch), str(ch)) for ch in channels]
    weights, dims = leading_subspace_weights(evals, evecs)
    diag = np.real(np.diagonal(matrices, axis1=-2, axis2=-1))

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0), sharex="col")
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    for iq in range(2):
        for m in range(min(3, evals.shape[-1])):
            axes[0, iq].plot(V, evals[:, iq, m], marker="o", ms=2.5, label=rf"$\lambda_{m+1}$")
        axes[0, iq].set_title(qtitles[iq])
        axes[0, iq].set_ylabel(f"{quantity} eigenvalues")
        axes[0, iq].legend(fontsize=8)
        axes[0, iq].grid(alpha=0.2)

        for ic, label in enumerate(labels):
            axes[1, iq].plot(V, diag[:, iq, ic], label=label)
        axes[1, iq].set_ylabel(rf"diagonal {quantity}$_{{\mu\mu}}$")
        axes[1, iq].legend(ncol=2, fontsize=8)
        axes[1, iq].grid(alpha=0.2)

        for ic, label in enumerate(labels):
            axes[2, iq].plot(V, weights[:, iq, ic], label=label)
        axes[2, iq].set_ylabel("leading-eigenspace channel weight")
        axes[2, iq].set_xlabel(r"$V$")
        axes[2, iq].set_ylim(-0.03, 1.03)
        axes[2, iq].legend(ncol=2, fontsize=8)
        axes[2, iq].grid(alpha=0.2)
        dset = sorted(set(int(x) for x in dims[:, iq]))
        axes[2, iq].set_title("leading eigenspace d=" + ",".join(map(str, dset)), fontsize=9)

    fig.suptitle(f"18-site canonical finite-T ED: {quantity}, T={temperature:g}", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    Vvals = _v_grid(args)
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.thermal_samples < 1:
        raise ValueError("--thermal-samples must be positive")
    if args.tau_points < 3 or args.tau_points % 2 != 1:
        raise ValueError("--tau-points must be an odd integer >=3")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    solver = ED18Solver(params, primitive_filling=args.filling)
    traces = random_phase_trace_vectors(
        solver.dimension, args.thermal_samples, seed=args.seed
    )

    print("=== 18-site Ruby finite-T ED/Krylov ===")
    print(f"filling={solver.primitive_filling:g}, particles={solver.n_particles}, dim={solver.dimension}")
    print(f"T={args.temperature:g}, beta={1.0/args.temperature:g}")
    print(f"ti={args.ti:g}, t1={args.t1:g}, t2={args.t2:g}")
    print(f"thermal samples={args.thermal_samples}, tau points={args.tau_points}, seed={args.seed}")
    print("canonical fixed-N ensemble; allowed q: Gamma and +/-Q only; M absent")

    nv = len(Vvals)
    nc = len(ED18_CHANNELS)
    qnames = np.asarray(["Gamma", "Q"])
    qvecs = np.asarray([Q_GAMMA, Q_PERIOD3])

    E0 = np.zeros(nv, dtype=float)
    ground_deg = np.zeros(nv, dtype=int)
    ground_gap = np.zeros(nv, dtype=float)
    zshift = np.zeros((nv, 2), dtype=float)
    means = np.zeros((nv, 2, nc), dtype=complex)
    Smat = np.zeros((nv, 2, nc, nc), dtype=complex)
    Seval = np.zeros((nv, 2, nc), dtype=float)
    Sevec = np.zeros((nv, 2, nc, nc), dtype=complex)
    Chimat = np.zeros((nv, 2, nc, nc), dtype=complex)
    Chieval = np.zeros((nv, 2, nc), dtype=float)
    Chivec = np.zeros((nv, 2, nc, nc), dtype=complex)

    previous = None
    for iv, V in enumerate(Vvals):
        spec = solver.solve(
            float(V),
            n_eigs=6,
            tol=args.eig_tol,
            maxiter=args.maxiter,
            v0=previous,
        )
        previous = spec.eigenvectors[:, 0].real.copy()
        E0[iv] = spec.energies[0]
        ground_deg[iv] = spec.ground_multiplicity
        ground_gap[iv] = spec.gap_above_manifold

        print(f"\nV={V:.6g}  E0={E0[iv]:+.10f}  GSdeg={ground_deg[iv]}  gap={ground_gap[iv]:.4e}")

        for iq, (qname, q) in enumerate(zip(qnames, qvecs)):
            resp = thermal_response(
                solver,
                float(V),
                args.temperature,
                q,
                trace_vectors=traces,
                tau_points=args.tau_points,
                energy_shift=E0[iv],
            )
            zshift[iv, iq] = resp.shifted_partition_estimate
            means[iv, iq] = resp.means
            Smat[iv, iq] = resp.structure_matrix
            Seval[iv, iq] = resp.structure_eigenvalues
            Sevec[iv, iq] = resp.structure_eigenvectors
            Chimat[iv, iq] = resp.susceptibility_matrix
            Chieval[iv, iq] = resp.susceptibility_eigenvalues
            Chivec[iv, iq] = resp.susceptibility_eigenvectors

            print(
                f"  {qname}: Smax={resp.structure_eigenvalues[0]:.6f}  "
                f"chi_max={resp.susceptibility_eigenvalues[0]:.6f}"
            )
            print(
                f"       chi lead: "
                + _mode_text(resp.channels, resp.susceptibility_eigenvectors[:, 0])
            )

    chi_weights, chi_dims = leading_subspace_weights(Chieval, Chivec)
    s_weights, s_dims = leading_subspace_weights(Seval, Sevec)

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        V=Vvals,
        filling=float(args.filling),
        temperature=float(args.temperature),
        beta=float(1.0 / args.temperature),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        n_sites=18,
        n_particles=solver.n_particles,
        dimension=solver.dimension,
        ensemble=np.asarray("canonical_fixed_N"),
        thermal_method=np.asarray("random_phase_typicality_Krylov_expm_multiply"),
        thermal_samples=int(args.thermal_samples),
        tau_points=int(args.tau_points),
        random_seed=int(args.seed),
        channels=np.asarray(ED18_CHANNELS),
        q_names=qnames,
        q_vectors=qvecs,
        E0=E0,
        ground_multiplicity=ground_deg,
        ground_gap=ground_gap,
        shifted_partition_estimate=zshift,
        thermal_means=means,
        thermal_structure_matrix=Smat,
        thermal_structure_eigenvalues=Seval,
        thermal_structure_eigenvectors=Sevec,
        thermal_structure_leading_weights=s_weights,
        thermal_structure_leading_dimension=s_dims,
        thermal_chi_matrix=Chimat,
        thermal_chi_eigenvalues=Chieval,
        thermal_chi_eigenvectors=Chivec,
        thermal_chi_leading_weights=chi_weights,
        thermal_chi_leading_dimension=chi_dims,
        cluster_warning=np.asarray(
            "18-site index-3 torus contains Gamma and +/-Q only; M is not commensurate"
        ),
    )
    print("\nsaved:", out)

    chi_plot = out.with_name(out.stem + "_chiT.png")
    s_plot = out.with_name(out.stem + "_ST.png")
    _plot_response(
        Vvals, Chieval, Chivec, Chimat, np.asarray(ED18_CHANNELS),
        r"$\chi(T)$", args.temperature, chi_plot, args.dpi,
    )
    _plot_response(
        Vvals, Seval, Sevec, Smat, np.asarray(ED18_CHANNELS),
        r"$S(T)$", args.temperature, s_plot, args.dpi,
    )
    print("saved:", chi_plot)
    print("saved:", s_plot)
    print("\nConvergence check: rerun with --thermal-samples 16 and/or --tau-points 25.")
    print("Reminder: this is canonical fixed-N finite-T ED; M is not represented.")


if __name__ == "__main__":
    main()
