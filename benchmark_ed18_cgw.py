#!/usr/bin/env python3
"""Benchmark 18-site finite-T ED against GW/cGW on the *same* PBC torus.

This script implements the comparison philosophy used in numerical benchmarks
of covariant response methods: compare the same observable on the same finite
geometry before discussing the thermodynamic limit.

Geometry
--------
The exact-diagonalization cluster is one index-three Ruby supercell (18 sites)
with PBC.  To match it exactly on the GW side this driver REQUIRES
``nk1=nk2=1`` for the 18-orbital supercell calculation.  In primitive-cell
language the three allowed momenta Gamma,+Q,-Q are folded into the 18 orbital
basis.  This removes the large ambiguity of comparing a three-cell ED torus to
a dense-k thermodynamic cGW calculation.

Observables
-----------
For the six normalized pseudospin channels

    x_even, x_odd, y_even, y_odd, z_same, z_opposite

we compare at primitive Gamma and Q=(1/3,1/3):

1. ED connected imaginary-time correlator

       C_ab(q,tau)=<O_a^dag(q,tau) O_b(q,0)>_c

   from canonical thermal typicality/Krylov propagation.

2. SC-GW ``GG`` bubble C_ab(q,tau), obtained from chi_GG(q,iOmega_m) and a
   truncated inverse bosonic Matsubara transform.  This diagnoses whether the
   one-particle GW background already disagrees with ED.

3. Static chi(q,iOmega=0) from ED, GG, and the production cGW vertex solver.
   This diagnoses how much of the discrepancy is introduced by vertex
   corrections.

Important scope
---------------
The current production cGW vertex kernel supports only zero *external bosonic
frequency*.  Therefore the full cGW result is compared to ED at iOmega=0 only;
the tau-resolved curve is ED versus GG.  Extending the full H/F/MT/AL vertex to
finite external bosonic frequency is a separate derivation and is intentionally
not faked by Fourier transforming a static result.

The ED calculation is canonical fixed-N while GW/cGW is grand canonical at a
chemical potential adjusted to the target average filling.  The geometry,
Hamiltonian parameters, temperature, channel normalization and PBC are matched,
but this finite-size ensemble difference remains and is reported explicitly.

Example
-------
First generate a converged same-torus SC-GW checkpoint, e.g.

    python run_supercell_gw.py --V 1 --primitive-filling 3 --T 0.08 \
        --nk1 1 --nk2 1 --nw 55 --nomega 12

then run

    python benchmark_ed18_cgw.py --V 1 --filling 3 --T 0.08 \
        --nw 55 --nomega 12 --stage full --out bench_n3_V1_T008.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.checkpoint import checkpoint_filename, load_supercell_checkpoint
from rubycgw.ed18 import ED18_CHANNELS, ED18Solver, Q_GAMMA, Q_PERIOD3
from rubycgw.ed18_thermal import random_phase_trace_vectors, thermal_response
from rubycgw.ed_cgw_benchmark import (
    bosonic_iomega_to_tau,
    bubble_iomega,
    eigh_desc_hermitian,
    project_complex_q,
    relative_frobenius_error,
    static_response_from_gammas,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import supercell_pseudospin_harmonic_vertices
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    physical_symmetric_susceptibility,
    solve_vertex_q0,
)
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    density_from_G_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from rubycgw.supercell_gw_split import compute_sigma_gw_split_matrix


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
    p.add_argument("--filling", "--primitive-filling", type=float, default=3.0)
    p.add_argument("--T", "--temperature", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)

    # Same-torus cGW: these are fixed to one supercell k point by design.
    p.add_argument("--nk1", type=int, default=1)
    p.add_argument("--nk2", type=int, default=1)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")

    p.add_argument(
        "--channels",
        nargs="+",
        default=list(ED18_CHANNELS),
        help="Pseudospin channels. Default: all six ED channels.",
    )
    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="full",
        help="Static cGW stage. Tau-resolved comparison is always ED versus GG.",
    )
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-verbose", action="store_true")

    p.add_argument("--thermal-samples", type=int, default=8)
    p.add_argument(
        "--tau-points",
        type=int,
        default=33,
        help="Odd ED imaginary-time grid size; 33 is recommended for benchmark plots.",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="benchmark_ed18_cgw.npz")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _checkpoint(args, params, grid):
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
    else:
        path = Path(args.checkpoint_dir) / checkpoint_filename(
            args.V, args.filling, grid
        )
    if not path.exists():
        raise FileNotFoundError(
            f"same-torus SC-GW checkpoint not found: {path}\n"
            "Generate it with, for example:\n"
            f"  python run_supercell_gw.py --V {args.V:g} "
            f"--primitive-filling {args.filling:g} --T {args.T:g} "
            f"--nk1 1 --nk2 1 --nw {args.nw} --nomega {args.nomega}"
        )
    seed, meta, density_saved = load_supercell_checkpoint(
        path, params, grid, args.filling
    )
    if not bool(meta.get("converged", False)):
        raise ValueError("benchmark requires a converged zero-source SC-GW checkpoint")
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("benchmark requires a zero-source SC-GW checkpoint")
    return path, seed, meta, density_saved


def _rebuild_scgw(seed, params, grid, backend):
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(h0, grid, seed.mu, seed.Sigma_H, seed.Sigma_GW)
    density = density_from_G_matrix(
        G, grid, h0=h0, mu=seed.mu, sigma_h=seed.Sigma_H
    )
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G, W, Vq, grid, h0, seed.mu, seed.Sigma_H, backend=backend
    )
    rH = float(np.max(np.abs(sigma_h_out - seed.Sigma_H)))
    rGW = float(np.max(np.abs(sigma_gw_out - seed.Sigma_GW)))
    return h0, Vq, G, P, W, density, rH, rGW


def _real_harmonic_vertices(channels):
    KG, _, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="q0")
    KC, _, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="Qc")
    KS, _, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="Qs")
    KQ = np.concatenate([KC, KS], axis=0)
    return np.asarray(KG), np.asarray(KQ)


def _solve_static_cgw(G, W, Vq, vertices, grid, args, label):
    """Return a physical symmetric static response in a real Hermitian basis."""
    vertices = np.asarray(vertices, dtype=complex)
    if args.stage == "gg":
        gammas = [np.broadcast_to(K, G.shape).copy() for K in vertices]
    else:
        opts = SupercellVertexOptions(
            max_iter=args.vertex_max_iter,
            tol=args.vertex_tol,
            mixing=args.vertex_mixing,
            solver=args.vertex_solver,
            gmres_restart=args.vertex_gmres_restart,
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=(args.stage == "full"),
            verbose=args.vertex_verbose,
            momentum_backend=args.momentum_backend,
        )
        gammas = []
        for i, K in enumerate(vertices, start=1):
            print(f"  {label}: vertex {i}/{len(vertices)} ({args.stage})")
            res = solve_vertex_q0(G, W, Vq, K, grid, opts=opts)
            if not res.converged:
                raise RuntimeError(
                    f"{label} vertex {i} failed: residual={res.final_error:.3e}"
                )
            gammas.append(res.Gamma)
    raw = static_response_from_gammas(G, vertices, gammas, grid)
    sym, imag_max = physical_symmetric_susceptibility(raw)
    return np.asarray(sym, dtype=float), float(imag_max)


def _leading_weights(matrix, tol=1e-8):
    vals, vecs = eigh_desc_hermitian(matrix)
    scale = max(1.0, abs(float(vals[0])))
    d = int(np.count_nonzero(np.abs(vals - vals[0]) <= float(tol) * scale))
    d = max(d, 1)
    weights = np.sum(np.abs(vecs[:, :d]) ** 2, axis=1) / float(d)
    weights /= max(float(np.sum(weights)), 1e-300)
    return vals, vecs, weights, d


def _mode_summary(matrix, channels):
    vals, _, w, d = _leading_weights(matrix)
    zidx = [i for i, ch in enumerate(channels) if str(ch).startswith("z_")]
    current = float(np.sum(w[zidx])) if zidx else 0.0
    orbital = 1.0 - current
    pieces = []
    for i in np.argsort(w)[::-1]:
        if w[i] >= 0.08:
            pieces.append(f"{channels[i]}:{w[i]:.3f}")
    return float(vals[0]), current, orbital, d, " ".join(pieces)


def _plot_tau(tau, beta, ed_tau, gg_tau, channels, out, dpi):
    x = np.asarray(tau, dtype=float) / float(beta)
    groups = [
        ("x", [i for i, c in enumerate(channels) if str(c).startswith("x_")]),
        ("y", [i for i, c in enumerate(channels) if str(c).startswith("y_")]),
        ("z/current", [i for i, c in enumerate(channels) if str(c).startswith("z_")]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.5), sharex=True)
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    for row, (gname, idxs) in enumerate(groups):
        for iq in range(2):
            ax = axes[row, iq]
            for ic in idxs:
                label = _CHANNEL_TEX.get(str(channels[ic]), str(channels[ic]))
                ax.plot(x, np.real(ed_tau[iq, :, ic, ic]), label=f"ED {label}")
                ax.plot(
                    x,
                    np.real(gg_tau[iq, :, ic, ic]),
                    linestyle="--",
                    label=f"GG {label}",
                )
            ax.axvline(0.5, linestyle=":", linewidth=0.9, alpha=0.6)
            ax.grid(alpha=0.2)
            ax.set_ylabel(r"$C_{\mu\mu}(q,\tau)$")
            if row == 0:
                ax.set_title(qtitles[iq])
            if row == 2:
                ax.set_xlabel(r"$\tau/\beta$")
            ax.legend(fontsize=7, ncol=2)
            ax.text(0.02, 0.05, gname, transform=ax.transAxes, fontsize=9)
    fig.suptitle(
        "same 18-site torus: ED imaginary-time correlation vs SC-GW GG bubble",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def _plot_static(ed, gg, cgw, channels, stage, out, dpi):
    labels = [_CHANNEL_TEX.get(str(ch), str(ch)) for ch in channels]
    xch = np.arange(len(channels), dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 11.0))
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    methods = [("ED", ed), ("GG", gg), (stage, cgw)]
    for iq in range(2):
        ax = axes[0, iq]
        for name, mats in methods:
            ax.plot(
                xch,
                np.real(np.diag(mats[iq])),
                marker="o",
                label=name,
            )
        ax.set_xticks(xch, labels, rotation=35, ha="right")
        ax.set_ylabel(r"diagonal $\chi_{\mu\mu}$")
        ax.set_title(qtitles[iq])
        ax.grid(alpha=0.2)
        ax.legend()

        ax = axes[1, iq]
        for name, mats in methods:
            vals, _ = eigh_desc_hermitian(mats[iq])
            ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=name)
        ax.set_xlabel("susceptibility eigenvalue index")
        ax.set_ylabel(r"$\lambda_\alpha(\chi)$")
        ax.grid(alpha=0.2)
        ax.legend()

        ax = axes[2, iq]
        for name, mats in methods:
            _, _, w, d = _leading_weights(mats[iq])
            ax.plot(xch, w, marker="o", label=f"{name} (d={d})")
        ax.set_xticks(xch, labels, rotation=35, ha="right")
        ax.set_ylabel("leading-eigenspace weight")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "same 18-site torus: static susceptibility ED vs GG vs cGW",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    if args.nk1 != 1 or args.nk2 != 1:
        raise ValueError(
            "same-torus benchmark requires --nk1 1 --nk2 1. "
            "Use dense-k cGW only after this finite-size benchmark."
        )
    if args.T <= 0.0:
        raise ValueError("T must be positive")
    if args.tau_points < 3 or args.tau_points % 2 != 1:
        raise ValueError("--tau-points must be an odd integer >=3")

    channels = tuple(str(x) for x in args.channels)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )

    print("=== same-torus 18-site ED vs GW/cGW benchmark ===")
    print(
        f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, "
        f"ti={args.ti:g}, t1={args.t1:g}, t2={args.t2:g}"
    )
    print(f"cGW grid: nk=1x1 supercell, nw={args.nw}, nOmega={args.nomega}")
    print("channels:", ", ".join(channels))
    print("ED ensemble: canonical fixed N")
    print("GW/cGW ensemble: grand canonical, mu adjusted to target average filling")

    # ------------------------------------------------------------------
    # Exact finite-T ED on the 18-site torus.
    # ------------------------------------------------------------------
    ed_solver = ED18Solver(
        RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0),
        primitive_filling=args.filling,
    )
    traces = random_phase_trace_vectors(
        ed_solver.dimension, args.thermal_samples, seed=args.seed
    )
    spec = ed_solver.solve(args.V, n_eigs=4)
    ed_responses = []
    for qname, q in (("Gamma", Q_GAMMA), ("Q", Q_PERIOD3)):
        print(f"ED finite-T response: {qname}")
        ed_responses.append(
            thermal_response(
                ed_solver,
                args.V,
                args.T,
                q,
                channels=channels,
                trace_vectors=traces,
                tau_points=args.tau_points,
                energy_shift=spec.energies[0],
            )
        )
    tau = np.asarray(ed_responses[0].tau_grid, dtype=float)
    ed_tau = np.stack([x.correlation_tau_matrix for x in ed_responses], axis=0)
    ed_static = np.stack([x.susceptibility_matrix for x in ed_responses], axis=0)

    # ------------------------------------------------------------------
    # SC-GW background on exactly one 18-site supercell PBC torus.
    # ------------------------------------------------------------------
    checkpoint, seed, meta, _ = _checkpoint(args, params, grid)
    print("SC-GW checkpoint:", checkpoint)
    h0, Vq, G, P, W, density, rH, rGW = _rebuild_scgw(
        seed, params, grid, args.momentum_backend
    )
    sc_res = max(rH, rGW)
    print(
        f"SC-GW fixed-point check: rH={rH:.3e}, rGW={rGW:.3e}, "
        f"max={sc_res:.3e}, n_primitive={np.sum(density)/3.0:.10f}"
    )
    if sc_res > args.max_scgw_residual:
        raise RuntimeError(
            f"checkpoint residual {sc_res:.3e} exceeds --max-scgw-residual"
        )

    KG, KQreal = _real_harmonic_vertices(channels)

    # Dynamic GG bubble.  For Q the 12x12 real Qc/Qs response is projected to
    # the exact complex O_Q=(Qc-iQs)/sqrt(2) convention used by ED.
    gg_iw_G = bubble_iomega(G, KG, grid)
    gg_iw_Q_real = bubble_iomega(G, KQreal, grid)
    gg_iw_Q = project_complex_q(gg_iw_Q_real, len(channels))
    gg_tau_G = bosonic_iomega_to_tau(gg_iw_G, grid, tau)
    gg_tau_Q = bosonic_iomega_to_tau(gg_iw_Q, grid, tau)
    gg_tau = np.stack([gg_tau_G, gg_tau_Q], axis=0)

    izero = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
    gg_static_G = 0.5 * (gg_iw_G[izero] + gg_iw_G[izero].conj().T)
    gg_static_Q = 0.5 * (gg_iw_Q[izero] + gg_iw_Q[izero].conj().T)
    gg_static = np.stack([gg_static_G, gg_static_Q], axis=0)

    # ------------------------------------------------------------------
    # Static cGW vertex correction at iOmega=0.
    # ------------------------------------------------------------------
    print(f"static cGW stage: {args.stage}")
    cgw_G_real, imag_G = _solve_static_cgw(
        G, W, Vq, KG, grid, args, "Gamma/q0"
    )
    cgw_Q_real, imag_Q = _solve_static_cgw(
        G, W, Vq, KQreal, grid, args, "Q/QcQs"
    )
    cgw_Q = project_complex_q(cgw_Q_real, len(channels))
    cgw_Q = 0.5 * (cgw_Q + cgw_Q.conj().T)
    cgw_static = np.stack(
        [np.asarray(cgw_G_real, dtype=complex), cgw_Q], axis=0
    )

    n = len(channels)
    qcc = cgw_Q_real[:n, :n]
    qss = cgw_Q_real[n:, n:]
    qcs = cgw_Q_real[:n, n:]
    qscale = max(float(np.linalg.norm(0.5 * (qcc + qss))), 1e-300)
    qc_qs_anisotropy = float(np.linalg.norm(qcc - qss) / qscale)
    qc_qs_cross = float(np.linalg.norm(qcs) / qscale)

    # ------------------------------------------------------------------
    # Human-readable diagnostic summary.
    # ------------------------------------------------------------------
    print("\n=== static susceptibility summary ===")
    for iq, qname in enumerate(("Gamma", "Q")):
        print(f"{qname}:")
        for name, mats in (("ED", ed_static), ("GG", gg_static), (args.stage, cgw_static)):
            lam, wc, wo, d, text = _mode_summary(mats[iq], channels)
            print(
                f"  {name:8s} lambda_max={lam:+.7e}  d={d}  "
                f"current_weight={wc:.3f} orbital_weight={wo:.3f}  {text}"
            )
        print(
            f"  relative Frobenius error: GG/ED="
            f"{relative_frobenius_error(ed_static[iq], gg_static[iq]):.4f}, "
            f"{args.stage}/ED="
            f"{relative_frobenius_error(ed_static[iq], cgw_static[iq]):.4f}"
        )
    print(
        f"Q folded-harmonic diagnostics: ||Qc-Qs||/scale={qc_qs_anisotropy:.3e}, "
        f"||Qc-Qs cross||/scale={qc_qs_cross:.3e}"
    )
    print(
        "Note: full cGW is static only here.  The tau plot compares ED with the "
        "GG bubble on the same SC-GW background."
    )

    # ------------------------------------------------------------------
    # Save raw benchmark data and figures.
    # ------------------------------------------------------------------
    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        V=float(args.V),
        filling=float(args.filling),
        temperature=float(args.T),
        beta=float(1.0 / args.T),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        channels=np.asarray(channels),
        q_names=np.asarray(["Gamma", "Q"]),
        q_vectors=np.asarray([Q_GAMMA, Q_PERIOD3]),
        geometry=np.asarray("one index-three 18-site supercell PBC torus"),
        ed_ensemble=np.asarray("canonical_fixed_N"),
        cgw_ensemble=np.asarray("grand_canonical_fixed_average_filling"),
        ed_n_particles=int(ed_solver.n_particles),
        ed_dimension=int(ed_solver.dimension),
        ed_thermal_samples=int(args.thermal_samples),
        ed_random_seed=int(args.seed),
        tau_grid=tau,
        ed_correlation_tau=ed_tau,
        ed_static_chi=ed_static,
        gg_m_values=np.asarray(grid.m_values),
        gg_Omega=np.asarray(grid.Omega),
        gg_iomega_Gamma=gg_iw_G,
        gg_iomega_Q=gg_iw_Q,
        gg_correlation_tau=gg_tau,
        gg_static_chi=gg_static,
        cgw_stage=np.asarray(args.stage),
        cgw_static_chi=cgw_static,
        cgw_static_Q_real_harmonic=cgw_Q_real,
        cgw_discarded_imag_Gamma=float(imag_G),
        cgw_discarded_imag_Q_real=float(imag_Q),
        Qc_Qs_anisotropy=float(qc_qs_anisotropy),
        Qc_Qs_cross=float(qc_qs_cross),
        scgw_mu=float(seed.mu),
        scgw_density=np.asarray(density),
        scgw_residual_H=float(rH),
        scgw_residual_GW=float(rGW),
        scgw_checkpoint=np.asarray(str(checkpoint)),
        cgw_dynamic_scope=np.asarray(
            "full cGW only at external iOmega=0; tau-resolved curve is ED vs GG"
        ),
    )
    print("saved:", out)

    tau_plot = out.with_name(out.stem + "_tau.png")
    static_plot = out.with_name(out.stem + "_static.png")
    _plot_tau(tau, 1.0 / args.T, ed_tau, gg_tau, channels, tau_plot, args.dpi)
    _plot_static(ed_static, gg_static, cgw_static, channels, args.stage, static_plot, args.dpi)
    print("saved:", tau_plot)
    print("saved:", static_plot)

    print("\nInterpretation order:")
    print("  1) ED vs GG in tau: does the SC-GW one-particle background already differ?")
    print("  2) ED vs GG at iOmega=0: integrated background-response error.")
    print("  3) ED vs cGW at iOmega=0: do vertex corrections improve or worsen it?")
    print("  4) only after this same-torus test compare dense-k cGW to thermodynamic physics.")


if __name__ == "__main__":
    main()
