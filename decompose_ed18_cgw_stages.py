#!/usr/bin/env python3
"""Decompose the same-torus ED/cGW mismatch by vertex stage.

This diagnostic starts from an existing ``benchmark_ed18_cgw.py`` NPZ and
reuses its finite-T ED and same-torus GG results.  It then solves the static
cGW vertex equation on the *same* one-supercell 18-site PBC torus in the
sequence

    GG -> H+F -> H+F+MT -> H+F+MT+AL (full),

so one can see exactly at which approximation stage the leading response
changes from orbital to loop-current character.

The expensive ED thermal trace and dynamic GG bubble are therefore NOT repeated.
Consecutive vertex stages are warm-started from the previous converged Gamma by
default.  Independent channel solves within each stage can optionally be run in
parallel with ``--vertex-workers``.

Important interpretation
------------------------
The plotted stage increments

    Delta_HF = chi_HF - chi_GG,
    Delta_MT = chi_HF+MT - chi_HF,
    Delta_AL = chi_full - chi_HF+MT

are *incremental changes after re-solving the vertex equation*.  They identify
which added kernel sector changes the physical response, but they are not bare,
additive single-diagram susceptibilities.  The full vertex result itself also
reports max norms of Gamma_H, Gamma_F, Gamma_MT, Gamma_AL1 and Gamma_AL2.

Example
-------

    python decompose_ed18_cgw_stages.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --nw 55 --nomega 12 ^
      --stages all ^
      --vertex-workers 2 ^
      --out bench_n3_V1_T008_stages.npz
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_ed18_cgw import (
    _checkpoint,
    _leading_weights,
    _mode_summary,
    _real_harmonic_vertices,
    _rebuild_scgw,
)
from rubycgw.ed_cgw_benchmark import (
    eigh_desc_hermitian,
    project_complex_q,
    relative_frobenius_error,
    static_response_from_gammas,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import supercell_pseudospin_harmonic_vertices
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    physical_symmetric_susceptibility,
    solve_vertex_q0,
)


_STAGE_ORDER = ("gg", "hf", "split-mt", "full")
_STAGE_LABEL = {
    "gg": "GG",
    "hf": "H+F",
    "split-mt": "H+F+MT",
    "full": "full(+AL)",
}
_PART_LABELS = ("H", "F", "MT", "AL1", "AL2")
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
    p.add_argument(
        "--benchmark",
        required=True,
        help="NPZ produced by benchmark_ed18_cgw.py on the same 18-site torus.",
    )
    p.add_argument("--nw", type=int, default=55)
    p.add_argument(
        "--nomega",
        type=int,
        default=None,
        help="Bosonic cutoff. If omitted, infer it from benchmark gg_m_values.",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["hf", "split-mt", "full", "all"],
        default=["all"],
        help="Vertex stages to solve. 'all' means H+F, H+F+MT and full.",
    )
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-workers", type=int, default=1)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Do not seed each stage from the previous stage's converged Gamma.",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _canonical_stages(raw):
    requested = set(str(x) for x in raw)
    if "all" in requested:
        requested = {"hf", "split-mt", "full"}
    stages = [s for s in _STAGE_ORDER if s == "gg" or s in requested]
    if "gg" not in stages:
        stages.insert(0, "gg")
    return stages


def _stage_options(stage, args):
    if stage == "hf":
        include_mt, include_al = False, False
    elif stage == "split-mt":
        include_mt, include_al = True, False
    elif stage == "full":
        include_mt, include_al = True, True
    else:
        raise ValueError(f"no vertex solve is needed for stage {stage!r}")
    return SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        mixing=args.vertex_mixing,
        solver=args.vertex_solver,
        gmres_restart=args.vertex_gmres_restart,
        include_hartree=True,
        include_fock=True,
        include_mt=include_mt,
        include_al=include_al,
        verbose=args.vertex_verbose,
        momentum_backend=args.momentum_backend,
    )


def _solve_stage(
    G,
    W,
    Vq,
    vertices,
    grid,
    args,
    stage,
    label,
    initial_gammas=None,
):
    """Solve one vertex stage, optionally warm-starting every driven channel."""
    vertices = np.asarray(vertices, dtype=complex)
    opts = _stage_options(stage, args)
    nvert = len(vertices)
    if initial_gammas is not None and len(initial_gammas) != nvert:
        raise ValueError("initial_gammas length mismatch")

    def solve_one(item):
        i, K = item
        initial = None if initial_gammas is None else initial_gammas[i]
        res = solve_vertex_q0(
            G,
            W,
            Vq,
            K,
            grid,
            opts=opts,
            initial_gamma=initial,
        )
        if not res.converged:
            raise RuntimeError(
                f"{label} {stage} vertex {i+1}/{nvert} failed: "
                f"residual={res.final_error:.3e}"
            )
        norms = np.asarray(
            [
                np.max(np.abs(res.Gamma_H)),
                np.max(np.abs(res.Gamma_F)),
                np.max(np.abs(res.Gamma_MT)),
                np.max(np.abs(res.Gamma_AL1)),
                np.max(np.abs(res.Gamma_AL2)),
            ],
            dtype=float,
        )
        return i, res.Gamma, int(res.iterations), float(res.final_error), norms

    print(
        f"  {label}: {_STAGE_LABEL[stage]}  vertices={nvert}  "
        f"workers={args.vertex_workers}  "
        f"warm_start={initial_gammas is not None}"
    )
    items = list(enumerate(vertices))
    if int(args.vertex_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(args.vertex_workers)) as pool:
            solved = list(pool.map(solve_one, items))
    else:
        solved = [solve_one(x) for x in items]
    solved.sort(key=lambda x: x[0])

    gammas = [x[1] for x in solved]
    iterations = np.asarray([x[2] for x in solved], dtype=int)
    residuals = np.asarray([x[3] for x in solved], dtype=float)
    part_norms = np.stack([x[4] for x in solved], axis=0)

    raw = static_response_from_gammas(G, vertices, gammas, grid)
    sym, imag_max = physical_symmetric_susceptibility(raw)
    print(
        f"    iterations: mean={np.mean(iterations):.1f}, "
        f"max={np.max(iterations)}, residual_max={np.max(residuals):.3e}"
    )
    return (
        np.asarray(sym, dtype=float),
        gammas,
        iterations,
        residuals,
        part_norms,
        float(imag_max),
    )


def _plot_stages(ed, stage_mats, channels, stage_names, out, dpi):
    labels = [_CHANNEL_TEX.get(str(ch), str(ch)) for ch in channels]
    xch = np.arange(len(channels), dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 11.0))
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    methods = [("ED", ed)] + [(_STAGE_LABEL[s], stage_mats[i]) for i, s in enumerate(stage_names)]

    for iq in range(2):
        ax = axes[0, iq]
        for name, mats in methods:
            ax.plot(xch, np.real(np.diag(mats[iq])), marker="o", label=name)
        ax.set_xticks(xch, labels, rotation=35, ha="right")
        ax.set_ylabel(r"diagonal $\chi_{\mu\mu}$")
        ax.set_title(qtitles[iq])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax = axes[1, iq]
        for name, mats in methods:
            vals, _ = eigh_desc_hermitian(mats[iq])
            ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=name)
        ax.set_xlabel("susceptibility eigenvalue index")
        ax.set_ylabel(r"$\lambda_\alpha(\chi)$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

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
        "same 18-site torus: where do vertex corrections change the leading mode?",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def _stage_metrics(ed, stage_mats, channels, stage_names):
    ns = len(stage_names)
    errors = np.zeros((ns, 2), dtype=float)
    lambdas = np.zeros((ns, 2), dtype=float)
    current = np.zeros((ns, 2), dtype=float)
    dimensions = np.zeros((ns, 2), dtype=int)
    zidx = [i for i, ch in enumerate(channels) if str(ch).startswith("z_")]
    for isg in range(ns):
        for iq in range(2):
            errors[isg, iq] = relative_frobenius_error(ed[iq], stage_mats[isg, iq])
            vals, _, w, d = _leading_weights(stage_mats[isg, iq])
            lambdas[isg, iq] = vals[0]
            current[isg, iq] = float(np.sum(w[zidx])) if zidx else 0.0
            dimensions[isg, iq] = d
    return errors, lambdas, current, dimensions


def _plot_metrics(ed, stage_mats, channels, stage_names, out, dpi):
    errors, lambdas, current, _ = _stage_metrics(ed, stage_mats, channels, stage_names)
    x = np.arange(len(stage_names), dtype=float)
    labels = [_STAGE_LABEL[s] for s in stage_names]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), sharex="col")
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]

    for iq in range(2):
        axes[0, iq].plot(x, errors[:, iq], marker="o")
        axes[0, iq].set_ylabel(r"$||\chi-\chi_{ED}||_F/||\chi_{ED}||_F$")
        axes[0, iq].set_title(qtitles[iq])
        axes[0, iq].grid(alpha=0.2)

        axes[1, iq].plot(x, lambdas[:, iq], marker="o", label="approximation")
        ed_lam, _, _, _, _ = _mode_summary(ed[iq], channels)
        axes[1, iq].axhline(ed_lam, linestyle="--", linewidth=1.0, label="ED")
        axes[1, iq].set_ylabel(r"leading $\lambda_{max}(\chi)$")
        axes[1, iq].grid(alpha=0.2)
        axes[1, iq].legend(fontsize=8)

        axes[2, iq].plot(x, current[:, iq], marker="o", label="approximation")
        _, ed_wc, _, _, _ = _mode_summary(ed[iq], channels)
        axes[2, iq].axhline(ed_wc, linestyle="--", linewidth=1.0, label="ED")
        axes[2, iq].set_ylabel("leading-mode current weight")
        axes[2, iq].set_ylim(-0.03, 1.03)
        axes[2, iq].set_xticks(x, labels, rotation=25, ha="right")
        axes[2, iq].grid(alpha=0.2)
        axes[2, iq].legend(fontsize=8)

    fig.suptitle("same-torus cGW stage diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def _plot_increments(stage_mats, channels, stage_names, out, dpi):
    labels = [_CHANNEL_TEX.get(str(ch), str(ch)) for ch in channels]
    x = np.arange(len(channels), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=False)
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    for iq in range(2):
        ax = axes[iq]
        for i in range(1, len(stage_names)):
            prev, cur = stage_names[i - 1], stage_names[i]
            delta = np.real(np.diag(stage_mats[i, iq] - stage_mats[i - 1, iq]))
            ax.plot(
                x,
                delta,
                marker="o",
                label=f"{_STAGE_LABEL[cur]} - {_STAGE_LABEL[prev]}",
            )
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel(r"increment $\Delta\chi_{\mu\mu}$")
        ax.set_title(qtitles[iq])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "incremental response after re-solving each enlarged vertex kernel",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def _load_benchmark(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = [
            "V", "filling", "temperature", "ti", "t1", "t2", "channels",
            "ed_static_chi", "gg_static_chi", "gg_m_values", "geometry",
        ]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"benchmark NPZ is missing keys: {missing}")
        out = {k: np.asarray(data[k]) for k in data.files}
    geometry = str(out["geometry"])
    if "18-site" not in geometry or "PBC" not in geometry:
        raise ValueError(f"benchmark geometry is not the required 18-site PBC torus: {geometry}")
    return path, out


def main():
    args = _args()
    if args.vertex_workers < 1:
        raise ValueError("--vertex-workers must be >=1")
    stage_names = _canonical_stages(args.stages)
    bench_path, base = _load_benchmark(args.benchmark)

    V = float(base["V"])
    filling = float(base["filling"])
    T = float(base["temperature"])
    ti, t1, t2 = float(base["ti"]), float(base["t1"]), float(base["t2"])
    channels = tuple(str(x) for x in np.asarray(base["channels"]).tolist())
    ed_static = np.asarray(base["ed_static_chi"], dtype=complex)
    gg_static = np.asarray(base["gg_static_chi"], dtype=complex)
    mvals = np.asarray(base["gg_m_values"], dtype=int)
    inferred_nomega = int((len(mvals) - 1) // 2)
    nomega = inferred_nomega if args.nomega is None else int(args.nomega)
    if nomega != inferred_nomega:
        raise ValueError(
            f"--nomega={nomega} does not match benchmark nOmega={inferred_nomega}"
        )

    # Populate fields expected by benchmark_ed18_cgw._checkpoint.
    args.V = V
    args.filling = filling
    args.T = T
    args.nk1 = 1
    args.nk2 = 1
    args.nomega = nomega

    print("=== staged same-torus ED/cGW decomposition ===")
    print("benchmark:", bench_path)
    print(f"V={V:g}, filling={filling:g}, T={T:g}, ti={ti:g}, t1={t1:g}, t2={t2:g}")
    print(f"channels: {', '.join(channels)}")
    print("stages:", " -> ".join(_STAGE_LABEL[s] for s in stage_names))
    print(
        "stage differences are re-solved-kernel increments, not isolated bare-diagram susceptibilities"
    )

    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=nomega, T=T)
    checkpoint, seed, meta, _ = _checkpoint(args, params, grid)
    print("SC-GW checkpoint:", checkpoint)
    h0, Vq, G, P, W, density, rH, rGW = _rebuild_scgw(
        seed, params, grid, args.momentum_backend
    )
    sc_res = max(rH, rGW)
    print(
        f"SC-GW fixed-point check: rH={rH:.3e}, rGW={rGW:.3e}, "
        f"n_primitive={np.sum(density)/3.0:.10f}"
    )
    if sc_res > args.max_scgw_residual:
        raise RuntimeError(
            f"checkpoint residual {sc_res:.3e} exceeds --max-scgw-residual"
        )

    KG, KQreal = _real_harmonic_vertices(channels)
    _, labels_G, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="q0")
    _, labels_C, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="Qc")
    _, labels_S, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="Qs")
    labels_Q = labels_C + labels_S

    # Stage arrays start from the already-computed GG response in the benchmark.
    stage_mats = [np.asarray(gg_static, dtype=complex)]
    solved_stage_names = ["gg"]
    real_Q_by_stage = []
    iterations_G = []
    residuals_G = []
    part_norms_G = []
    iterations_Q = []
    residuals_Q = []
    part_norms_Q = []
    imag_G_all = []
    imag_Q_all = []

    prev_G = None
    prev_Q = None
    for stage in [s for s in stage_names if s != "gg"]:
        print(f"\n--- stage {_STAGE_LABEL[stage]} ---")
        init_G = None if args.no_warm_start else prev_G
        init_Q = None if args.no_warm_start else prev_Q
        (
            chi_G,
            prev_G_new,
            itG,
            resG,
            normG,
            imagG,
        ) = _solve_stage(G, W, Vq, KG, grid, args, stage, "Gamma/q0", init_G)
        (
            chi_Q_real,
            prev_Q_new,
            itQ,
            resQ,
            normQ,
            imagQ,
        ) = _solve_stage(G, W, Vq, KQreal, grid, args, stage, "Q/QcQs", init_Q)
        if not args.no_warm_start:
            prev_G, prev_Q = prev_G_new, prev_Q_new

        chi_Q = project_complex_q(chi_Q_real, len(channels))
        chi_Q = 0.5 * (chi_Q + chi_Q.conj().T)
        stage_mats.append(
            np.stack([np.asarray(chi_G, dtype=complex), chi_Q], axis=0)
        )
        solved_stage_names.append(stage)
        real_Q_by_stage.append(np.asarray(chi_Q_real, dtype=float))
        iterations_G.append(itG)
        residuals_G.append(resG)
        part_norms_G.append(normG)
        iterations_Q.append(itQ)
        residuals_Q.append(resQ)
        part_norms_Q.append(normQ)
        imag_G_all.append(imagG)
        imag_Q_all.append(imagQ)

    stage_mats = np.stack(stage_mats, axis=0)
    stage_names = solved_stage_names
    errors, lambdas, current_weight, leading_dims = _stage_metrics(
        ed_static, stage_mats, channels, stage_names
    )

    print("\n=== stage summary ===")
    for iq, qname in enumerate(("Gamma", "Q")):
        ed_lam, ed_wc, ed_wo, ed_d, ed_text = _mode_summary(ed_static[iq], channels)
        print(
            f"{qname} ED: lambda={ed_lam:+.7e}, d={ed_d}, "
            f"current={ed_wc:.3f}, orbital={ed_wo:.3f}, {ed_text}"
        )
        for isg, stage in enumerate(stage_names):
            lam, wc, wo, d, text = _mode_summary(stage_mats[isg, iq], channels)
            print(
                f"  {_STAGE_LABEL[stage]:10s} lambda={lam:+.7e}, d={d}, "
                f"current={wc:.3f}, orbital={wo:.3f}, "
                f"err_ED={errors[isg,iq]:.4f}  {text}"
            )

    # Report final-stage decomposed Gamma-part norms.  These are norms of the
    # actual terms entering the final converged vertex equation, not response
    # increments.
    if part_norms_G:
        final_stage = stage_names[-1]
        final_G_norms = part_norms_G[-1]
        final_Q_norms = part_norms_Q[-1]
        print(f"\n=== {_STAGE_LABEL[final_stage]} vertex-part max norms ===")
        print("Gamma/q0:")
        print("  channel                 H          F         MT        AL1        AL2")
        for label, row in zip(labels_G, final_G_norms):
            print(f"  {label:20s} " + " ".join(f"{x:9.2e}" for x in row))
        print("Q real harmonics:")
        print("  channel                 H          F         MT        AL1        AL2")
        for label, row in zip(labels_Q, final_Q_norms):
            print(f"  {label:20s} " + " ".join(f"{x:9.2e}" for x in row))

    out = Path(args.out) if args.out is not None else bench_path.with_name(
        bench_path.stem + "_stages.npz"
    )
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ragged solve diagnostics are kept as separate Gamma/Q arrays.  GG has no
    # vertex solve and is therefore absent from these arrays.
    np.savez_compressed(
        out,
        source_benchmark=np.asarray(str(bench_path)),
        V=V,
        filling=filling,
        temperature=T,
        ti=ti,
        t1=t1,
        t2=t2,
        nw=int(args.nw),
        nomega=int(nomega),
        channels=np.asarray(channels),
        q_names=np.asarray(["Gamma", "Q"]),
        stage_names=np.asarray(stage_names),
        stage_labels=np.asarray([_STAGE_LABEL[s] for s in stage_names]),
        ed_static_chi=ed_static,
        stage_static_chi=stage_mats,
        stage_relative_error_to_ed=errors,
        stage_leading_lambda=lambdas,
        stage_leading_current_weight=current_weight,
        stage_leading_dimension=leading_dims,
        solved_vertex_stages=np.asarray(stage_names[1:]),
        cgw_Q_real_harmonic=np.asarray(real_Q_by_stage),
        vertex_iterations_Gamma=np.asarray(iterations_G, dtype=int),
        vertex_residuals_Gamma=np.asarray(residuals_G, dtype=float),
        vertex_part_norms_Gamma=np.asarray(part_norms_G, dtype=float),
        vertex_iterations_Q_real=np.asarray(iterations_Q, dtype=int),
        vertex_residuals_Q_real=np.asarray(residuals_Q, dtype=float),
        vertex_part_norms_Q_real=np.asarray(part_norms_Q, dtype=float),
        vertex_part_labels=np.asarray(_PART_LABELS),
        vertex_labels_Gamma=np.asarray(labels_G),
        vertex_labels_Q_real=np.asarray(labels_Q),
        discarded_imag_Gamma=np.asarray(imag_G_all, dtype=float),
        discarded_imag_Q_real=np.asarray(imag_Q_all, dtype=float),
        warm_start=np.asarray(not args.no_warm_start),
        vertex_workers=int(args.vertex_workers),
        scgw_checkpoint=np.asarray(str(checkpoint)),
        scgw_mu=float(seed.mu),
        scgw_density=np.asarray(density),
        scgw_residual_H=float(rH),
        scgw_residual_GW=float(rGW),
        increment_interpretation=np.asarray(
            "stage differences are re-solved-kernel increments; not isolated bare-diagram susceptibilities"
        ),
    )
    print("\nsaved:", out)

    stages_plot = out.with_name(out.stem + "_stages.png")
    metrics_plot = out.with_name(out.stem + "_metrics.png")
    increments_plot = out.with_name(out.stem + "_increments.png")
    _plot_stages(ed_static, stage_mats, channels, stage_names, stages_plot, args.dpi)
    _plot_metrics(ed_static, stage_mats, channels, stage_names, metrics_plot, args.dpi)
    _plot_increments(stage_mats, channels, stage_names, increments_plot, args.dpi)
    print("saved:", stages_plot)
    print("saved:", metrics_plot)
    print("saved:", increments_plot)

    print("\nRead the diagnostics in this order:")
    print("  1) metrics: at which stage does leading current weight jump?")
    print("  2) increments: which enlarged kernel changes each diagonal channel most?")
    print("  3) final vertex-part norms: which H/F/MT/AL term is numerically large in Gamma?")
    print("  4) remember that stage increments include re-summation of the enlarged kernel.")


if __name__ == "__main__":
    main()
