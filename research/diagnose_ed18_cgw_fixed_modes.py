#!/usr/bin/env python3
"""Diagnose the MT-driven mode switch with fixed physical source modes.

This is the follow-up to ``decompose_ed18_cgw_stages.py``.  Instead of letting
the leading eigenvector change at every cGW stage, it fixes three reference
modes from the exact finite-T ED susceptibility on the same 18-site PBC torus:

  * ``ED-leading``: the exact ED leading eigenvector;
  * ``ED-orbital``: the strongest ED mode restricted to x/y channels;
  * ``ED-current``: the strongest ED mode restricted to z channels.

For each of those fixed sources and for each vertex stage

    GG -> H+F -> H+F+MT -> full(+AL),

the script solves only the corresponding linear vertex equation.  This is much
cheaper than re-solving all 6 Gamma and 12 Qc/Qs basis sources at every stage.
It then reports the fixed-mode susceptibility and source-conditioned diagnostics
of the matrix-free equation

    (I-L) Gamma = K.

Important: ``sigma_source = ||(I-L)Gamma||/||Gamma||`` is only an UPPER BOUND
on the global ``sigma_min(I-L)``.  It answers how close the particular physical
source direction is to a pole.  It is not labeled as the exact global singular
value.  The Rayleigh quotient ``lambda_eff=<Gamma,L Gamma>/<Gamma,Gamma>`` is
shown together with its eigen-residual; only when that residual is small should
``lambda_eff`` be interpreted as an actual kernel eigenvalue.

Example
-------

    python diagnose_ed18_cgw_fixed_modes.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --stages all ^
      --vertex-workers 2 ^
      --out bench_n3_V1_T008_fixed_modes.npz
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_ed18_cgw import _checkpoint, _real_harmonic_vertices, _rebuild_scgw
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.response_mode_diagnostics import (
    combine_complex_q_vertex,
    combine_vertex,
    hermitian_leading_mode,
    mode_response,
    scalar_static_response,
    vertex_pole_diagnostics,
)
from rubycgw.supercell_cgw import SupercellVertexOptions, solve_vertex_q0


_STAGE_ORDER = ("gg", "hf", "split-mt", "full")
_STAGE_LABEL = {
    "gg": "GG",
    "hf": "H+F",
    "split-mt": "H+F+MT",
    "full": "full(+AL)",
}
_MODE_NAMES = ("ed_leading", "ed_orbital", "ed_current")
_MODE_LABELS = {
    "ed_leading": "ED-leading",
    "ed_orbital": "ED-orbital",
    "ed_current": "ED-current",
}
_PART_NAMES = ("H", "F", "MT", "AL1", "AL2")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--stages", nargs="+", choices=["hf", "split-mt", "full", "all"], default=["all"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--nw", type=int, default=None, help="Fermionic cutoff; default: infer from benchmark metadata when possible, else 55.")
    p.add_argument("--nomega", type=int, default=None, help="Bosonic cutoff; default: infer from benchmark gg_m_values.")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-workers", type=int, default=1)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--degeneracy-tol", type=float, default=1e-8)
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _canonical_stages(raw):
    req = set(str(x) for x in raw)
    if "all" in req:
        req = {"hf", "split-mt", "full"}
    return ["gg"] + [s for s in ("hf", "split-mt", "full") if s in req]


def _stage_options(stage, args):
    if stage == "hf":
        include_mt, include_al = False, False
    elif stage == "split-mt":
        include_mt, include_al = True, False
    elif stage == "full":
        include_mt, include_al = True, True
    else:
        raise ValueError(stage)
    return SupercellVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        mixing=float(args.vertex_mixing),
        solver=str(args.vertex_solver),
        gmres_restart=int(args.vertex_gmres_restart),
        include_hartree=True,
        include_fock=True,
        include_mt=include_mt,
        include_al=include_al,
        verbose=bool(args.vertex_verbose),
        momentum_backend=str(args.momentum_backend),
    )


def _leading_dimension(matrix, tol):
    vals = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))[::-1]
    scale = max(1.0, abs(float(vals[0].real)))
    return int(np.count_nonzero(np.abs(vals.real - vals[0].real) <= float(tol) * scale))


def _reference_modes(ed_static, channels, tol):
    channels = tuple(str(x) for x in channels)
    orb = [i for i, c in enumerate(channels) if c.startswith("x_") or c.startswith("y_")]
    cur = [i for i, c in enumerate(channels) if c.startswith("z_")]
    if not orb or not cur:
        raise ValueError("benchmark channels must contain both x/y orbital and z current sectors")

    coeff = np.zeros((2, len(_MODE_NAMES), len(channels)), dtype=complex)
    values = np.zeros((2, len(_MODE_NAMES)), dtype=float)
    leading_dim = np.zeros(2, dtype=int)
    for iq in range(2):
        mat = np.asarray(ed_static[iq], dtype=complex)
        leading_dim[iq] = _leading_dimension(mat, tol)
        values[iq, 0], coeff[iq, 0] = hermitian_leading_mode(mat)
        values[iq, 1], coeff[iq, 1] = hermitian_leading_mode(mat, orb)
        values[iq, 2], coeff[iq, 2] = hermitian_leading_mode(mat, cur)
    return coeff, values, leading_dim


def _make_source(iq, coeff, KG, KQreal):
    if int(iq) == 0:
        return combine_vertex(KG, coeff)
    return combine_complex_q_vertex(KQreal, coeff)


def _solve_one_mode(G, W, Vq, K, grid, opts, initial):
    res = solve_vertex_q0(G, W, Vq, K, grid, opts=opts, initial_gamma=initial)
    if not res.converged:
        raise RuntimeError(f"vertex solve failed: residual={res.final_error:.3e}")
    kernel = res.Gamma_H + res.Gamma_F + res.Gamma_MT + res.Gamma_AL1 + res.Gamma_AL2
    diag = vertex_pole_diagnostics(
        np.broadcast_to(K, G.shape),
        res.Gamma,
        kernel,
    )
    part = np.asarray(
        [
            np.linalg.norm(res.Gamma_H.ravel()),
            np.linalg.norm(res.Gamma_F.ravel()),
            np.linalg.norm(res.Gamma_MT.ravel()),
            np.linalg.norm(res.Gamma_AL1.ravel()),
            np.linalg.norm(res.Gamma_AL2.ravel()),
        ],
        dtype=float,
    ) / max(float(np.linalg.norm(res.Gamma.ravel())), 1e-300)
    chi = scalar_static_response(G, K, res.Gamma, grid.T, grid.nk)
    return res, diag, part, chi


def _plot_fixed(stage_names, chi, ed_values, out, dpi):
    x = np.arange(len(stage_names), dtype=float)
    labels = [_STAGE_LABEL[s] for s in stage_names]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), sharex="col")
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    for iq in range(2):
        ax = axes[0, iq]
        ax.plot(x, chi[:, iq, 0], marker="o", label="fixed ED-leading")
        ax.axhline(ed_values[iq, 0], linestyle="--", linewidth=1.0, label="ED exact")
        ax.set_ylabel(r"$\chi[v_{ED}]$")
        ax.set_title(qtitles[iq])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax = axes[1, iq]
        ax.plot(x, chi[:, iq, 1], marker="o", label="fixed ED-orbital")
        ax.plot(x, chi[:, iq, 2], marker="o", label="fixed ED-current")
        ax.axhline(ed_values[iq, 1], linestyle="--", linewidth=0.9, alpha=0.8, label="ED orbital")
        ax.axhline(ed_values[iq, 2], linestyle=":", linewidth=0.9, alpha=0.8, label="ED current")
        ax.set_ylabel("fixed-mode susceptibility")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ratio = chi[:, iq, 2] / np.where(np.abs(chi[:, iq, 1]) > 1e-14, chi[:, iq, 1], np.nan)
        ed_ratio = ed_values[iq, 2] / ed_values[iq, 1]
        ax = axes[2, iq]
        ax.plot(x, ratio, marker="o")
        ax.axhline(1.0, linewidth=0.8, alpha=0.5)
        ax.axhline(ed_ratio, linestyle="--", linewidth=1.0, label="ED ratio")
        ax.set_ylabel(r"$\chi_{current}/\chi_{orbital}$")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("same-torus fixed ED modes: where does current overtake orbital?", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def _plot_pole(stage_names, gain, sigma, lam, eigres, eqres, out, dpi):
    x = np.arange(len(stage_names), dtype=float)
    labels = [_STAGE_LABEL[s] for s in stage_names]
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12.0), sharex="col")
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    modes = [(1, "ED-orbital"), (2, "ED-current")]
    for iq in range(2):
        for im, label in modes:
            axes[0, iq].plot(x, gain[:, iq, im], marker="o", label=label)
            axes[1, iq].plot(x, sigma[:, iq, im], marker="o", label=label)
            axes[2, iq].plot(x, np.real(lam[:, iq, im]), marker="o", label=label)
            axes[3, iq].plot(x, eigres[:, iq, im], marker="o", label=f"eig-res {label}")
            axes[3, iq].plot(x, eqres[:, iq, im], marker="x", linestyle="--", label=f"eq-res {label}")
        axes[0, iq].set_title(qtitles[iq])
        axes[0, iq].set_ylabel(r"source gain $||\Gamma||/||K||$")
        axes[1, iq].set_ylabel(r"$|| (I-L)\Gamma||/||\Gamma||$\n(upper bound on $\sigma_{min}$)")
        axes[2, iq].set_ylabel(r"Re $\lambda_{eff}$")
        axes[2, iq].axhline(1.0, linestyle="--", linewidth=0.9)
        axes[3, iq].set_ylabel("kernel/equation residual")
        axes[3, iq].set_yscale("log")
        axes[3, iq].set_xticks(x, labels, rotation=25, ha="right")
        for row in range(4):
            axes[row, iq].grid(alpha=0.2)
            axes[row, iq].legend(fontsize=7)
    fig.suptitle("source-conditioned cGW pole diagnostics (not a global SVD)", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    bpath = Path(args.benchmark)
    data = np.load(bpath, allow_pickle=False)
    required = ("V", "filling", "temperature", "ti", "t1", "t2", "channels", "ed_static_chi", "gg_static_chi")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"benchmark NPZ is missing keys: {missing}")

    args.V = float(data["V"])
    args.filling = float(data["filling"])
    args.T = float(data["temperature"])
    args.ti = float(data["ti"])
    args.t1 = float(data["t1"])
    args.t2 = float(data["t2"])
    args.nk1 = 1
    args.nk2 = 1
    if args.nw is None:
        args.nw = int(data["nw"]) if "nw" in data else 55
    if args.nomega is None:
        if "gg_m_values" not in data:
            raise ValueError("cannot infer --nomega: benchmark has no gg_m_values")
        args.nomega = int(np.max(np.abs(np.asarray(data["gg_m_values"], dtype=int))))

    channels = tuple(str(x) for x in np.asarray(data["channels"]).tolist())
    ed_static = np.asarray(data["ed_static_chi"], dtype=complex)
    gg_static = np.asarray(data["gg_static_chi"], dtype=complex)
    if ed_static.shape != (2, len(channels), len(channels)):
        raise ValueError("unexpected ed_static_chi shape")

    coeff, ed_values, leading_dim = _reference_modes(ed_static, channels, args.degeneracy_tol)
    for iq, qname in enumerate(("Gamma", "Q")):
        if leading_dim[iq] > 1:
            print(
                f"WARNING: ED leading eigenspace at {qname} has dimension d={leading_dim[iq]}; "
                "the ED-leading pole diagnostic uses one arbitrary normalized basis vector. "
                "The separately defined orbital/current reference modes remain unambiguous."
            )

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    checkpoint, seed, _, _ = _checkpoint(args, params, grid)
    _, Vq, G, _, W, density, rH, rGW = _rebuild_scgw(seed, params, grid, args.momentum_backend)
    sc_res = max(rH, rGW)
    if sc_res > args.max_scgw_residual:
        raise RuntimeError(f"SC-GW checkpoint residual {sc_res:.3e} exceeds tolerance")
    KG, KQreal = _real_harmonic_vertices(channels)

    stages = _canonical_stages(args.stages)
    ns, nq, nm = len(stages), 2, len(_MODE_NAMES)
    chi = np.zeros((ns, nq, nm), dtype=float)
    gain = np.ones((ns, nq, nm), dtype=float)
    sigma = np.ones((ns, nq, nm), dtype=float)
    lam = np.zeros((ns, nq, nm), dtype=complex)
    eigres = np.zeros((ns, nq, nm), dtype=float)
    eqres = np.zeros((ns, nq, nm), dtype=float)
    corr_ratio = np.zeros((ns, nq, nm), dtype=float)
    alignment = np.ones((ns, nq, nm), dtype=float)
    iterations = np.zeros((ns, nq, nm), dtype=int)
    solver_residual = np.zeros((ns, nq, nm), dtype=float)
    part_norms = np.zeros((ns, nq, nm, len(_PART_NAMES)), dtype=float)

    sources = [[_make_source(iq, coeff[iq, im], KG, KQreal) for im in range(nm)] for iq in range(nq)]
    previous = [[None for _ in range(nm)] for _ in range(nq)]

    # GG stage: Gamma=K and L=0.
    for iq in range(nq):
        for im in range(nm):
            K = sources[iq][im]
            Gamma = np.broadcast_to(K, G.shape).copy()
            chi[0, iq, im] = scalar_static_response(G, K, Gamma, grid.T, grid.nk)
            previous[iq][im] = Gamma
            matrix_value = mode_response(gg_static[iq], coeff[iq, im])
            rel = abs(chi[0, iq, im] - matrix_value) / max(abs(matrix_value), 1e-12)
            if rel > 5e-6:
                print(f"WARNING: GG fixed-mode direct/matrix mismatch at q={iq}, mode={_MODE_NAMES[im]}: rel={rel:.3e}")

    for isg, stage in enumerate(stages[1:], start=1):
        opts = _stage_options(stage, args)
        print(f"\n=== {_STAGE_LABEL[stage]} ===")
        tasks = [(iq, im) for iq in range(nq) for im in range(nm)]

        def run(task):
            iq, im = task
            initial = None if args.no_warm_start else previous[iq][im]
            res, diag, parts, response = _solve_one_mode(
                G, W, Vq, sources[iq][im], grid, opts, initial
            )
            return iq, im, res, diag, parts, response

        if int(args.vertex_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(args.vertex_workers)) as pool:
                solved = list(pool.map(run, tasks))
        else:
            solved = [run(x) for x in tasks]

        for iq, im, res, diag, parts, response in solved:
            previous[iq][im] = res.Gamma
            chi[isg, iq, im] = response
            gain[isg, iq, im] = diag.source_gain
            sigma[isg, iq, im] = diag.sigma_source_upper
            lam[isg, iq, im] = diag.lambda_eff
            eigres[isg, iq, im] = diag.lambda_eigen_residual
            eqres[isg, iq, im] = diag.equation_residual_relative
            corr_ratio[isg, iq, im] = diag.correction_ratio
            alignment[isg, iq, im] = diag.source_alignment
            iterations[isg, iq, im] = int(res.iterations)
            solver_residual[isg, iq, im] = float(res.final_error)
            part_norms[isg, iq, im] = parts

        for iq, qname in enumerate(("Gamma", "Q")):
            print(qname)
            for im, mname in enumerate(_MODE_NAMES):
                print(
                    f"  {_MODE_LABELS[mname]:10s} chi={chi[isg,iq,im]:+.6e}  "
                    f"gain={gain[isg,iq,im]:.4f}  sigma_src<={sigma[isg,iq,im]:.4e}  "
                    f"lambda_eff={lam[isg,iq,im].real:+.4f}{lam[isg,iq,im].imag:+.4f}i  "
                    f"eig_res={eigres[isg,iq,im]:.3e}"
                )

    print("\n=== fixed-mode crossing summary ===")
    for iq, qname in enumerate(("Gamma", "Q")):
        print(qname)
        for isg, stage in enumerate(stages):
            ratio = chi[isg, iq, 2] / chi[isg, iq, 1] if abs(chi[isg, iq, 1]) > 1e-14 else np.nan
            print(
                f"  {_STAGE_LABEL[stage]:10s} orbital={chi[isg,iq,1]:+.6e}  "
                f"current={chi[isg,iq,2]:+.6e}  current/orbital={ratio:.4f}"
            )

    out = Path(args.out) if args.out else bpath.with_name(bpath.stem + "_fixed_modes.npz")
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        benchmark=np.asarray(str(bpath)),
        checkpoint=np.asarray(str(checkpoint)),
        V=float(args.V),
        filling=float(args.filling),
        temperature=float(args.T),
        nw=int(args.nw),
        nomega=int(args.nomega),
        channels=np.asarray(channels),
        q_names=np.asarray(["Gamma", "Q"]),
        stage_names=np.asarray(stages),
        stage_labels=np.asarray([_STAGE_LABEL[s] for s in stages]),
        mode_names=np.asarray(_MODE_NAMES),
        mode_coefficients=coeff,
        ed_reference_response=ed_values,
        ed_leading_dimension=leading_dim,
        fixed_mode_chi=chi,
        source_gain=gain,
        sigma_source_upper=sigma,
        lambda_eff=lam,
        lambda_eigen_residual=eigres,
        equation_residual_relative=eqres,
        correction_ratio=corr_ratio,
        source_alignment=alignment,
        iterations=iterations,
        solver_residual=solver_residual,
        part_names=np.asarray(_PART_NAMES),
        part_norm_over_gamma=part_norms,
        scgw_density=np.asarray(density),
        scgw_residual_H=float(rH),
        scgw_residual_GW=float(rGW),
        diagnostic_scope=np.asarray(
            "sigma_source_upper is a physical-source upper bound on global sigma_min(I-L), not the exact global singular value"
        ),
    )
    print("saved:", out)

    fixed_plot = out.with_name(out.stem + "_fixed_response.png")
    pole_plot = out.with_name(out.stem + "_pole_proxy.png")
    _plot_fixed(stages, chi, ed_values, fixed_plot, args.dpi)
    _plot_pole(stages, gain, sigma, lam, eigres, eqres, pole_plot, args.dpi)
    print("saved:", fixed_plot)
    print("saved:", pole_plot)
    print(
        "\nInterpretation: a drop in sigma_source_upper or rise in source gain after MT, "
        "together with Re(lambda_eff)->1 and a small eigen-residual, is evidence that MT "
        "softens that physical mode.  If the eigen-residual is large, describe it instead "
        "as source amplification/non-normal mixing rather than a clean kernel eigenmode."
    )


if __name__ == "__main__":
    main()
