#!/usr/bin/env python3
"""Scan normal-state cGW loop-current curvatures versus interaction V.

The scan follows one zero-source, symmetry-preserving 18-site SC-GW branch at
fixed primitive-cell filling and evaluates the cGW current response at every V.

Project conventions:

    r_+ = physical-opposite uniform q=0 loop-current curvature
    r_- = physical-same     uniform q=0 loop-current curvature

The finite-Q minimum curvature is also saved as ``r_Q_min``.  The V grid is
constructed automatically as ``np.linspace(V_start, V_stop, V_num)``.

A true cold start is allowed.  The first V uses the existing weak-V SC-GW
bootstrap when no ``--start-checkpoint`` is supplied; later V points use the
previous converged normal GW state as a continuation seed.  Every converged
normal state is checkpointed in a scan-specific directory, by default

    results/supercell18/r_vs_V/<stage_timestamp>/normal_checkpoints/

which is deliberately separate from branch-search checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.checkpoint import (
    checkpoint_filename,
    load_supercell_checkpoint,
    read_checkpoint_metadata,
    save_supercell_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.lc_branch import current_diagnostics
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_interaction, charge_order_diagnostics
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    curvature_from_susceptibility,
    solve_vertex_q0,
    supercell_current_vertices,
    susceptibility_matrix_q0,
)
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_supercell_gw_anderson


Q_HARMONIC_INDICES = np.array([1, 2, 4, 5], dtype=int)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primitive-filling", type=float, default=3.0)
    p.add_argument(
        "--V-start",
        type=float,
        required=True,
        help="First interaction value in the scan.",
    )
    p.add_argument(
        "--V-stop",
        type=float,
        required=True,
        help="Last interaction value in the scan (included).",
    )
    p.add_argument(
        "--V-num",
        type=int,
        required=True,
        help="Number of equally spaced V points, including both endpoints.",
    )
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nk1", type=int, default=3)
    p.add_argument("--nk2", type=int, default=3)
    p.add_argument("--nw", type=int, default=47)
    p.add_argument("--nomega", type=int, default=10)

    p.add_argument(
        "--start-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional compatible zero-source GW warm start. If omitted, the first V "
            "is obtained by the built-in weak-V cold-start bootstrap."
        ),
    )
    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--mu-tol", type=float, default=5e-12)
    p.add_argument("--mu-max-iter", type=int, default=60)
    p.add_argument("--gw-verbose", action="store_true")

    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="split-mt",
        help="cGW response level: gg, split-mt=H+F+MT(W-V), or full=split-mt+AL1+AL2.",
    )
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument(
        "--vertex-mixing",
        type=float,
        default=0.25,
        help="Used only by --vertex-solver linear; ignored by GMRES.",
    )
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")

    p.add_argument(
        "--normal-threshold",
        type=float,
        default=1e-6,
        help=(
            "Abort if the continued background develops charge order or a uniform "
            "loop current larger than this per primitive cell."
        ),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: results/supercell18/r_vs_V/<stage_timestamp>. "
            "Normal GW checkpoints are stored in its normal_checkpoints subdirectory."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional override for this scan's normal GW checkpoint directory.",
    )
    return p.parse_args()


def _prepare_v_values(V_start: float, V_stop: float, V_num: int) -> list[float]:
    start = float(V_start)
    stop = float(V_stop)
    num = int(V_num)
    if not (np.isfinite(start) and np.isfinite(stop)):
        raise ValueError("--V-start and --V-stop must be finite")
    if start < 0.0 or stop < 0.0:
        raise ValueError("V values must be nonnegative")
    if stop < start:
        raise ValueError("Require --V-stop >= --V-start for ascending continuation")
    if num < 1:
        raise ValueError("--V-num must be at least 1")
    if num == 1:
        if not np.isclose(start, stop):
            raise ValueError("With --V-num 1, require --V-start == --V-stop")
        return [start]
    return [float(x) for x in np.linspace(start, stop, num)]


def _gw_options(args, mu0: float, target_N: float) -> GWOptions:
    return GWOptions(
        mu=float(mu0),
        target_filling=float(target_N),
        max_iter=int(args.gw_max_iter),
        tol=float(args.gw_tol),
        mixing=0.20,
        mixing_method="linear",
        pulay_history=6,
        pulay_start=3,
        pulay_regularization=1e-10,
        mu_tol=float(args.mu_tol),
        mu_max_iter=int(args.mu_max_iter),
        verbose=bool(args.gw_verbose),
        momentum_backend=str(args.momentum_backend),
    )


def _normal_diagnostics(gw, grid: MatsubaraGrid) -> dict[str, float]:
    charge = charge_order_diagnostics(np.asarray(gw.density, dtype=float))
    currents = current_diagnostics(gw.G, grid)
    m_same_pc = currents["same_q0"] / np.sqrt(3.0)
    m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
    return {
        "Phi_abs": float(abs(complex(charge["Phi"]))),
        "Delta_Q": float(charge["Delta_Q"]),
        "Delta_translation_rms": float(charge["Delta_translation_rms"]),
        "Delta_intra": float(charge["Delta_intra"]),
        "Delta_A": float(charge["Delta_A"]),
        "Delta_B": float(charge["Delta_B"]),
        "Delta_AB": float(charge["Delta_AB"]),
        "m_plus_pc_abs": float(abs(m_opp_pc)),
        "m_minus_pc_abs": float(abs(m_same_pc)),
    }


def _normal_breaking_scale(diag: dict[str, float]) -> float:
    return max(
        abs(float(diag["Delta_Q"])),
        abs(float(diag["Delta_intra"])),
        abs(float(diag["Delta_AB"])),
        abs(float(diag["m_plus_pc_abs"])),
        abs(float(diag["m_minus_pc_abs"])),
    )


def _solve_cgw_response(gw, Vq: np.ndarray, grid: MatsubaraGrid, args):
    Klocal, _ = supercell_current_vertices()
    bare = [np.broadcast_to(K, gw.G.shape).copy() for K in Klocal]

    if args.stage == "gg":
        gammas = bare
        residual_max = 0.0
        total_vertex_iterations = 0
    else:
        vopts = SupercellVertexOptions(
            max_iter=int(args.vertex_max_iter),
            tol=float(args.vertex_tol),
            mixing=float(args.vertex_mixing),
            solver=str(args.vertex_solver),
            gmres_restart=int(args.vertex_gmres_restart),
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=bool(args.stage == "full"),
            verbose=bool(args.vertex_verbose),
            momentum_backend=str(args.momentum_backend),
        )
        gammas = []
        residuals: list[float] = []
        total_vertex_iterations = 0
        for ich, K in enumerate(Klocal, start=1):
            result = solve_vertex_q0(gw.G, gw.W, Vq, K, grid, opts=vopts)
            if not result.converged:
                raise RuntimeError(
                    f"cGW vertex {ich}/6 failed: solver={result.solver}, "
                    f"iterations={result.iterations}, residual={result.final_error:.3e}"
                )
            gammas.append(result.Gamma)
            residuals.append(float(result.final_error))
            total_vertex_iterations += int(result.iterations)
        residual_max = max(residuals) if residuals else 0.0

    chi = susceptibility_matrix_q0(gw.G, Klocal, gammas, grid)
    analysis = curvature_from_susceptibility(chi)
    return analysis, float(residual_max), int(total_vertex_iterations)


def _curvature_fields(analysis: dict) -> dict[str, float]:
    Ru = np.asarray(analysis["R_uniform_relaxed"], dtype=float)
    Rc = np.asarray(analysis["R_uniform_constrained"], dtype=float)
    Rh = np.asarray(analysis["R_harmonic"], dtype=float)
    RQ = Rh[np.ix_(Q_HARMONIC_INDICES, Q_HARMONIC_INDICES)]
    RQ = 0.5 * (RQ + RQ.T)
    q_eigs = np.linalg.eigvalsh(RQ)
    return {
        "r_plus": float(Ru[0, 0]),
        "r_minus": float(Ru[1, 1]),
        "r_plusminus": float(Ru[0, 1]),
        "r_plus_constrained": float(Rc[0, 0]),
        "r_minus_constrained": float(Rc[1, 1]),
        "r_plusminus_constrained": float(Rc[0, 1]),
        "r_Q_min": float(q_eigs[0]),
        "r_soft_full": float(np.asarray(analysis["R_eigenvalues"], dtype=float)[0]),
        "soft_weight_opposite": float(analysis["soft_weight_opposite"]),
        "soft_weight_same": float(analysis["soft_weight_same"]),
        "soft_weight_q0": float(analysis["soft_weight_q0"]),
        "soft_weight_Q": float(analysis["soft_weight_Q"]),
        "chi_imag_max": float(analysis["chi_imag_max"]),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _zero_crossings(V: np.ndarray, y: np.ndarray) -> list[tuple[float, float, float]]:
    """Return (V_left,V_right,V_linear) for adjacent sign changes."""
    V = np.asarray(V, dtype=float)
    y = np.asarray(y, dtype=float)
    out: list[tuple[float, float, float]] = []
    for i in range(len(V) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            out.append((float(V[i]), float(V[i]), float(V[i])))
            continue
        if y0 * y1 < 0.0:
            v0, v1 = float(V[i]), float(V[i + 1])
            vc = v0 - y0 * (v1 - v0) / (y1 - y0)
            out.append((v0, v1, float(vc)))
    if len(V) and np.isfinite(y[-1]) and float(y[-1]) == 0.0:
        out.append((float(V[-1]), float(V[-1]), float(V[-1])))
    return out


def _write_crossings(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    V = np.asarray([r["V"] for r in rows], dtype=float)
    data = []
    for channel, key in (
        ("r_plus", "r_plus"),
        ("r_minus", "r_minus"),
        ("r_Q_min", "r_Q_min"),
    ):
        y = np.asarray([r[key] for r in rows], dtype=float)
        for v0, v1, vc in _zero_crossings(V, y):
            data.append(
                {
                    "channel": channel,
                    "V_left": v0,
                    "V_right": v1,
                    "V_linear_estimate": vc,
                }
            )
    if data:
        _write_rows(path, data)


def _plot_rows(outdir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    V = np.asarray([r["V"] for r in rows], dtype=float)
    rp = np.asarray([r["r_plus"] for r in rows], dtype=float)
    rm = np.asarray([r["r_minus"] for r in rows], dtype=float)
    rq = np.asarray([r["r_Q_min"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(V, rp, marker="o", label=r"$r_+$ (physical opposite, $q=0$)")
    ax.plot(V, rm, marker="s", label=r"$r_-$ (physical same, $q=0$)")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$r$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "r_plus_minus_vs_V.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(V, rp, marker="o", label=r"$r_+$, $q=0$")
    ax.plot(V, rm, marker="s", label=r"$r_-$, $q=0$")
    ax.plot(V, rq, marker="^", linestyle="--", label=r"$r_{Q,\min}$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$r$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "r_plus_minus_Q_vs_V.png", dpi=220)
    plt.close(fig)


def main():
    args = _parse_args()
    V_values = _prepare_v_values(args.V_start, args.V_stop, args.V_num)
    if args.primitive_filling <= 0.0:
        raise ValueError("--primitive-filling must be positive")
    if args.gw_tol <= 0.0 or args.vertex_tol <= 0.0:
        raise ValueError("GW and vertex tolerances must be positive")
    if args.normal_threshold <= 0.0:
        raise ValueError("--normal-threshold must be positive")
    if args.vertex_max_iter < 1 or args.vertex_gmres_restart < 1:
        raise ValueError("vertex iteration limits must be positive")

    grid = MatsubaraGrid(
        nk1=int(args.nk1),
        nk2=int(args.nk2),
        nw=int(args.nw),
        nOmega=int(args.nomega),
        T=float(args.T),
    )
    target_N = 3.0 * float(args.primitive_filling)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.outdir is None:
        outdir = Path("results") / "supercell18" / "r_vs_V" / f"{args.stage}_{stamp}"
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_dir is None:
        checkpoint_dir = outdir / "normal_checkpoints"
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("Fixed-filling normal-state cGW curvature scan")
    print(
        f"primitive filling={args.primitive_filling:g}, T={grid.T:g}, "
        f"grid={grid.nk1}x{grid.nk2}, nw={grid.nw}, nOmega={grid.nOmega}, stage={args.stage}"
    )
    print(
        f"V grid: linspace({args.V_start:g}, {args.V_stop:g}, {args.V_num}) "
        f"with dV={(V_values[1]-V_values[0]) if len(V_values)>1 else 0.0:g}"
    )
    print("V schedule:", " -> ".join(f"{v:g}" for v in V_values))
    print("output:", outdir)
    print("normal GW checkpoints:", checkpoint_dir)
    print("(This directory is separate from results/supercell18/branch_search/...)")
    print("=" * 110)

    previous = None
    if args.start_checkpoint is not None:
        start_path = Path(args.start_checkpoint)
        meta = read_checkpoint_metadata(start_path)
        if not bool(meta.get("converged", False)):
            raise ValueError("--start-checkpoint is not marked converged")
        if not np.isclose(float(meta.get("source", np.nan)), 0.0, atol=1e-14):
            raise ValueError("--start-checkpoint must be a zero-source GW state")
        p0 = RubyParameters(
            ti=float(args.ti), t1=float(args.t1), t2=float(args.t2), V=V_values[0]
        )
        previous, _, _ = load_supercell_checkpoint(
            start_path, p0, grid, float(args.primitive_filling)
        )
        print(f"warm start: {start_path} (checkpoint V={float(meta['V']):g})")
    else:
        print("warm start: none; first V will use the built-in weak-V bootstrap")

    rows: list[dict] = []
    csv_path = outdir / "r_vs_V.csv"

    for iv, V in enumerate(V_values, start=1):
        params = RubyParameters(
            ti=float(args.ti),
            t1=float(args.t1),
            t2=float(args.t2),
            V=float(V),
        )
        mu0 = float(previous.mu) if previous is not None else 0.0
        opts = _gw_options(args, mu0, target_N)

        print(f"\n[{iv}/{len(V_values)}] V={V:g}: normal zero-source SC-GW")
        print("  seed:", "previous V" if previous is not None else "cold/bootstrap")
        t0 = time.perf_counter()
        gw = solve_supercell_gw_anderson(
            params,
            grid,
            opts=opts,
            source_strength=0.0,
            initial=previous,
            anderson=AndersonOptions(),
        )
        gw_runtime = time.perf_counter() - t0
        print(
            f"  GW converged={gw.converged}, it={gw.iterations}, "
            f"res={gw.final_error:.3e}, mu={gw.mu:.10f}, time={gw_runtime:.1f}s"
        )
        if not gw.converged:
            raise RuntimeError(
                f"normal SC-GW failed at V={V:g}; residual={gw.final_error:.3e}. "
                "Do not compute cGW curvature from a nonconverged background."
            )

        normal = _normal_diagnostics(gw, grid)
        breaking = _normal_breaking_scale(normal)
        print(
            f"  normal diagnostics: Delta_Q={normal['Delta_Q']:.3e}, "
            f"Delta_intra={normal['Delta_intra']:.3e}, Delta_AB={normal['Delta_AB']:+.3e}, "
            f"|m+|/pc={normal['m_plus_pc_abs']:.3e}, |m-|/pc={normal['m_minus_pc_abs']:.3e}"
        )
        if breaking > float(args.normal_threshold):
            raise RuntimeError(
                f"continued background at V={V:g} is no longer symmetry-preserving: "
                f"max order diagnostic={breaking:.3e} > threshold={args.normal_threshold:.3e}. "
                "The requested r_+(V), r_-(V) curve is defined on the normal branch."
            )

        ckpt = checkpoint_dir / checkpoint_filename(
            float(V), float(args.primitive_filling), grid
        )
        save_supercell_checkpoint(
            ckpt,
            gw,
            params,
            grid,
            float(args.primitive_filling),
            source=0.0,
        )
        print("  checkpoint:", ckpt)

        Vq = build_supercell_interaction(grid.qmesh(), params)
        print(
            f"  cGW response: stage={args.stage}, solver={args.vertex_solver}, "
            f"tol={args.vertex_tol:.1e}"
        )
        t1 = time.perf_counter()
        analysis, vertex_residual, vertex_iterations = _solve_cgw_response(
            gw, Vq, grid, args
        )
        cgw_runtime = time.perf_counter() - t1
        curv = _curvature_fields(analysis)
        print(
            f"  r+={curv['r_plus']:+.10e}, r-={curv['r_minus']:+.10e}, "
            f"rQmin={curv['r_Q_min']:+.10e}, mix={curv['r_plusminus']:+.3e}"
        )
        print(
            f"  cGW time={cgw_runtime:.1f}s, max vertex residual={vertex_residual:.3e}, "
            f"total vertex iterations={vertex_iterations}"
        )

        row = {
            "V": float(V),
            "primitive_filling": float(args.primitive_filling),
            "T": float(grid.T),
            "stage": str(args.stage),
            "mu": float(gw.mu),
            "actual_primitive_filling": float(np.sum(gw.density) / 3.0),
            "gw_converged": bool(gw.converged),
            "gw_iterations": int(gw.iterations),
            "gw_residual": float(gw.final_error),
            "gw_runtime_s": float(gw_runtime),
            "vertex_solver": str(args.vertex_solver),
            "vertex_iterations_total": int(vertex_iterations),
            "vertex_residual_max": float(vertex_residual),
            "cgw_runtime_s": float(cgw_runtime),
            **normal,
            **curv,
            "checkpoint": str(ckpt),
        }
        rows.append(row)
        _write_rows(csv_path, rows)
        _write_crossings(outdir / "critical_V_estimates.csv", rows)
        _plot_rows(outdir, rows)

        previous = gw

    print("\n" + "=" * 110)
    print("SCAN COMPLETE")
    print("=" * 110)
    print("table:", csv_path)
    print("r+/r- plot:", outdir / "r_plus_minus_vs_V.png")
    print("diagnostic r+/r-/rQ plot:", outdir / "r_plus_minus_Q_vs_V.png")
    crossings = outdir / "critical_V_estimates.csv"
    if crossings.exists():
        print("linear zero-crossing estimates:", crossings)
    else:
        print("linear zero-crossing estimates: none bracketed by the supplied V grid")
    print("normal GW checkpoints:", checkpoint_dir)


if __name__ == "__main__":
    main()
