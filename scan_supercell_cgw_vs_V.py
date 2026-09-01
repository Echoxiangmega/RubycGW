#!/usr/bin/env python3
"""Scan normal-state cGW loop-current curvatures versus interaction V.

The scan follows one zero-source, symmetry-preserving 18-site SC-GW branch at
fixed primitive-cell filling and evaluates the cGW current response at every
requested V.

Project conventions:

    r_+ = physical-opposite uniform q=0 loop-current curvature
    r_- = physical-same     uniform q=0 loop-current curvature

The finite-Q minimum curvature is also saved as ``r_Q_min``.  The requested V
grid is constructed automatically as ``np.linspace(V_start, V_stop, V_num)``.

A true cold start is allowed.  The first requested V uses the existing weak-V
SC-GW bootstrap when no ``--start-checkpoint`` is supplied; later requested V
points use the previous converged normal GW state as a continuation seed.

If a direct V-continuation step fails, the driver first retries the same V from
the best finite state with a fresh Pulay history.  If that still fails and a
previous converged V is available, it recursively inserts midpoint bridge
couplings until the target converges or the configured bridge depth/minimum
step is exhausted.  Bridge points are used only to continue the normal GW
branch: cGW response is evaluated only on the user-requested linspace grid.

Every converged requested normal state is checkpointed in a scan-specific
directory, by default

    results/supercell18/r_vs_V/<stage_timestamp>/normal_checkpoints/

and automatically inserted bridge checkpoints go into

    .../normal_checkpoints/bridges/

Both are deliberately separate from branch-search checkpoints.
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
        help="First interaction value in the requested scan.",
    )
    p.add_argument(
        "--V-stop",
        type=float,
        required=True,
        help="Last interaction value in the requested scan (included).",
    )
    p.add_argument(
        "--V-num",
        type=int,
        required=True,
        help="Number of equally spaced requested V points, including both endpoints.",
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
            "Optional compatible converged zero-source normal-GW warm start. Its V "
            "must not exceed the first requested V. If it equals the first requested "
            "V, that GW state is reused directly and only cGW is recomputed."
        ),
    )
    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument(
        "--gw-retry-max-iter",
        type=int,
        default=250,
        help=(
            "Maximum iterations for the same-V fresh-history retry after a failed "
            "normal-GW continuation attempt."
        ),
    )
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--mu-tol", type=float, default=5e-12)
    p.add_argument("--mu-max-iter", type=int, default=60)
    p.add_argument("--gw-verbose", action="store_true")
    p.add_argument(
        "--bridge-min-step",
        type=float,
        default=0.005,
        help=(
            "Smallest allowed distance from a converged left point to an inserted "
            "midpoint bridge. Bridge points are not included in r(V) output."
        ),
    )
    p.add_argument(
        "--bridge-max-depth",
        type=int,
        default=6,
        help="Maximum recursive midpoint subdivision depth for one requested V step.",
    )
    p.add_argument(
        "--no-auto-bridge",
        action="store_true",
        help="Disable automatic midpoint bridge insertion after a failed same-V retry.",
    )

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
            "Reject a continued background if charge order or a uniform loop current "
            "exceeds this per primitive cell. Bridge points obey the same test."
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


def _gw_options(
    args,
    mu0: float,
    target_N: float,
    max_iter: int | None = None,
) -> GWOptions:
    return GWOptions(
        mu=float(mu0),
        target_filling=float(target_N),
        max_iter=int(args.gw_max_iter if max_iter is None else max_iter),
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


def _finite_gw_state(gw) -> bool:
    if gw is None:
        return False
    try:
        return bool(
            np.isfinite(float(gw.final_error))
            and np.isfinite(float(gw.mu))
            and np.all(np.isfinite(gw.Sigma_H))
            and np.all(np.isfinite(gw.Sigma_GW))
        )
    except Exception:
        return False


def _bridge_midpoint(
    left_V: float | None,
    right_V: float,
    min_step: float,
    depth: int,
    max_depth: int,
) -> float | None:
    """Return a legal midpoint bridge, or None when subdivision must stop."""
    if left_V is None or int(depth) >= int(max_depth):
        return None
    left = float(left_V)
    right = float(right_V)
    if right <= left:
        return None
    half_step = 0.5 * (right - left)
    if half_step < float(min_step) - 1e-15:
        return None
    return left + half_step


def _bridge_checkpoint_name(
    V: float,
    primitive_filling: float,
    grid: MatsubaraGrid,
) -> str:
    base = checkpoint_filename(V, primitive_filling, grid)
    stem = base[:-4] if base.endswith(".npz") else base
    return f"{stem}_bridge.npz"


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_normal_gw_attempt(
    args,
    V: float,
    requested_V: float,
    seed,
    seed_V: float | None,
    target_N: float,
    grid: MatsubaraGrid,
    role: str,
    depth: int,
    max_iter: int,
) -> tuple[object, dict, dict[str, float] | None, float]:
    params = RubyParameters(
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        V=float(V),
    )
    mu0 = float(seed.mu) if seed is not None else 0.0
    opts = _gw_options(args, mu0, target_N, max_iter=max_iter)

    print(
        f"    GW attempt role={role}, V={V:.10g}, "
        f"seed_V={'cold' if seed_V is None else f'{seed_V:.10g}'}, "
        f"depth={depth}, max_iter={max_iter}"
    )
    t0 = time.perf_counter()
    gw = solve_supercell_gw_anderson(
        params,
        grid,
        opts=opts,
        source_strength=0.0,
        initial=seed,
        anderson=AndersonOptions(),
    )
    runtime = time.perf_counter() - t0

    diag = _normal_diagnostics(gw, grid) if _finite_gw_state(gw) else None
    breaking = _normal_breaking_scale(diag) if diag is not None else float("inf")
    accepted_normal = bool(
        gw is not None
        and bool(gw.converged)
        and diag is not None
        and breaking <= float(args.normal_threshold)
    )

    print(
        f"      converged={bool(gw.converged)}, it={int(gw.iterations)}, "
        f"res={float(gw.final_error):.3e}, mu={float(gw.mu):.10f}, "
        f"normal_scale={breaking:.3e}, accepted_normal={accepted_normal}, "
        f"time={runtime:.1f}s"
    )

    row = {
        "requested_V": float(requested_V),
        "solve_V": float(V),
        "role": str(role),
        "depth": int(depth),
        "seed_V": np.nan if seed_V is None else float(seed_V),
        "max_iter_budget": int(max_iter),
        "converged": bool(gw.converged),
        "accepted_normal": accepted_normal,
        "iterations": int(gw.iterations),
        "residual": float(gw.final_error),
        "mu": float(gw.mu),
        "normal_breaking_scale": float(breaking),
        "runtime_s": float(runtime),
        "checkpoint": "",
    }
    return gw, row, diag, float(breaking)


def _advance_normal_branch(
    args,
    left_V: float | None,
    left_state,
    target_V: float,
    requested_V: float,
    target_N: float,
    grid: MatsubaraGrid,
    primitive_filling: float,
    bridge_dir: Path,
    continuation_rows: list[dict],
    continuation_log_path: Path,
    depth: int = 0,
):
    """Advance a converged normal state to target_V with retry + midpoint bridges."""

    is_requested = bool(np.isclose(float(target_V), float(requested_V), atol=1e-13))
    direct_role = "requested-direct" if is_requested else "bridge-direct"
    retry_role = "requested-retry" if is_requested else "bridge-retry"

    gw, row, diag, breaking = _run_normal_gw_attempt(
        args,
        target_V,
        requested_V,
        left_state,
        left_V,
        target_N,
        grid,
        direct_role,
        depth,
        int(args.gw_max_iter),
    )
    continuation_rows.append(row)
    _write_rows(continuation_log_path, continuation_rows)

    if bool(row["accepted_normal"]):
        if not is_requested:
            params = RubyParameters(
                ti=float(args.ti),
                t1=float(args.t1),
                t2=float(args.t2),
                V=float(target_V),
            )
            ckpt = bridge_dir / _bridge_checkpoint_name(
                target_V, primitive_filling, grid
            )
            save_supercell_checkpoint(
                ckpt, gw, params, grid, primitive_filling, source=0.0
            )
            row["checkpoint"] = str(ckpt)
            _write_rows(continuation_log_path, continuation_rows)
            print("      bridge checkpoint:", ckpt)
        return gw

    best_failed = gw if _finite_gw_state(gw) else None
    retry_seed = best_failed if best_failed is not None else left_state
    retry_seed_V = float(target_V) if best_failed is not None else left_V

    gw_retry, retry_row, retry_diag, retry_breaking = _run_normal_gw_attempt(
        args,
        target_V,
        requested_V,
        retry_seed,
        retry_seed_V,
        target_N,
        grid,
        retry_role,
        depth,
        int(args.gw_retry_max_iter),
    )
    continuation_rows.append(retry_row)
    _write_rows(continuation_log_path, continuation_rows)

    if bool(retry_row["accepted_normal"]):
        if not is_requested:
            params = RubyParameters(
                ti=float(args.ti),
                t1=float(args.t1),
                t2=float(args.t2),
                V=float(target_V),
            )
            ckpt = bridge_dir / _bridge_checkpoint_name(
                target_V, primitive_filling, grid
            )
            save_supercell_checkpoint(
                ckpt, gw_retry, params, grid, primitive_filling, source=0.0
            )
            retry_row["checkpoint"] = str(ckpt)
            _write_rows(continuation_log_path, continuation_rows)
            print("      bridge checkpoint:", ckpt)
        return gw_retry

    if _finite_gw_state(gw_retry) and (
        best_failed is None
        or float(gw_retry.final_error) < float(best_failed.final_error)
    ):
        best_failed = gw_retry

    if bool(args.no_auto_bridge):
        return None

    midpoint = _bridge_midpoint(
        left_V,
        target_V,
        float(args.bridge_min_step),
        int(depth),
        int(args.bridge_max_depth),
    )
    if midpoint is None or left_state is None:
        return None

    print(
        f"    direct/retry failed at V={target_V:.10g}; insert bridge "
        f"V={midpoint:.10g} between {float(left_V):.10g} and {target_V:.10g}"
    )

    bridge_state = _advance_normal_branch(
        args,
        left_V,
        left_state,
        midpoint,
        requested_V,
        target_N,
        grid,
        primitive_filling,
        bridge_dir,
        continuation_rows,
        continuation_log_path,
        depth=depth + 1,
    )
    if bridge_state is None:
        return None

    return _advance_normal_branch(
        args,
        midpoint,
        bridge_state,
        target_V,
        requested_V,
        target_N,
        grid,
        primitive_filling,
        bridge_dir,
        continuation_rows,
        continuation_log_path,
        depth=depth + 1,
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
    if args.gw_max_iter < 1 or args.gw_retry_max_iter < 1:
        raise ValueError("GW iteration limits must be positive")
    if args.bridge_min_step <= 0.0 or args.bridge_max_depth < 0:
        raise ValueError("bridge minimum step must be positive and max depth nonnegative")
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
    bridge_dir = checkpoint_dir / "bridges"
    bridge_dir.mkdir(parents=True, exist_ok=True)

    continuation_log_path = outdir / "gw_continuation_log.csv"
    continuation_rows: list[dict] = []

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
    print("requested V schedule:", " -> ".join(f"{v:g}" for v in V_values))
    print(
        f"adaptive GW continuation: retry_max_iter={args.gw_retry_max_iter}, "
        f"bridge_min_step={args.bridge_min_step:g}, bridge_max_depth={args.bridge_max_depth}, "
        f"auto_bridge={not args.no_auto_bridge}"
    )
    print("output:", outdir)
    print("requested-point normal GW checkpoints:", checkpoint_dir)
    print("bridge-only GW checkpoints:", bridge_dir)
    print("(These directories are separate from results/supercell18/branch_search/...)")
    print("=" * 110)

    previous = None
    previous_V: float | None = None
    if args.start_checkpoint is not None:
        start_path = Path(args.start_checkpoint)
        meta = read_checkpoint_metadata(start_path)
        if not bool(meta.get("converged", False)):
            raise ValueError("--start-checkpoint is not marked converged")
        if not np.isclose(float(meta.get("source", np.nan)), 0.0, atol=1e-14):
            raise ValueError("--start-checkpoint must be a zero-source GW state")
        previous_V = float(meta["V"])
        if previous_V > float(V_values[0]) + 1e-12:
            raise ValueError(
                f"--start-checkpoint has V={previous_V:g}, which exceeds first requested "
                f"V={V_values[0]:g}. This driver follows V in ascending order."
            )
        p0 = RubyParameters(
            ti=float(args.ti), t1=float(args.t1), t2=float(args.t2), V=V_values[0]
        )
        previous, _, _ = load_supercell_checkpoint(
            start_path, p0, grid, float(args.primitive_filling)
        )
        print(f"warm start: {start_path} (checkpoint V={previous_V:g})")
    else:
        print("warm start: none; first V will use the built-in weak-V bootstrap")

    rows: list[dict] = []
    csv_path = outdir / "r_vs_V.csv"

    for iv, V in enumerate(V_values, start=1):
        print(f"\n[{iv}/{len(V_values)}] requested V={V:g}: normal zero-source SC-GW")
        attempt_start = len(continuation_rows)

        if previous is not None and previous_V is not None and np.isclose(
            float(previous_V), float(V), atol=1e-12
        ):
            gw = previous
            print("  seed checkpoint is already at this requested V; reuse converged GW state")
            normal = _normal_diagnostics(gw, grid)
            breaking = _normal_breaking_scale(normal)
            if breaking > float(args.normal_threshold):
                raise RuntimeError(
                    f"start checkpoint at V={V:g} is not symmetry-preserving: "
                    f"normal scale={breaking:.3e}"
                )
        else:
            print(
                "  seed:",
                "cold/bootstrap" if previous is None else f"converged normal V={previous_V:g}",
            )
            gw = _advance_normal_branch(
                args,
                previous_V,
                previous,
                float(V),
                float(V),
                target_N,
                grid,
                float(args.primitive_filling),
                bridge_dir,
                continuation_rows,
                continuation_log_path,
                depth=0,
            )
            if gw is None:
                recent = continuation_rows[attempt_start:]
                best_res = min(
                    [float(r["residual"]) for r in recent if np.isfinite(r["residual"])],
                    default=float("nan"),
                )
                raise RuntimeError(
                    f"normal SC-GW continuation failed at requested V={V:g} even after "
                    f"fresh-history retry and adaptive bridges; best residual={best_res:.3e}. "
                    f"See {continuation_log_path}."
                )
            normal = _normal_diagnostics(gw, grid)
            breaking = _normal_breaking_scale(normal)

        recent_attempts = continuation_rows[attempt_start:]
        gw_runtime = float(sum(float(r["runtime_s"]) for r in recent_attempts))
        gw_iterations_total = int(
            sum(int(r["iterations"]) for r in recent_attempts if np.isfinite(r["iterations"]))
        )
        bridge_points_used = len(
            {
                float(r["solve_V"])
                for r in recent_attempts
                if str(r["role"]).startswith("bridge-") and bool(r["accepted_normal"])
            }
        )

        print(
            f"  accepted GW: res={gw.final_error:.3e}, mu={gw.mu:.10f}, "
            f"continuation attempts={len(recent_attempts)}, bridge points used={bridge_points_used}, "
            f"continuation time={gw_runtime:.1f}s"
        )
        print(
            f"  normal diagnostics: Delta_Q={normal['Delta_Q']:.3e}, "
            f"Delta_intra={normal['Delta_intra']:.3e}, Delta_AB={normal['Delta_AB']:+.3e}, "
            f"|m+|/pc={normal['m_plus_pc_abs']:.3e}, |m-|/pc={normal['m_minus_pc_abs']:.3e}"
        )
        if breaking > float(args.normal_threshold):
            raise RuntimeError(
                f"continued background at V={V:g} is no longer symmetry-preserving: "
                f"max order diagnostic={breaking:.3e} > threshold={args.normal_threshold:.3e}."
            )

        params = RubyParameters(
            ti=float(args.ti),
            t1=float(args.t1),
            t2=float(args.t2),
            V=float(V),
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
        print("  requested-point checkpoint:", ckpt)

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
            "gw_iterations_final_attempt": int(gw.iterations),
            "gw_iterations_continuation_total": int(gw_iterations_total),
            "gw_continuation_attempts": int(len(recent_attempts)),
            "gw_bridge_points_used": int(bridge_points_used),
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
        previous_V = float(V)

    print("\n" + "=" * 110)
    print("SCAN COMPLETE")
    print("=" * 110)
    print("table:", csv_path)
    print("GW continuation log:", continuation_log_path)
    print("r+/r- plot:", outdir / "r_plus_minus_vs_V.png")
    print("diagnostic r+/r-/rQ plot:", outdir / "r_plus_minus_Q_vs_V.png")
    crossings = outdir / "critical_V_estimates.csv"
    if crossings.exists():
        print("linear zero-crossing estimates:", crossings)
    else:
        print("linear zero-crossing estimates: none bracketed by the supplied V grid")
    print("requested-point normal GW checkpoints:", checkpoint_dir)
    print("bridge-only normal GW checkpoints:", bridge_dir)


if __name__ == "__main__":
    main()
