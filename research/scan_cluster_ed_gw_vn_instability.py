#!/usr/bin/env python3
"""Build a coarse V-filling instability/softness map by continuation + all-q JF.

Workflow
--------
1. Solve one orientation-resolved physical-pair cluster-ED+GW reference branch
   on a rectangular (V, filling) grid.
2. Traverse the grid in a snake path so every new point is warm-started from a
   nearest-neighbour point in parameter space.
3. For every converged checkpoint, build the finite-bath JF tangent once and
   scan all q points.
4. Save the softest LC and CO pole distances

       d_alpha(V,n) = min_q |1-lambda_alpha(q;V,n)|

   together with q*, lambda*, mode weights, and convergence diagnostics.

This is an instability/softness map of the followed reference branch.  It is
not by itself a thermodynamic phase diagram when first-order coexistence is
possible; source-ramped ordered branches should be used afterwards to validate
candidate instability lines.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = _REPO_ROOT / "research" / "run_cluster_ed_gw_vprime_vcross_orientation.py"
JF = _REPO_ROOT / "research" / "analyze_cluster_ed_gw_vprime_vcross_orientation_lambda.py"
TRACKER = _REPO_ROOT / "research" / "track_cluster_ed_gw_vn_modes.py"


def _csv_floats(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return vals


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--V-values", type=_csv_floats, required=True,
        help='comma-separated V grid, e.g. "0.8,1.0,1.2,1.4,1.6,1.8"',
    )
    p.add_argument(
        "--fillings", type=_csv_floats, required=True,
        help='comma-separated total filling per six-site primitive cell, e.g. "1.6,1.8,2.0,2.2"',
    )
    p.add_argument("--orientation", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--Vprime", "--Vp", dest="Vprime", type=float, default=-0.10)
    p.add_argument("--Vcross", "--Vx", dest="Vcross", type=float, default=-0.07)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)

    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--nbath", type=int, default=6)

    p.add_argument("--gw-max", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument(
        "--allow-unconverged-gw-background",
        action="store_true",
        help=(
            "allow the standalone SC-GW stage to hit --gw-max and pass its last "
            "finite iterate to cluster ED+GW as an initializer"
        ),
    )

    p.add_argument("--embed-max", type=int, default=120)
    p.add_argument("--embed-tol", type=float, default=5e-7)
    p.add_argument("--embed-mixing-method", choices=("linear", "pulay", "broyden"), default="broyden")
    p.add_argument("--embed-mixing", type=float, default=0.5)
    p.add_argument("--embed-broyden-history", type=int, default=8)
    p.add_argument("--embed-broyden-step-cap", type=float, default=3.0)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--bath-fit-metric", choices=("delta", "g0"), default="delta")

    p.add_argument("--nev", type=int, default=16)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--stage", choices=("mt", "full"), default="full")

    p.add_argument(
        "--out", type=Path,
        default=Path("results/vn_instability_orientation"),
    )
    p.add_argument(
        "--start-from", type=Path, default=None,
        help="optional converged physical-pair checkpoint used to seed the first grid point",
    )
    p.add_argument(
        "--background-only", action="store_true",
        help="build/continue the reference checkpoints but skip all-q JF",
    )
    p.add_argument(
        "--rerun", action="store_true",
        help="rerun points even when their checkpoint/JF output already exists",
    )
    p.add_argument(
        "--keep-going", action="store_true",
        help="record failed points and continue the grid instead of stopping",
    )
    p.add_argument(
        "--skip-tracking", action="store_true",
        help="do not run overlap-based mode tracking after the grid finishes",
    )
    p.add_argument("--quiet-solvers", action="store_true")
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _checkpoint_path(bgdir: Path, args, V: float, filling: float) -> Path:
    return bgdir / (
        f"cluster_ed_gw_vprime_vcross_ori{args.orientation}_"
        f"L{args.Lx}x{args.Ly}_V{_tag(V)}_Vp{_tag(args.Vprime)}_"
        f"Vx{_tag(args.Vcross)}_fill{_tag(filling)}.npz"
    )


def _jf_path(jfdir: Path, args, V: float, filling: float) -> Path:
    return jfdir / (
        f"lambda_allq_ori{args.orientation}_L{args.Lx}x{args.Ly}_"
        f"V{_tag(V)}_Vp{_tag(args.Vprime)}_Vx{_tag(args.Vcross)}_"
        f"fill{_tag(filling)}.npz"
    )


def _npz_scalar(path: Path, key: str, default=np.nan):
    try:
        with np.load(path, allow_pickle=False) as z:
            if key not in z:
                return default
            return np.asarray(z[key]).reshape(()).item()
    except Exception:
        return default


def _checkpoint_is_converged(path: Path, tol: float) -> bool:
    if not path.exists():
        return False
    conv = bool(_npz_scalar(path, "converged", False))
    err = float(_npz_scalar(path, "final_error", np.inf))
    return conv and np.isfinite(err) and err <= float(tol)


def _jf_has_full_spectrum(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {
                "full_spectrum_schema",
                "mode_vectors_saved",
                "mode_static_matrix",
            }
            if not required.issubset(set(z.files)):
                return False
            schema = int(np.asarray(z["full_spectrum_schema"]).reshape(()))
            return (
                schema >= 2
                and bool(np.asarray(z["mode_vectors_saved"]).reshape(()))
            )
    except Exception:
        return False


def _run(cmd: list[str], *, cwd: Path):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _background_command(args, V, filling, bgdir, seed):
    cmd = [
        sys.executable, str(RUNNER),
        "--orientation", str(args.orientation),
        "--Lx", str(args.Lx), "--Ly", str(args.Ly),
        "--V", repr(float(V)),
        "--Vp", repr(float(args.Vprime)),
        "--Vx", repr(float(args.Vcross)),
        "--ti", repr(float(args.ti)),
        "--t1", repr(float(args.t1)),
        "--t2", repr(float(args.t2)),
        "--filling", repr(float(filling)),
        "--T", repr(float(args.T)),
        "--nw", str(args.nw),
        "--nomega", str(args.nomega),
        "--gw-max", str(args.gw_max),
        "--gw-tol", repr(float(args.gw_tol)),
        "--gw-mixing", repr(float(args.gw_mixing)),
        "--gw-mixing-method", str(args.gw_mixing_method),
        "--embed-max", str(args.embed_max),
        "--embed-tol", repr(float(args.embed_tol)),
        "--embed-mixing-method", str(args.embed_mixing_method),
        "--embed-mixing", repr(float(args.embed_mixing)),
        "--embed-broyden-history", str(args.embed_broyden_history),
        "--embed-broyden-step-cap", repr(float(args.embed_broyden_step_cap)),
        "--nbath", str(args.nbath),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-fit-max-nfev", str(args.bath_fit_max_nfev),
        "--bath-fit-metric", str(args.bath_fit_metric),
        "--out", str(bgdir),
    ]
    if bool(args.allow_unconverged_gw_background):
        cmd.append("--allow-unconverged-gw-background")
    if seed is not None:
        cmd.extend(["--continue-from", str(seed)])
    if bool(args.quiet_solvers):
        cmd.extend(["--quiet-gw", "--quiet-embed"])
    return cmd


def _jf_command(args, checkpoint, outfile):
    cmd = [
        sys.executable, str(JF), str(checkpoint),
        "--all-q",
        "--nev", str(args.nev),
        "--stage", str(args.stage),
        "--bath-rank", str(args.bath_rank),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-svd-rcond", repr(float(args.bath_svd_rcond)),
        "--bath-fd-step", repr(float(args.bath_fd_step)),
        "--out", str(outfile),
    ]
    if bool(args.quiet_solvers):
        cmd.append("--quiet")
    return cmd


def _snake_points(Vs, fillings):
    points = []
    for j, n in enumerate(fillings):
        indices = range(len(Vs)) if j % 2 == 0 else range(len(Vs) - 1, -1, -1)
        for i in indices:
            points.append((i, j, float(Vs[i]), float(n)))
    return points


def _summarize_jf(path: Path):
    with np.load(path, allow_pickle=False) as z:
        q = np.asarray(z["q_index"], dtype=int)
        dlc = np.asarray(z["distance_to_one_lc"], dtype=float)
        dco = np.asarray(z["distance_to_one_co"], dtype=float)
        llc = np.asarray(z["lambda_lc"], dtype=complex)
        lco = np.asarray(z["lambda_co"], dtype=complex)
        same = np.asarray(z["lc_same_weight"], dtype=float)
        opp = np.asarray(z["lc_opposite_weight"], dtype=float)
        even = np.asarray(z["co_even_weight"], dtype=float)
        odd = np.asarray(z["co_odd_weight"], dtype=float)
    ilc = int(np.nanargmin(dlc))
    ico = int(np.nanargmin(dco))
    return dict(
        d_lc=float(dlc[ilc]),
        d_co=float(dco[ico]),
        q_lc=np.asarray(q[ilc], dtype=int),
        q_co=np.asarray(q[ico], dtype=int),
        lambda_lc=complex(llc[ilc]),
        lambda_co=complex(lco[ico]),
        lc_same=float(same[ilc]),
        lc_opposite=float(opp[ilc]),
        co_even=float(even[ico]),
        co_odd=float(odd[ico]),
    )


def _save_summary(path, args, Vs, fillings, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        V_values=np.asarray(Vs, dtype=float),
        fillings=np.asarray(fillings, dtype=float),
        orientation=int(args.orientation),
        Vprime=float(args.Vprime),
        Vcross=float(args.Vcross),
        T=float(args.T),
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        Lx=int(args.Lx), Ly=int(args.Ly),
        nw=int(args.nw), nomega=int(args.nomega), nbath=int(args.nbath),
        stage=np.asarray(str(args.stage)),
        **arrays,
    )


def main():
    args = _args()
    Vs = [float(x) for x in args.V_values]
    fillings = [float(x) for x in args.fillings]
    if len(set(Vs)) != len(Vs) or len(set(fillings)) != len(fillings):
        raise ValueError("V-values and fillings must not contain duplicates")

    args.out.mkdir(parents=True, exist_ok=True)
    bgdir = args.out / "backgrounds"
    jfdir = args.out / "jf_allq"
    bgdir.mkdir(parents=True, exist_ok=True)
    jfdir.mkdir(parents=True, exist_ok=True)

    shape = (len(fillings), len(Vs))
    arrays = dict(
        background_converged=np.zeros(shape, dtype=bool),
        background_residual=np.full(shape, np.nan),
        background_bath_error=np.full(shape, np.nan),
        background_impurity_mismatch=np.full(shape, np.nan),
        mu=np.full(shape, np.nan),
        d_lc=np.full(shape, np.nan),
        d_co=np.full(shape, np.nan),
        delta_d_co_minus_lc=np.full(shape, np.nan),
        lambda_lc=np.full(shape, np.nan + 1j * np.nan, dtype=complex),
        lambda_co=np.full(shape, np.nan + 1j * np.nan, dtype=complex),
        q_lc=np.full(shape + (2,), -1, dtype=int),
        q_co=np.full(shape + (2,), -1, dtype=int),
        lc_same_weight=np.full(shape, np.nan),
        lc_opposite_weight=np.full(shape, np.nan),
        co_even_weight=np.full(shape, np.nan),
        co_odd_weight=np.full(shape, np.nan),
        point_failed=np.zeros(shape, dtype=bool),
    )
    summary_path = args.out / "vn_instability_summary.npz"

    seed = args.start_from
    if seed is not None and not seed.exists():
        raise FileNotFoundError(seed)

    points = _snake_points(Vs, fillings)
    print(
        "=== V-filling physical-pair instability scan ===\n"
        f"orientation={args.orientation}, grid={len(Vs)} x {len(fillings)}, "
        f"L={args.Lx}x{args.Ly}, T={args.T:g}\n"
        f"V'={args.Vprime:g}, Vx={args.Vcross:g}\n"
        f"snake points={len(points)}, all-q JF={'off' if args.background_only else 'on'}",
        flush=True,
    )

    for ip, (iv, inn, V, filling) in enumerate(points, 1):
        chk = _checkpoint_path(bgdir, args, V, filling)
        jfout = _jf_path(jfdir, args, V, filling)
        print(
            f"\n=== point {ip}/{len(points)}: V={V:g}, filling={filling:g} "
            f"[grid n-index={inn}, V-index={iv}] ===",
            flush=True,
        )
        try:
            if bool(args.rerun) or not _checkpoint_is_converged(chk, args.embed_tol):
                _run(
                    _background_command(args, V, filling, bgdir, seed),
                    cwd=_REPO_ROOT,
                )
            if not _checkpoint_is_converged(chk, args.embed_tol):
                raise RuntimeError(f"background did not converge: {chk}")

            arrays["background_converged"][inn, iv] = True
            arrays["background_residual"][inn, iv] = float(
                _npz_scalar(chk, "final_error", np.nan)
            )
            arrays["background_bath_error"][inn, iv] = float(
                _npz_scalar(chk, "bath_fit_error", np.nan)
            )
            arrays["background_impurity_mismatch"][inn, iv] = float(
                _npz_scalar(chk, "impurity_mismatch", np.nan)
            )
            arrays["mu"][inn, iv] = float(_npz_scalar(chk, "mu", np.nan))

            # The next snake point is always seeded from the current converged
            # checkpoint, including row changes in filling.
            seed = chk

            if not bool(args.background_only):
                if bool(args.rerun) or not _jf_has_full_spectrum(jfout):
                    if jfout.exists() and not bool(args.rerun):
                        print(
                            "[JF] existing output lacks full-spectrum vectors; "
                            "recomputing tracking-ready spectrum",
                            flush=True,
                        )
                    _run(_jf_command(args, chk, jfout), cwd=_REPO_ROOT)
                s = _summarize_jf(jfout)
                arrays["d_lc"][inn, iv] = s["d_lc"]
                arrays["d_co"][inn, iv] = s["d_co"]
                arrays["delta_d_co_minus_lc"][inn, iv] = s["d_co"] - s["d_lc"]
                arrays["lambda_lc"][inn, iv] = s["lambda_lc"]
                arrays["lambda_co"][inn, iv] = s["lambda_co"]
                arrays["q_lc"][inn, iv] = s["q_lc"]
                arrays["q_co"][inn, iv] = s["q_co"]
                arrays["lc_same_weight"][inn, iv] = s["lc_same"]
                arrays["lc_opposite_weight"][inn, iv] = s["lc_opposite"]
                arrays["co_even_weight"][inn, iv] = s["co_even"]
                arrays["co_odd_weight"][inn, iv] = s["co_odd"]

                print(
                    "[grid summary] "
                    f"LC: d={s['d_lc']:.4e}, q*={tuple(s['q_lc'])}, "
                    f"lambda={s['lambda_lc'].real:+.6f}{s['lambda_lc'].imag:+.2e}i; "
                    f"CO: d={s['d_co']:.4e}, q*={tuple(s['q_co'])}, "
                    f"lambda={s['lambda_co'].real:+.6f}{s['lambda_co'].imag:+.2e}i; "
                    f"dCO-dLC={s['d_co']-s['d_lc']:+.4e}",
                    flush=True,
                )
        except BaseException as exc:
            arrays["point_failed"][inn, iv] = True
            print(f"[FAILED] V={V:g}, filling={filling:g}: {exc}", flush=True)
            _save_summary(summary_path, args, Vs, fillings, arrays)
            if not bool(args.keep_going):
                raise
            # Do not replace the seed with a failed point.  The next point
            # continues from the most recent converged neighbour instead.

        _save_summary(summary_path, args, Vs, fillings, arrays)

    print(f"\nsaved grid summary: {summary_path}", flush=True)
    if not bool(args.background_only) and not bool(args.skip_tracking):
        print("\n=== overlap tracking of retained JF branches ===", flush=True)
        _run(
            [sys.executable, str(TRACKER), str(summary_path)],
            cwd=_REPO_ROOT,
        )


if __name__ == "__main__":
    main()
