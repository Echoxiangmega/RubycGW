#!/usr/bin/env python3
"""Evaluate direct all-q physical susceptibilities on an existing V-filling grid.

This driver reuses the converged physical-pair cluster-ED+GW backgrounds from
vn_instability_summary.npz.  It does not rerun nonlinear ED+GW.  At every
available grid point it calls the direct physical susceptibility analyzer and
stores both the full q-resolved soft CO/LC susceptibilities and convenient
all-q softest summaries.

The primary continuous-instability diagnostics are the signed inverse physical
susceptibilities

    m_CO(q) = 1 / chi_CO,soft(q),
    m_LC(q) = 1 / chi_LC,soft(q),

with a branch instability at m -> 0.  These are physical response quantities;
the large Jacobian eigenspectrum is not needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = (
    _REPO_ROOT
    / "research"
    / "analyze_cluster_ed_gw_vprime_vcross_orientation_chi.py"
)


def _csv_floats(text):
    if text is None:
        return None
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return vals


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--V-values", type=_csv_floats, default=None)
    p.add_argument("--fillings", type=_csv_floats, default=None)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--bath-fd-scheme", choices=("centered", "forward"), default="centered")
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--stage", choices=("mt", "full"), default="full")
    p.add_argument("--jf-tol", type=float, default=1e-8)
    p.add_argument("--jf-maxiter", type=int, default=100)
    p.add_argument("--jf-restart", type=int, default=28)
    p.add_argument("--jf-recycle-dim", type=int, default=14)
    p.add_argument("--no-recycle", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _scalar(d, key, cast=float):
    return cast(np.asarray(d[key]).reshape(()))


def _tag(x):
    return f"{float(x):g}"


def _find_index(values, value):
    idx = np.where(np.isclose(values, float(value), rtol=0.0, atol=1e-10))[0]
    if len(idx) != 1:
        raise ValueError(f"value {value:g} is not a unique point of the saved grid")
    return int(idx[0])


def _checkpoint_path(root, meta, V, filling):
    return root / "backgrounds" / (
        f"cluster_ed_gw_vprime_vcross_ori{meta['orientation']}_"
        f"L{meta['Lx']}x{meta['Ly']}_V{_tag(V)}_"
        f"Vp{_tag(meta['Vprime'])}_Vx{_tag(meta['Vcross'])}_"
        f"fill{_tag(filling)}.npz"
    )


def _chi_path(chidir, meta, V, filling):
    return chidir / (
        f"chi_allq_ori{meta['orientation']}_L{meta['Lx']}x{meta['Ly']}_"
        f"V{_tag(V)}_Vp{_tag(meta['Vprime'])}_Vx{_tag(meta['Vcross'])}_"
        f"fill{_tag(filling)}.npz"
    )


def _valid_chi(path):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return (
                "schema" in z
                and int(np.asarray(z["schema"]).reshape(())) >= 1
                and "inv_chi_co_soft" in z
                and "inv_chi_lc_soft" in z
            )
    except Exception:
        return False


def _command(args, checkpoint, outfile):
    cmd = [
        sys.executable,
        str(ANALYZER),
        str(checkpoint),
        "--all-q",
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-svd-rcond", repr(float(args.bath_svd_rcond)),
        "--bath-rank", str(args.bath_rank),
        "--bath-fd-step", repr(float(args.bath_fd_step)),
        "--bath-fd-scheme", str(args.bath_fd_scheme),
        "--discard-weight-tol", repr(float(args.discard_weight_tol)),
        "--stage", str(args.stage),
        "--jf-tol", repr(float(args.jf_tol)),
        "--jf-maxiter", str(args.jf_maxiter),
        "--jf-restart", str(args.jf_restart),
        "--jf-recycle-dim", str(args.jf_recycle_dim),
        "--out", str(outfile),
    ]
    if args.no_recycle:
        cmd.append("--no-recycle")
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def _run(cmd):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


def _reshape_q(values, q, Lx, Ly, *, trailing=()):
    out = np.full((Lx, Ly) + tuple(trailing), np.nan, dtype=np.asarray(values).dtype)
    for i, qn in enumerate(np.asarray(q, dtype=int)):
        out[int(qn[0]), int(qn[1])] = values[i]
    return out


def _summarize_one(path, Lx, Ly):
    d = _load(path)
    q = np.asarray(d["q_index"], dtype=int)
    chi_co = _reshape_q(np.asarray(d["chi_co_soft"], dtype=float), q, Lx, Ly)
    inv_co = _reshape_q(np.asarray(d["inv_chi_co_soft"], dtype=float), q, Lx, Ly)
    chi_lc = _reshape_q(np.asarray(d["chi_lc_soft"], dtype=float), q, Lx, Ly)
    inv_lc = _reshape_q(np.asarray(d["inv_chi_lc_soft"], dtype=float), q, Lx, Ly)
    chi_full = _reshape_q(np.asarray(d["chi_full_soft"], dtype=float), q, Lx, Ly)
    inv_full = _reshape_q(np.asarray(d["inv_chi_full_soft"], dtype=float), q, Lx, Ly)

    co_w = _reshape_q(
        np.asarray(d["co_soft_weights"], dtype=float), q, Lx, Ly, trailing=(4,)
    )
    lc_w = _reshape_q(
        np.asarray(d["lc_soft_weights"], dtype=float), q, Lx, Ly, trailing=(2,)
    )
    full_w = _reshape_q(
        np.asarray(d["full_soft_weights"], dtype=float), q, Lx, Ly, trailing=(6,)
    )
    qpair = _reshape_q(np.asarray(d["q_pair_residual"], dtype=float), q, Lx, Ly)
    cross = _reshape_q(np.asarray(d["co_lc_cross_ratio"], dtype=float), q, Lx, Ly)

    ico = np.unravel_index(int(np.nanargmin(np.abs(inv_co))), inv_co.shape)
    ilc = np.unravel_index(int(np.nanargmin(np.abs(inv_lc))), inv_lc.shape)
    ifull = np.unravel_index(int(np.nanargmin(np.abs(inv_full))), inv_full.shape)
    return dict(
        chi_co_q=chi_co,
        inv_co_q=inv_co,
        chi_lc_q=chi_lc,
        inv_lc_q=inv_lc,
        chi_full_q=chi_full,
        inv_full_q=inv_full,
        co_weights_q=co_w,
        lc_weights_q=lc_w,
        full_weights_q=full_w,
        qpair_residual_q=qpair,
        cross_ratio_q=cross,
        chi_co=float(chi_co[ico]),
        inv_co=float(inv_co[ico]),
        q_co=np.asarray(ico, dtype=int),
        co_even=float(co_w[ico][0] + co_w[ico][1]),
        co_odd=float(co_w[ico][2] + co_w[ico][3]),
        chi_lc=float(chi_lc[ilc]),
        inv_lc=float(inv_lc[ilc]),
        q_lc=np.asarray(ilc, dtype=int),
        lc_same=float(lc_w[ilc][0]),
        lc_opposite=float(lc_w[ilc][1]),
        chi_full=float(chi_full[ifull]),
        inv_full=float(inv_full[ifull]),
        q_full=np.asarray(ifull, dtype=int),
        full_co_weight=float(np.sum(full_w[ifull][:4])),
        full_lc_weight=float(np.sum(full_w[ifull][4:])),
        max_qpair_residual=float(np.nanmax(qpair)),
        max_cross_ratio=float(np.nanmax(cross)),
        max_solver_residual=float(np.nanmax(np.asarray(d["solver_residual"], dtype=float))),
    )


def _save(path, source, meta, Vs, fillings, arrays):
    np.savez_compressed(
        path,
        schema=np.asarray(1, dtype=int),
        source_summary=np.asarray(str(source)),
        V_values=np.asarray(Vs, dtype=float),
        fillings=np.asarray(fillings, dtype=float),
        orientation=int(meta["orientation"]),
        Vprime=float(meta["Vprime"]),
        Vcross=float(meta["Vcross"]),
        T=float(meta["T"]),
        Lx=int(meta["Lx"]),
        Ly=int(meta["Ly"]),
        channels=np.asarray((
            "x_even", "y_even", "x_odd", "y_odd", "z_even", "z_odd"
        )),
        **arrays,
    )


def main():
    args = _args()
    src = _load(args.summary)
    src_Vs = np.asarray(src["V_values"], dtype=float)
    src_ns = np.asarray(src["fillings"], dtype=float)
    Vs = np.asarray(args.V_values if args.V_values is not None else src_Vs, dtype=float)
    fillings = np.asarray(
        args.fillings if args.fillings is not None else src_ns, dtype=float
    )
    for V in Vs:
        _find_index(src_Vs, V)
    for n in fillings:
        _find_index(src_ns, n)

    meta = dict(
        orientation=_scalar(src, "orientation", int),
        Vprime=_scalar(src, "Vprime", float),
        Vcross=_scalar(src, "Vcross", float),
        T=_scalar(src, "T", float),
        Lx=_scalar(src, "Lx", int),
        Ly=_scalar(src, "Ly", int),
    )
    root = args.summary.parent
    outdir = args.out or (root / "susceptibility")
    chidir = outdir / "chi_allq"
    outdir.mkdir(parents=True, exist_ok=True)
    chidir.mkdir(parents=True, exist_ok=True)

    shape = (len(fillings), len(Vs))
    qshape = shape + (meta["Lx"], meta["Ly"])
    arrays = dict(
        point_failed=np.zeros(shape, dtype=bool),
        chi_co=np.full(shape, np.nan),
        inv_chi_co=np.full(shape, np.nan),
        q_co=np.full(shape + (2,), -1, dtype=int),
        co_even_weight=np.full(shape, np.nan),
        co_odd_weight=np.full(shape, np.nan),
        chi_lc=np.full(shape, np.nan),
        inv_chi_lc=np.full(shape, np.nan),
        q_lc=np.full(shape + (2,), -1, dtype=int),
        lc_same_weight=np.full(shape, np.nan),
        lc_opposite_weight=np.full(shape, np.nan),
        chi_full=np.full(shape, np.nan),
        inv_chi_full=np.full(shape, np.nan),
        q_full=np.full(shape + (2,), -1, dtype=int),
        full_co_weight=np.full(shape, np.nan),
        full_lc_weight=np.full(shape, np.nan),
        chi_co_q=np.full(qshape, np.nan),
        inv_chi_co_q=np.full(qshape, np.nan),
        chi_lc_q=np.full(qshape, np.nan),
        inv_chi_lc_q=np.full(qshape, np.nan),
        chi_full_q=np.full(qshape, np.nan),
        inv_chi_full_q=np.full(qshape, np.nan),
        co_soft_weights_q=np.full(qshape + (4,), np.nan),
        lc_soft_weights_q=np.full(qshape + (2,), np.nan),
        full_soft_weights_q=np.full(qshape + (6,), np.nan),
        q_pair_residual_q=np.full(qshape, np.nan),
        co_lc_cross_ratio_q=np.full(qshape, np.nan),
        max_q_pair_residual=np.full(shape, np.nan),
        max_co_lc_cross_ratio=np.full(shape, np.nan),
        max_solver_residual=np.full(shape, np.nan),
    )
    summary_path = outdir / "vn_susceptibility_summary.npz"

    for inn, filling in enumerate(fillings):
        src_in = _find_index(src_ns, filling)
        for iv, V in enumerate(Vs):
            src_iv = _find_index(src_Vs, V)
            if (
                "background_converged" in src
                and not bool(src["background_converged"][src_in, src_iv])
            ):
                print(f"[skip] V={V:g}, filling={filling:g}: background not converged")
                arrays["point_failed"][inn, iv] = True
                _save(summary_path, args.summary, meta, Vs, fillings, arrays)
                continue

            checkpoint = _checkpoint_path(root, meta, V, filling)
            outfile = _chi_path(chidir, meta, V, filling)
            print(f"\n=== chi V={V:g}, filling={filling:g} ===", flush=True)
            try:
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                if args.rerun or not _valid_chi(outfile):
                    _run(_command(args, checkpoint, outfile))
                s = _summarize_one(outfile, meta["Lx"], meta["Ly"])

                arrays["chi_co"][inn, iv] = s["chi_co"]
                arrays["inv_chi_co"][inn, iv] = s["inv_co"]
                arrays["q_co"][inn, iv] = s["q_co"]
                arrays["co_even_weight"][inn, iv] = s["co_even"]
                arrays["co_odd_weight"][inn, iv] = s["co_odd"]
                arrays["chi_lc"][inn, iv] = s["chi_lc"]
                arrays["inv_chi_lc"][inn, iv] = s["inv_lc"]
                arrays["q_lc"][inn, iv] = s["q_lc"]
                arrays["lc_same_weight"][inn, iv] = s["lc_same"]
                arrays["lc_opposite_weight"][inn, iv] = s["lc_opposite"]
                arrays["chi_full"][inn, iv] = s["chi_full"]
                arrays["inv_chi_full"][inn, iv] = s["inv_full"]
                arrays["q_full"][inn, iv] = s["q_full"]
                arrays["full_co_weight"][inn, iv] = s["full_co_weight"]
                arrays["full_lc_weight"][inn, iv] = s["full_lc_weight"]

                arrays["chi_co_q"][inn, iv] = s["chi_co_q"]
                arrays["inv_chi_co_q"][inn, iv] = s["inv_co_q"]
                arrays["chi_lc_q"][inn, iv] = s["chi_lc_q"]
                arrays["inv_chi_lc_q"][inn, iv] = s["inv_lc_q"]
                arrays["chi_full_q"][inn, iv] = s["chi_full_q"]
                arrays["inv_chi_full_q"][inn, iv] = s["inv_full_q"]
                arrays["co_soft_weights_q"][inn, iv] = s["co_weights_q"]
                arrays["lc_soft_weights_q"][inn, iv] = s["lc_weights_q"]
                arrays["full_soft_weights_q"][inn, iv] = s["full_weights_q"]
                arrays["q_pair_residual_q"][inn, iv] = s["qpair_residual_q"]
                arrays["co_lc_cross_ratio_q"][inn, iv] = s["cross_ratio_q"]
                arrays["max_q_pair_residual"][inn, iv] = s["max_qpair_residual"]
                arrays["max_co_lc_cross_ratio"][inn, iv] = s["max_cross_ratio"]
                arrays["max_solver_residual"][inn, iv] = s["max_solver_residual"]

                print(
                    "[chi summary] "
                    f"CO: chi={s['chi_co']:+.6e}, 1/chi={s['inv_co']:+.6e}, "
                    f"q*={tuple(s['q_co'])}, even/odd={s['co_even']:.3f}/{s['co_odd']:.3f}; "
                    f"LC: chi={s['chi_lc']:+.6e}, 1/chi={s['inv_lc']:+.6e}, "
                    f"q*={tuple(s['q_lc'])}, same/opp={s['lc_same']:.3f}/{s['lc_opposite']:.3f}",
                    flush=True,
                )
            except BaseException as exc:
                arrays["point_failed"][inn, iv] = True
                print(f"[FAILED] V={V:g}, filling={filling:g}: {exc}", flush=True)
                _save(summary_path, args.summary, meta, Vs, fillings, arrays)
                if not args.keep_going:
                    raise

            _save(summary_path, args.summary, meta, Vs, fillings, arrays)

    print(f"\nsaved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
