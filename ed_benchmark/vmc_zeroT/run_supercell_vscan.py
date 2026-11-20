#!/usr/bin/env python3
"""Raw-only V scan at fixed filling 1/3 for ring3 supercells.

Primary use on a cluster:
  * one process / Slurm-array task per (L1,L2,seed,direction);
  * V values are serial inside that task when mode=continuation, so every
    seed tracks its own metastable branch by warm-starting from the previous V;
  * different seeds, sizes and up/down sweeps are independent and parallel.

Independent mode deliberately starts every V from the named seed.  It is
intended for sparse anchor points and phase-boundary checks, not as a
replacement for continuation when tracking branch energies.

Only the unprojected Raw state is run.  PAV and one-step Lanczos are disabled.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


HERE=Path(__file__).resolve().parent
RAW_RUNNER=HERE/"run_supercell_raw_pav.py"

DEFAULT_SEEDS=[
    "normal","sd","isd","lc_same","lc_opposite","random"
]


def _vtag(v: float) -> str:
    sign="m" if v < 0 else "p"
    x=f"{abs(float(v)):.8f}".rstrip("0").rstrip(".")
    return "V_"+sign+x.replace(".","p")


def _build_v_values(args):
    if args.V_values:
        vals=[float(x) for x in args.V_values]
    else:
        if args.V_min is None or args.V_max is None or args.V_step is None:
            raise ValueError(
                "Provide --V-values or all of --V-min/--V-max/--V-step"
            )
        if args.V_step<=0:
            raise ValueError("--V-step must be positive")
        n=int(math.floor(
            (float(args.V_max)-float(args.V_min))/float(args.V_step)+1e-10
        ))
        vals=[
            float(args.V_min)+i*float(args.V_step)
            for i in range(n+1)
        ]
        if not vals or vals[-1] < float(args.V_max)-1e-9:
            vals.append(float(args.V_max))
    vals=sorted(set(round(float(x),12) for x in vals))
    if args.direction=="down":
        vals=list(reversed(vals))
    return vals


def _auto_chunks(L1,L2):
    # Main fast-local temporary scales approximately linearly in chunk size
    # and quadratically with linear supercell size.  These are conservative
    # starting values for 24-40 GB accelerators.
    ns=18*int(L1)*int(L2)
    if ns<=288:      # through 4x4
        return 256,128
    if ns<=450:      # 5x5
        return 128,64
    if ns<=648:      # 6x6
        return 64,64
    return 32,32


def _auto_sweep(L1,L2):
    nf=6*int(L1)*int(L2)
    # About 2/3 of Nf reproduces the tested 2x2 value 16 while keeping
    # decorrelation effort proportional to system size.
    return max(16,int(round(2*nf/3)))


def _write_rows(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_one_row(point_dir,seed,correlator):
    summary=point_dir/"optimized_branch_summary.csv"
    raw_summary=point_dir/f"large_raw_{correlator}__{seed}"/"summary.json"
    if summary.exists():
        with summary.open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        if rows:
            return dict(rows[-1])
    if raw_summary.exists():
        return json.loads(raw_summary.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No completed Raw summary in {point_dir}"
    )


def _completed(point_dir,seed,correlator):
    stage=point_dir/f"large_raw_{correlator}__{seed}"
    return (
        (stage/"final_params.npz").exists()
        and (stage/"summary.json").exists()
    )


def _raw_stage_dir(point_dir,seed,correlator):
    return point_dir/f"large_raw_{correlator}__{seed}"


def _schedule_args(anchor: bool,args):
    if anchor or args.mode=="independent":
        return [
            "--raw-iter",str(args.anchor_iter),
            "--raw-learning-rate",str(args.anchor_lr),
            "--raw-learning-rate-final",str(args.anchor_lr_mid),
            "--raw-learning-rate-refine",str(args.anchor_lr_refine),
            "--raw-lr-switch-iter",str(args.anchor_switch1),
            "--raw-refine-switch-iter",str(args.anchor_switch2),
            "--train-samples-switch-iter",str(args.anchor_switch1),
            "--train-samples-refine-switch-iter",str(args.anchor_switch2),
            "--diag-shift",str(args.anchor_diag),
            "--diag-shift-mid",str(args.anchor_diag_mid),
            "--diag-shift-refine",str(args.anchor_diag_refine),
        ]
    return [
        "--raw-iter",str(args.cont_iter),
        "--raw-learning-rate",str(args.cont_lr),
        "--raw-learning-rate-final",str(args.cont_lr_mid),
        "--raw-learning-rate-refine",str(args.cont_lr_refine),
        "--raw-lr-switch-iter",str(args.cont_switch1),
        "--raw-refine-switch-iter",str(args.cont_switch2),
        "--train-samples-switch-iter",str(args.cont_switch1),
        "--train-samples-refine-switch-iter",str(args.cont_switch2),
        "--diag-shift",str(args.cont_diag),
        "--diag-shift-mid",str(args.cont_diag_mid),
        "--diag-shift-refine",str(args.cont_diag_refine),
    ]


def parser():
    p=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L1",type=int,required=True)
    p.add_argument("--L2",type=int,required=True)
    p.add_argument("--seed",choices=DEFAULT_SEEDS,required=True)
    p.add_argument(
        "--mode",choices=["continuation","independent"],
        default="continuation",
    )
    p.add_argument("--direction",choices=["up","down"],default="up")
    p.add_argument("--V-values",nargs="*",type=float,default=None)
    p.add_argument("--V-min",type=float,default=None)
    p.add_argument("--V-max",type=float,default=None)
    p.add_argument("--V-step",type=float,default=None)

    p.add_argument("--t",type=float,default=1.0)
    p.add_argument("--t1",type=float,default=0.5)
    p.add_argument("--t2",type=float,default=-0.5)
    p.add_argument("--seed-amplitude",type=float,default=0.08)
    p.add_argument("--correlator",choices=["backflow"],default="backflow")
    p.add_argument("--jastrow-shells",type=int,default=4)

    p.add_argument("--anchor-iter",type=int,default=230)
    p.add_argument("--anchor-switch1",type=int,default=80)
    p.add_argument("--anchor-switch2",type=int,default=180)
    p.add_argument("--anchor-lr",type=float,default=0.03)
    p.add_argument("--anchor-lr-mid",type=float,default=0.015)
    p.add_argument("--anchor-lr-refine",type=float,default=0.005)
    p.add_argument("--anchor-diag",type=float,default=0.05)
    p.add_argument("--anchor-diag-mid",type=float,default=0.05)
    p.add_argument("--anchor-diag-refine",type=float,default=0.02)

    p.add_argument("--cont-iter",type=int,default=160)
    p.add_argument("--cont-switch1",type=int,default=50)
    p.add_argument("--cont-switch2",type=int,default=120)
    p.add_argument("--cont-lr",type=float,default=0.015)
    p.add_argument("--cont-lr-mid",type=float,default=0.0075)
    p.add_argument("--cont-lr-refine",type=float,default=0.003)
    p.add_argument("--cont-diag",type=float,default=0.05)
    p.add_argument("--cont-diag-mid",type=float,default=0.03)
    p.add_argument("--cont-diag-refine",type=float,default=0.02)

    p.add_argument("--train-samples-early",type=int,default=1024)
    p.add_argument("--train-samples-late",type=int,default=2048)
    p.add_argument("--train-samples-refine",type=int,default=4096)
    p.add_argument("--eval-samples",type=int,default=8192)
    p.add_argument("--n-chains",type=int,default=32)
    p.add_argument("--n-discard",type=int,default=16)
    p.add_argument(
        "--sweep-size",type=int,default=None,
        help="Default auto is about 2*Nf/3 (16 for 2x2).",
    )
    p.add_argument("--chunk-size",type=int,default=None)
    p.add_argument("--qgt-chunk-size",type=int,default=None)
    p.add_argument("--cg-tol",type=float,default=1e-5)
    p.add_argument("--cg-maxiter",type=int,default=100)

    p.add_argument("--rng-seed",type=int,default=2026)
    p.add_argument("--output",required=True)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--dry-run",action="store_true")
    return p


def main():
    a=parser().parse_args()
    if a.L1<1 or a.L2<1:
        raise ValueError("L1/L2 must be positive")
    values=_build_v_values(a)
    if not values:
        raise ValueError("No V values")

    nf=6*a.L1*a.L2  # fixed filling = Nf/Ns = 1/3
    ns=18*a.L1*a.L2
    chunk_auto,qgt_chunk_auto=_auto_chunks(a.L1,a.L2)
    chunk=chunk_auto if a.chunk_size is None else int(a.chunk_size)
    qgt_chunk=(
        qgt_chunk_auto
        if a.qgt_chunk_size is None
        else int(a.qgt_chunk_size)
    )
    sweep=(
        _auto_sweep(a.L1,a.L2)
        if a.sweep_size is None
        else int(a.sweep_size)
    )

    root=Path(a.output).expanduser().resolve()
    branch_root=(
        root
        /f"L{a.L1}x{a.L2}"
        /f"seed_{a.seed}"
        /f"{a.mode}_{a.direction}"
    )
    branch_root.mkdir(parents=True,exist_ok=True)

    settings={
        **vars(a),
        "Ns":ns,
        "Nf":nf,
        "filling":1.0/3.0,
        "V_scan_order":values,
        "chunk_size_effective":chunk,
        "qgt_chunk_size_effective":qgt_chunk,
        "sweep_size_effective":sweep,
        "runner":str(RAW_RUNNER),
    }
    (branch_root/"scan_settings.json").write_text(
        json.dumps(settings,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )

    rows=[]
    previous_stage=None
    for iv,V in enumerate(values):
        point_dir=branch_root/_vtag(V)
        stage_dir=_raw_stage_dir(
            point_dir,a.seed,a.correlator
        )

        if a.resume and _completed(
            point_dir,a.seed,a.correlator
        ):
            row=_read_one_row(
                point_dir,a.seed,a.correlator
            )
            row.update({
                "scan_mode":a.mode,
                "scan_direction":a.direction,
                "branch_seed":a.seed,
                "scan_index":iv,
                "scan_V":float(V),
                "scan_reused_existing":True,
            })
            rows.append(row)
            previous_stage=stage_dir
            print(
                f"[scan] reuse completed seed={a.seed} V={V:g}",
                flush=True,
            )
            continue

        anchor=(
            a.mode=="independent"
            or previous_stage is None
        )

        cmd=[
            sys.executable,str(RAW_RUNNER),
            "--L1",str(a.L1),"--L2",str(a.L2),
            "--nf",str(nf),
            "--t",str(a.t),"--t1",str(a.t1),"--t2",str(a.t2),
            "--V",str(float(V)),
            "--seeds",a.seed,
            "--seed-amplitude",str(a.seed_amplitude),
            "--correlator","backflow",
            "--jastrow-mode","bond_aware",
            "--jastrow-shells",str(a.jastrow_shells),
            "--trainer","shared_sr",
            "--train-samples-early",str(a.train_samples_early),
            "--train-samples-late",str(a.train_samples_late),
            "--train-samples-refine",str(a.train_samples_refine),
            "--n-samples",str(a.eval_samples),
            "--eval-samples",str(a.eval_samples),
            "--n-chains",str(a.n_chains),
            "--n-discard",str(a.n_discard),
            "--sweep-size",str(sweep),
            "--chunk-size",str(chunk),
            "--qgt","auto",
            "--qgt-chunk-size",str(qgt_chunk),
            "--sr-solver","cg",
            "--cg-tol",str(a.cg_tol),
            "--cg-maxiter",str(a.cg_maxiter),
            "--packed-kernel",
            "--fast-local-energy",
            "--no-fast-local-gradient-check",
            "--fast-sampler",
            "--orbital-parameterization","free",
            "--orbital-gauge-fix-interval","20",
            "--no-analytic-logderiv",
            "--no-shared-sr-check",
            "--skip-pav","--skip-lanczos",
            "--rng-seed",str(
                int(a.rng_seed)
                +100000*iv
                +(0 if a.direction=="up" else 50000000)
            ),
            "--output",str(point_dir),
        ]
        cmd.extend(_schedule_args(anchor,a))
        if (
            a.mode=="continuation"
            and previous_stage is not None
        ):
            cmd.extend([
                "--raw-warmstart-dir",
                str(previous_stage),
            ])

        print(
            "[scan] "
            f"seed={a.seed} V={V:g} "
            f"mode={a.mode} direction={a.direction} "
            f"{'anchor' if anchor else 'warm-start'}",
            flush=True,
        )
        print(" ".join(cmd),flush=True)
        if not a.dry_run:
            subprocess.run(cmd,check=True)

        if a.dry_run:
            previous_stage=stage_dir
            continue

        row=_read_one_row(
            point_dir,a.seed,a.correlator
        )
        row.update({
            "scan_mode":a.mode,
            "scan_direction":a.direction,
            "branch_seed":a.seed,
            "scan_index":iv,
            "scan_V":float(V),
            "scan_reused_existing":False,
            "scan_warmstart_from":(
                None if anchor else str(previous_stage)
            ),
        })
        rows.append(row)
        previous_stage=stage_dir

        _write_rows(branch_root/"branch_scan.csv",rows)

    if rows:
        _write_rows(branch_root/"branch_scan.csv",rows)
        print(
            f"[scan] wrote {branch_root/'branch_scan.csv'}",
            flush=True,
        )


if __name__=="__main__":
    main()
