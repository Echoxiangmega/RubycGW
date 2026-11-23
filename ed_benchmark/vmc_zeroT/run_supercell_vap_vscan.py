#!/usr/bin/env python3
"""Parallel-by-seed Fast-VAP driver for existing Ruby Raw V-scan checkpoints.

One SLURM array task handles ONE seed selected from --seeds; within that task,
requested V values are processed serially. Every V point reads its own
previously optimized Raw checkpoint. There is NO VAP continuation between
different V points (unlike the original Raw V-scan).

Example (single value):
    python ed_benchmark/vmc_zeroT/run_supercell_vap_vscan.py \
        --L1 3 --L2 3 --nf 54 \
        --t 1 --t1 -0.5 --t2 -0.5 \
        --V-values 3.6 \
        --seeds sd isd lc_same lc_opposite \
        --array-index 0 \
        --scan-root results/vscan_3x3_down \
        --output-root results/vap_3x3_down_array \
        --direction down \
        --tint-iter 20 --spatial-iter 15 --tr-iter 15 \
        --sampler standard --sweep-size 54 --spatial-sweep-size 27 \
        --samples 1024 --spatial-samples 512

All unrecognized options are passed unchanged to run_supercell_vap_fast.py.
Use --dry-run to inspect paths and commands without starting VAP or NetKet.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FAST_VAP = HERE / "run_supercell_vap_fast.py"


def vtag(value: float) -> str:
    """Match run_supercell_vscan.py's Raw V directory naming exactly."""
    sign = "m" if value < 0 else "p"
    digits = f"{abs(float(value)):.8f}".rstrip("0").rstrip(".")
    return "V_" + sign + digits.replace(".", "p")


def requested_values(opt: argparse.Namespace) -> list[float]:
    has_range = any(v is not None for v in (opt.V_min, opt.V_max, opt.V_step))
    n_sources = int(bool(opt.V_values)) + int(opt.V is not None) + int(has_range)
    if n_sources != 1:
        raise ValueError("Choose exactly one: --V, --V-values, or all of --V-min/--V-max/--V-step")
    if opt.V is not None:
        values = [float(opt.V)]
    elif opt.V_values:
        values = [float(v) for v in opt.V_values]
    else:
        if any(v is None for v in (opt.V_min, opt.V_max, opt.V_step)):
            raise ValueError("Range requires --V-min, --V-max and --V-step")
        lo, hi, step = float(opt.V_min), float(opt.V_max), float(opt.V_step)
        if not all(math.isfinite(x) for x in (lo, hi, step)):
            raise ValueError("V range must be finite")
        if step <= 0 or hi < lo:
            raise ValueError("Require --V-step > 0 and --V-max >= --V-min")
        count = int(math.floor((hi - lo) / step + 1e-10))
        values = [lo + i * step for i in range(count + 1)]
        if values[-1] < hi - 1e-9:
            values.append(hi)
    values = sorted({round(float(v), 12) for v in values})
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("No valid V values selected")
    if opt.direction == "down":
        values.reverse()
    return values


def select_seed(opt: argparse.Namespace) -> tuple[int, str]:
    if len(set(opt.seeds)) != len(opt.seeds):
        raise ValueError("Duplicate seeds would cause output-directory collisions")
    index = opt.array_index
    if index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is not None:
            try:
                index = int(env_index)
            except ValueError as exc:
                raise ValueError("SLURM_ARRAY_TASK_ID is not an integer") from exc
        elif len(opt.seeds) == 1:
            index = 0
        else:
            raise ValueError(
                "For multiple --seeds, specify --array-index N or submit a SLURM array "
                "(SLURM_ARRAY_TASK_ID). One task runs one seed."
            )
    if not 0 <= index < len(opt.seeds):
        raise ValueError(
            f"array index {index} is outside 0..{len(opt.seeds)-1} "
            f"for seeds {opt.seeds}"
        )
    return index, opt.seeds[index]


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--seeds", nargs="+", required=True,
                   help="Seed list; each SLURM array task selects one index")
    p.add_argument("--array-index", type=int, default=None,
                   help="0-based seed index; default is SLURM_ARRAY_TASK_ID")
    p.add_argument("--L1", type=int, required=True)
    p.add_argument("--L2", type=int, required=True)
    p.add_argument("--nf", type=int, default=None)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--t1", type=float, default=0.5)
    p.add_argument("--t2", type=float, default=-0.5)
    p.add_argument("--V", type=float, default=None)
    p.add_argument("--V-values", nargs="+", type=float, default=None)
    p.add_argument("--V-min", type=float, default=None)
    p.add_argument("--V-max", type=float, default=None)
    p.add_argument("--V-step", type=float, default=None)
    p.add_argument("--mode", choices=("continuation", "independent"),
                   default="continuation")
    p.add_argument("--direction", choices=("up", "down"), default="down")
    p.add_argument("--scan-root", type=Path, required=True,
                   help="Root of existing Raw V-scan, e.g. results/vscan_3x3_down")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Separate output root for Fast-VAP results")
    p.add_argument("--missing-raw", choices=("skip", "error"), default="skip",
                   help="Skip missing Raw V points or fail immediately")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue to next V after a VAP subprocess error; exit nonzero at end")
    p.add_argument("--dry-run", action="store_true",
                   help="Show selected seed, paths and commands without running")
    opt, passthrough = p.parse_known_args(argv)

    # Never permit forwarded options to override the per-array-task identity
    # or the input/output checkpoints resolved by this driver.
    reserved = {
        "--seed", "--seeds", "--seed-index", "--array-index",
        "--L1", "--L2", "--nf", "--V", "--V-values", "--V-min",
        "--V-max", "--V-step", "--t", "--t1", "--t2",
        "--raw-stage", "--scan-root", "--output", "--output-root",
        "--mode", "--direction", "--random-seed", "--dry-run",
    }
    for arg in passthrough:
        if arg.split("=", 1)[0] in reserved:
            raise ValueError(f"Duplicate/unsupported wrapper argument in VAP passthrough: {arg}")
    if any(x <= 0 for x in (opt.L1, opt.L2)) or (opt.nf is not None and opt.nf <= 0):
        raise ValueError("L1, L2 and nf must be positive")

    return opt, passthrough


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run(argv: list[str] | None = None) -> int:
    opt, passthrough = parse_args(argv)
    idx, seed = select_seed(opt)
    values = requested_values(opt)
    nf = opt.nf if opt.nf is not None else 6 * opt.L1 * opt.L2
    raw_root = opt.scan_root.expanduser().resolve()
    output_root = opt.output_root.expanduser().resolve()
    branch_subdir = (Path(f"L{opt.L1}x{opt.L2}") /
                     f"seed_{seed}" / f"{opt.mode}_{opt.direction}")
    branch = output_root / branch_subdir
    if not FAST_VAP.is_file():
        raise FileNotFoundError(f"Fast-VAP runner not found: {FAST_VAP}")

    print(f"[array] index={idx}/{len(opt.seeds)-1} selected_seed={seed}", flush=True)
    print(f"[array] V order={values}", flush=True)
    print(f"[array] raw_root={raw_root}", flush=True)
    print(f"[array] output_branch={branch}", flush=True)
    print("[array] Each V reads its own existing Raw checkpoint; "
          "no cross-V VAP warm-start.", flush=True)
    if not opt.dry_run:
        branch.mkdir(parents=True, exist_ok=True)
        (branch / "vap_scan_settings.json").write_text(
            json.dumps({
                "seeds": opt.seeds, "array_index": idx, "selected_seed": seed,
                "V_values": values, "L1": opt.L1, "L2": opt.L2, "nf": nf,
                "t": opt.t, "t1": opt.t1, "t2": opt.t2,
                "mode": opt.mode, "direction": opt.direction,
                "raw_root": str(raw_root), "output_branch": str(branch),
                "fast_vap_passthrough": passthrough,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    rows = []
    found_raw = 0
    failures = 0
    for iv, v in enumerate(values):
        tag = vtag(v)
        raw_stage = (raw_root / branch_subdir / tag /
                     f"large_raw_backflow__{seed}")
        output = branch / tag
        checkpoint = raw_stage / "final_params.npz"
        row = {
            "array_index": idx, "seed": seed, "V": v,
            "raw_stage": str(raw_stage), "output": str(output),
            "status": None, "return_code": None,
        }
        if not checkpoint.is_file():
            row["status"] = "missing_raw"
            rows.append(row)
            print(f"[array] {seed} V={v:g} missing {checkpoint}", flush=True)
            if not opt.dry_run:
                write_rows(branch / "vap_scan_status.csv", rows)
            if opt.missing_raw == "error":
                raise FileNotFoundError(str(checkpoint))
            continue

        found_raw += 1
        cmd = [
            sys.executable, "-u", str(FAST_VAP),
            "--L1", str(opt.L1), "--L2", str(opt.L2),
            "--nf", str(nf),
            "--t", str(opt.t), "--t1", str(opt.t1), "--t2", str(opt.t2),
            "--V", str(v), "--seed", seed,
            "--raw-stage", str(raw_stage),
            "--output", str(output),
            *passthrough,
        ]
        print(f"[array] {seed} V={v:g} ({iv+1}/{len(values)}): "
              + shlex.join(cmd), flush=True)
        if opt.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        try:
            subprocess.run(cmd, check=True)
            row["status"] = "ok"
            row["return_code"] = 0
        except subprocess.CalledProcessError as exc:
            failures += 1
            row["status"] = "failed"
            row["return_code"] = int(exc.returncode)
            print(f"[array] FAILED seed={seed} V={v:g} exit={exc.returncode}",
                  flush=True)
        rows.append(row)
        write_rows(branch / "vap_scan_status.csv", rows)
        if row["status"] == "failed" and not opt.keep_going:
            print("[array] Stopping this seed's V sequence after first failure; "
                  "other SLURM array tasks are unaffected.", flush=True)
            return 1

    if found_raw == 0:
        print("[array] ERROR: no Raw checkpoint found for selected seed and "
              "V values. Check --scan-root, --mode and --direction.",
              file=sys.stderr, flush=True)
        return 2
    print(f"[array] seed={seed} finished: "
          f"{sum(r['status'] == 'ok' for r in rows)} completed, "
          f"{sum(r['status'] == 'missing_raw' for r in rows)} missing, "
          f"{failures} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
