#!/usr/bin/env python3
"""Compare all-q JF response files across attractive V'."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vprime_study.analysis import summarize_response


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    return p.parse_args()


def main():
    args = _args()
    rows = [summarize_response(p) for p in args.files]
    rows.sort(key=lambda r: r["Vprime"])

    header = (
        "V'        global(lambda,q,mode)                         "
        "z_same@G   z_same(max,q)          z_opp@G    xy(max,q)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        gmode = f"{r['global_component']}_{r['global_parity']}"
        print(
            f"{r['Vprime']:+.4f}  "
            f"{r['global_lambda']:9.5f} ({r['global_q1']:+.3f},{r['global_q2']:+.3f}) "
            f"{gmode:12s}  "
            f"{r['z_same_gamma']:9.5f}  "
            f"{r['z_same_max']:9.5f} ({r['z_same_q1']:+.3f},{r['z_same_q2']:+.3f})  "
            f"{r['z_opposite_gamma']:9.5f}  "
            f"{r['xy_max']:9.5f} ({r['xy_q1']:+.3f},{r['xy_q2']:+.3f})"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
