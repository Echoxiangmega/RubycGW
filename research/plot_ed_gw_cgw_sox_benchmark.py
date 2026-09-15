#!/usr/bin/env python3
"""Plot a saved ED/GG/cGW/cGW+SOX/post-GW benchmark.npz file."""
from __future__ import annotations

import argparse
from pathlib import Path

from rubycgw.benchmark_plot import plot_benchmark_npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", nargs="?", type=Path,
                   default=Path("results/ed_gw_cgw_sox_benchmark/benchmark.npz"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    made = plot_benchmark_npz(args.npz, outdir=args.out)
    for path in made:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
