#!/usr/bin/env python3
"""Replot zero-temperature susceptibility from an ED18 scan NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

from rubycgw.ed18_chi_plot import save_ed18_chi_plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", help="NPZ produced by scan_ed18_vs_V.py --with-chi")
    p.add_argument("--out", default=None)
    p.add_argument("--top-modes", type=int, default=3)
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    src = Path(args.npz)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_chi.png")
    path = save_ed18_chi_plot(src, chi_path=out, top_modes=args.top_modes, dpi=args.dpi)
    print("saved:", path)


if __name__ == "__main__":
    main()
