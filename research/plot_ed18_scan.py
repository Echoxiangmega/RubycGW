#!/usr/bin/env python3
"""Replot an existing 18-site Ruby ED V-scan NPZ without rerunning Lanczos."""

from __future__ import annotations

import argparse
from pathlib import Path

from rubycgw.ed18_plot import save_ed18_plots


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="NPZ produced by scan_ed18_vs_V.py")
    p.add_argument("--summary", default=None, help="Summary PNG path.")
    p.add_argument("--modes", default=None, help="Channel-resolved modes PNG path.")
    p.add_argument("--top-spectrum", type=int, default=6)
    p.add_argument("--top-structure", type=int, default=3)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main():
    args = _args()
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)
    summary = Path(args.summary) if args.summary else inp.with_suffix(".png")
    modes = (
        Path(args.modes)
        if args.modes
        else inp.with_name(inp.stem + "_modes.png")
    )
    out1, out2 = save_ed18_plots(
        inp,
        summary_path=summary,
        modes_path=modes,
        top_spectrum=args.top_spectrum,
        top_structure=args.top_structure,
        dpi=args.dpi,
    )
    print("saved:", out1)
    print("saved:", out2)


if __name__ == "__main__":
    main()
