#!/usr/bin/env python3
"""Low-to-high finite-source continuation for ED / GW / GW+SOX / GW+sSOSEX.

The production no-Gamma scan sorts source fields from high to low.  This thin
wrapper reverses that ordering so each converged low-field solution is used as
the warm start for the next, larger field.  It is intended for hysteresis /
branch-stability diagnostics against the ordinary high-to-low scan.

For a strict reverse-branch test, start sufficiently deep on the low-current
branch (for example h_ref=0.005 or smaller) before increasing h.
"""
from __future__ import annotations

import sys
from pathlib import Path

import scan_primitive_finite_source_gw_ssosex_fast as scan


def _hvalues_low_to_high(values):
    # Reuse all validation/deduplication from the production helper, then only
    # reverse its canonical high->low ordering.
    return scan._hvalues_original(values)[::-1]


def main():
    scan._hvalues_original = scan._hvalues
    scan._hvalues = _hvalues_low_to_high

    # Keep reverse-scan products separate from the ordinary down-scan unless
    # the caller explicitly chose an output directory.
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(Path("results/primitive_gw_ssosex_reverse"))])

    scan.main()


if __name__ == "__main__":
    main()
