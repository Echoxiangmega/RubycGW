#!/usr/bin/env python3
"""Optimized wrapper for the dense-k finite-source GW vs GW+SOX scan.

This script keeps all physics, continuation, retry and output logic from
``scan_primitive_finite_source_gw_sox_vs_nk.py`` but swaps in the vectorized
periodic SOX self-energy kernel.
"""
from __future__ import annotations

import scan_primitive_finite_source_gw_sox_vs_nk as scan
from rubycgw.gw_sox_fast import solve_matrix_gw_sox_fast


def main():
    scan.solve_matrix_gw_sox = solve_matrix_gw_sox_fast
    scan.main()


if __name__ == "__main__":
    main()
