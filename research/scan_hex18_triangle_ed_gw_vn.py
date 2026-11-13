#!/usr/bin/env python3
"""Scan the hex18 ordered phase diagram with two independent 3-site ED impurities
per primitive cell (A and B), each with three bath orbitals by default.

This is a convenience entry point for the partition-aware ordered-phase solver.
It uses the COMPLETE 18-site hopping lattice, including all intertriangle t1/t2
hoppings. Only the interacting impurity blocks are reduced to 3 sites.
For historical six-site clusters, use scan_hex18_cluster_order_vn.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None,""):
    root=str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0,root)

from research.scan_hex18_cluster_order_vn import main as ordered_main


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if "--cluster-partition" in arguments:
        raise ValueError(
            "this entry point is fixed to triangle3; use "
            "scan_hex18_cluster_order_vn.py for other partitions"
        )
    return ordered_main(["--cluster-partition","triangle3",*arguments])


if __name__=="__main__":
    main()
