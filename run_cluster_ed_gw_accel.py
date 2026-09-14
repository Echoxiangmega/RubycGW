#!/usr/bin/env python3
"""Run cluster-ED+GW with scale-invariant late-stage Pulay acceleration.

This is a drop-in launcher for ``run_cluster_ed_gw.py``.  It keeps all command
line arguments and physics unchanged, but replaces the Pulay coefficient solve
with a residual-scale-invariant form before importing the production driver.

Use exactly the same arguments as for ``run_cluster_ed_gw.py``::

    python run_cluster_ed_gw_accel.py --Lx 2 --Ly 1 --V 1.0 --filling 2 \
        --embed-mixing-method pulay --embed-pulay-history 8

The main benefit appears after the residual has already become small.  The
historical absolute regularization floor can then dominate the residual Gram
matrix and turn DIIS into an almost linearly convergent iteration.
"""
from __future__ import annotations

from rubycgw.pulay_accel import install_scale_invariant_pulay


install_scale_invariant_pulay()

from run_cluster_ed_gw import main  # noqa: E402


if __name__ == "__main__":
    main()
