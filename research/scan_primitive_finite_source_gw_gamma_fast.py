#!/usr/bin/env python3
"""Pulay/DIIS accelerated wrapper for the primitive GW-Gamma_P scan.

It preserves the CLI and output/continuation logic of
``scan_primitive_finite_source_gw_gamma.py`` while replacing only the Gamma_P
outer solver/options with :mod:`rubycgw.hedin_gamma_fast`.

The fast solver uses Pulay by default, reuses density-vertex solutions between
outer iterations, avoids a redundant final BSE solve, and prints per-iteration
timing diagnostics.
"""
import scan_primitive_finite_source_gw_gamma as scan

from rubycgw.hedin_gamma_fast import (
    GammaPFeedbackOptions,
    solve_matrix_gw_gamma_feedback,
)

scan.GammaPFeedbackOptions = GammaPFeedbackOptions
scan.solve_matrix_gw_gamma_feedback = solve_matrix_gw_gamma_feedback


if __name__ == "__main__":
    scan.main()
