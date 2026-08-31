#!/usr/bin/env python3
"""Inspect order parameters of a converged zero-source 18-site GW checkpoint.

The checkpoint stores Sigma_H, Sigma_GW and mu rather than G itself.  This tool
reconstructs the zero-source Green function

    G^{-1}(k,iw) = (iw+mu) I - h0(k) - Sigma_H - Sigma_GW(k,iw),

then directly evaluates the already-defined physical loop-current expectation
values

    m_opposite = <eta_+,q=0>,
    m_same     = <eta_-,q=0>,

as well as the period-three charge-order amplitude Phi.  No cGW vertex solve is
performed: these are one-point expectation values of the converged GW state.

The q=0 current vertices use the project normalization

    K_q0 = (K_s0 + K_s1 + K_s2)/sqrt(3),

so the current per primitive cell is m_q0/sqrt(3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.checkpoint import (
    load_supercell_checkpoint,
    read_checkpoint_metadata,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.lc_branch import current_diagnostics
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, charge_order_parameter
from rubycgw.supercell_gw import dyson_from_sigma_matrix


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=str, help="Converged zero-source supercell GW checkpoint (.npz).")
    p.add_argument(
        "--current-threshold",
        type=float,
        default=1e-6,
        help="Per-primitive-cell |m| above which a current channel is labeled finite.",
    )
    p.add_argument(
        "--co-threshold",
        type=float,
        default=1e-6,
        help="|Phi| above which period-three charge order is labeled finite.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    path = Path(args.checkpoint)
    meta = read_checkpoint_metadata(path)

    if not bool(meta.get("converged", False)):
        raise ValueError(f"Checkpoint is not marked converged: {path}")

    source = float(meta.get("source", 0.0))
    if not np.isclose(source, 0.0, rtol=0.0, atol=1e-14):
        raise ValueError(
            "This inspector reconstructs the physical zero-source Hamiltonian, but the "
            f"checkpoint metadata has source={source:g}. Use an h=0 checkpoint."
        )

    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=float(meta["V"]),
    )
    grid = MatsubaraGrid(
        nk1=int(meta["nk1"]),
        nk2=int(meta["nk2"]),
        nw=int(meta["nw"]),
        nOmega=int(meta["nOmega"]),
        T=float(meta["T"]),
    )
    primitive_filling_target = float(meta["primitive_filling"])

    seed, _, stored_density = load_supercell_checkpoint(
        path,
        params,
        grid,
        primitive_filling_target,
    )
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    G = dyson_from_sigma_matrix(
        h0,
        grid,
        float(seed.mu),
        np.asarray(seed.Sigma_H, dtype=complex),
        np.asarray(seed.Sigma_GW, dtype=complex),
    )

    currents = current_diagnostics(G, grid)
    m_same = complex(currents["same_q0"])
    m_opposite = complex(currents["opposite_q0"])
    sqrt3 = np.sqrt(3.0)
    m_same_pc = m_same / sqrt3
    m_opposite_pc = m_opposite / sqrt3

    # eta_+ = (eta_A + eta_B)/sqrt(2), eta_- = (eta_A - eta_B)/sqrt(2).
    # These are algebraic A/B loop expectations; because the two triangles have
    # opposite geometric handedness, eta_- is the PHYSICAL same-circulation mode.
    eta_A_pc = (m_opposite_pc + m_same_pc) / np.sqrt(2.0)
    eta_B_pc = (m_opposite_pc - m_same_pc) / np.sqrt(2.0)

    phi = charge_order_parameter(stored_density)
    n_sc = float(np.sum(stored_density))
    n_pc = n_sc / 3.0

    print("=" * 92)
    print("18-site Ruby GW checkpoint order parameters")
    print("checkpoint:", path)
    print(
        f"V={params.V:g}, T={grid.T:g}, target primitive filling={primitive_filling_target:g}, "
        f"grid={grid.nk1}x{grid.nk2}, nw={grid.nw}, nOmega={grid.nOmega}"
    )
    print(
        f"mu={float(seed.mu):.10f}, stored residual={float(meta.get('final_error', np.nan)):.3e}, "
        f"actual primitive filling={n_pc:.10f}"
    )
    print("-" * 92)
    print(
        f"Phi_CO = {phi.real:+.12e}{phi.imag:+.12e}i,  |Phi_CO|={abs(phi):.12e}"
    )
    print(
        f"m_same(q0)     = {m_same.real:+.12e}{m_same.imag:+.12e}i,  "
        f"|m|={abs(m_same):.12e}"
    )
    print(
        f"m_opposite(q0) = {m_opposite.real:+.12e}{m_opposite.imag:+.12e}i,  "
        f"|m|={abs(m_opposite):.12e}"
    )
    print("normalized per primitive cell (divide q0 harmonic by sqrt(3)):")
    print(
        f"  m_same/pc     = {m_same_pc.real:+.12e}{m_same_pc.imag:+.12e}i,  "
        f"|m|={abs(m_same_pc):.12e}"
    )
    print(
        f"  m_opposite/pc = {m_opposite_pc.real:+.12e}{m_opposite_pc.imag:+.12e}i,  "
        f"|m|={abs(m_opposite_pc):.12e}"
    )
    print("algebraic triangle currents per primitive cell:")
    print(
        f"  eta_A = {eta_A_pc.real:+.12e}{eta_A_pc.imag:+.12e}i,  |eta_A|={abs(eta_A_pc):.12e}"
    )
    print(
        f"  eta_B = {eta_B_pc.real:+.12e}{eta_B_pc.imag:+.12e}i,  |eta_B|={abs(eta_B_pc):.12e}"
    )
    print("-" * 92)

    has_co = abs(phi) > float(args.co_threshold)
    has_same = abs(m_same_pc) > float(args.current_threshold)
    has_opposite = abs(m_opposite_pc) > float(args.current_threshold)

    if has_same and has_opposite:
        current_label = "mixed same+opposite current"
    elif has_same:
        current_label = "physical SAME-circulation LC"
    elif has_opposite:
        current_label = "physical OPPOSITE-circulation LC"
    else:
        current_label = "no resolved uniform q=0 loop current"

    if has_co:
        print(f"classification: period-3 CO + {current_label}")
    else:
        print(f"classification: no period-3 CO; {current_label}")
    print(
        f"thresholds: |Phi|>{float(args.co_threshold):.1e}, "
        f"|m|/primitive-cell>{float(args.current_threshold):.1e}"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
