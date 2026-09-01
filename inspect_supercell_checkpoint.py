#!/usr/bin/env python3
"""Inspect order parameters of a converged zero-source 18-site GW checkpoint.

The checkpoint stores Sigma_H, Sigma_GW and mu rather than G itself.  This tool
reconstructs the zero-source Green function

    G^{-1}(k,iw) = (iw+mu) I - h0(k) - Sigma_H - Sigma_GW(k,iw),

then evaluates the physical uniform loop-current expectation values and a
complete set of density diagnostics inside the 18-site Q=(1/3,1/3) supercell.

Charge diagnostics distinguish:

* the previously selected period-three projection Phi;
* generic period-three translation breaking Delta_Q, independent of form factor;
* q=0 intra-triangle charge disproportionation Delta_A and Delta_B;
* q=0 mean charge imbalance Delta_AB between the two triangles.

Thus Phi=0 by itself is no longer interpreted as absence of charge order.
No cGW vertex solve is performed: these are one-point expectation values of the
converged GW state.

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
from rubycgw.supercell import (
    build_supercell_h0,
    charge_order_diagnostics,
)
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
        help=(
            "Charge-order threshold applied to Delta_Q, Delta_A, Delta_B and |Delta_AB|. "
            "The selected |Phi| is reported separately."
        ),
    )
    return p.parse_args()


def _complex_line(values: np.ndarray) -> str:
    return "  ".join(f"{z.real:+.6e}{z.imag:+.6e}i" for z in np.asarray(values, dtype=complex))


def main():
    args = _parse_args()
    if args.current_threshold <= 0.0 or args.co_threshold <= 0.0:
        raise ValueError("thresholds must be positive")

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

    charge = charge_order_diagnostics(stored_density)
    phi = complex(charge["Phi"])
    delta_Q = float(charge["Delta_Q"])
    delta_trans = float(charge["Delta_translation_rms"])
    delta_A = float(charge["Delta_A"])
    delta_B = float(charge["Delta_B"])
    delta_AB = float(charge["Delta_AB"])
    n_q0 = np.asarray(charge["n_q0"], dtype=float)
    n_Q = np.asarray(charge["n_Q"], dtype=complex)

    n_sc = float(np.sum(stored_density))
    n_pc = n_sc / 3.0

    print("=" * 108)
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
    print("-" * 108)
    print("charge diagnostics:")
    print(
        f"  selected Phi = {phi.real:+.12e}{phi.imag:+.12e}i,  |Phi|={abs(phi):.12e}"
    )
    print(f"  Delta_Q                 = {delta_Q:.12e}   (generic period-3 translation breaking)")
    print(f"  Delta_translation_rms   = {delta_trans:.12e}")
    print(f"  Delta_A                 = {delta_A:.12e}   (q=0 internal CO on triangle A: sites 0,1,2)")
    print(f"  Delta_B                 = {delta_B:.12e}   (q=0 internal CO on triangle B: sites 3,4,5)")
    print(f"  Delta_AB                = {delta_AB:+.12e}   (mean A density - mean B density)")
    print("  n_q0[a] (sector-averaged primitive-cell densities):")
    print("    " + "  ".join(f"a{a}={x:.10f}" for a, x in enumerate(n_q0)))
    print("  n_Q[a] = (1/3) sum_s exp(-2 pi i s/3) n[s,a]:")
    print("    " + _complex_line(n_Q))

    print("-" * 108)
    print("uniform q=0 loop currents:")
    print(
        f"  m_same(q0)     = {m_same.real:+.12e}{m_same.imag:+.12e}i,  "
        f"|m|={abs(m_same):.12e}"
    )
    print(
        f"  m_opposite(q0) = {m_opposite.real:+.12e}{m_opposite.imag:+.12e}i,  "
        f"|m|={abs(m_opposite):.12e}"
    )
    print("  normalized per primitive cell (divide q0 harmonic by sqrt(3)):")
    print(
        f"    m_same/pc     = {m_same_pc.real:+.12e}{m_same_pc.imag:+.12e}i,  "
        f"|m|={abs(m_same_pc):.12e}"
    )
    print(
        f"    m_opposite/pc = {m_opposite_pc.real:+.12e}{m_opposite_pc.imag:+.12e}i,  "
        f"|m|={abs(m_opposite_pc):.12e}"
    )
    print("  algebraic triangle currents per primitive cell:")
    print(
        f"    eta_A = {eta_A_pc.real:+.12e}{eta_A_pc.imag:+.12e}i,  |eta_A|={abs(eta_A_pc):.12e}"
    )
    print(
        f"    eta_B = {eta_B_pc.real:+.12e}{eta_B_pc.imag:+.12e}i,  |eta_B|={abs(eta_B_pc):.12e}"
    )
    print("-" * 108)

    cthr = float(args.co_threshold)
    has_Q = delta_Q > cthr
    has_A = delta_A > cthr
    has_B = delta_B > cthr
    has_AB = abs(delta_AB) > cthr
    selected_phi = abs(phi) > cthr
    has_same = abs(m_same_pc) > float(args.current_threshold)
    has_opposite = abs(m_opposite_pc) > float(args.current_threshold)

    charge_labels = []
    if has_Q:
        charge_labels.append("period-3 Q-CO")
    if has_A:
        charge_labels.append("q0 intra-A CO")
    if has_B:
        charge_labels.append("q0 intra-B CO")
    if has_AB:
        charge_labels.append("q0 A/B imbalance")

    current_labels = []
    if has_same:
        current_labels.append("physical SAME-circulation LC")
    if has_opposite:
        current_labels.append("physical OPPOSITE-circulation LC")

    pieces = charge_labels + current_labels
    classification = " + ".join(pieces) if pieces else "normal (no resolved charge or uniform-current order)"
    print("classification:", classification)
    print(
        f"selected Phi above threshold: {selected_phi}  "
        "(reported as a form-factor diagnostic, not used alone to decide whether charge order exists)"
    )
    print(
        f"thresholds: charge amplitudes>{cthr:.1e}, "
        f"|m|/primitive-cell>{float(args.current_threshold):.1e}"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
