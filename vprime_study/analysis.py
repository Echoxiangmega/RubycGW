"""Post-processing helpers for V-prime all-q JF response files."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _idx(channels, name: str) -> int:
    names = [str(x) for x in np.asarray(channels).tolist()]
    if name not in names:
        raise KeyError(f"missing channel {name!r}; available={names}")
    return names.index(name)


def _mode_label(vec: np.ndarray, channels) -> tuple[str, str, float]:
    names = [str(x) for x in np.asarray(channels).tolist()]
    c = {name: complex(vec[i]) for i, name in enumerate(names)}
    comp_weights = {}
    parity_amp = {}
    for comp in "xyz":
        A = c.get(f"A{comp}", 0.0j)
        B = c.get(f"B{comp}", 0.0j)
        comp_weights[comp] = abs(A) ** 2 + abs(B) ** 2
        parity_amp[(comp, "even")] = abs((A + B) / np.sqrt(2.0))
        parity_amp[(comp, "odd")] = abs((A - B) / np.sqrt(2.0))
    comp = max(comp_weights, key=comp_weights.get)
    parity = "even" if parity_amp[(comp, "even")] >= parity_amp[(comp, "odd")] else "odd"
    if comp == "z":
        parity = "same" if parity == "even" else "opposite"
    return comp, parity, float(comp_weights[comp])


def _projection_vector(channels, comp: str, parity: str) -> np.ndarray:
    nc = len(channels)
    v = np.zeros(nc, dtype=complex)
    ia = _idx(channels, f"A{comp}")
    ib = _idx(channels, f"B{comp}")
    v[ia] = 1.0 / np.sqrt(2.0)
    v[ib] = (1.0 if parity == "even" else -1.0) / np.sqrt(2.0)
    return v


def _expect(chi: np.ndarray, v: np.ndarray) -> float:
    return float(np.real(np.vdot(v, np.asarray(chi, dtype=complex) @ v)))


def summarize_response(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        channels = np.asarray(z["channels"])
        q = np.asarray(z["q_centered"], dtype=float)
        chi = np.asarray(z["chi_hermitian"], dtype=complex)
        eigvals = np.asarray(z["eigenvalues"], dtype=float)
        eigvecs = np.asarray(z["eigenvectors"], dtype=complex)
        V = float(np.asarray(z["V"]).reshape(()))
        filling = float(np.asarray(z["filling"]).reshape(()))
        T = float(np.asarray(z["T"]).reshape(()))
        if "Vprime" in z:
            vp = float(np.asarray(z["Vprime"]).reshape(()))
        elif "Vp" in z:
            vp = float(np.asarray(z["Vp"]).reshape(()))
        else:
            vp = 0.0

    lead = eigvals[:, 0]
    ig = int(np.argmax(lead))
    comp, parity, weight = _mode_label(eigvecs[ig, :, 0], channels)

    gamma_candidates = np.flatnonzero(np.linalg.norm(q, axis=1) < 1e-12)
    gamma = int(gamma_candidates[0]) if gamma_candidates.size else -1

    z_same_v = _projection_vector(channels, "z", "even")
    z_opp_v = _projection_vector(channels, "z", "odd")
    z_same = np.asarray([_expect(m, z_same_v) for m in chi])
    z_opp = np.asarray([_expect(m, z_opp_v) for m in chi])

    xy_indices = [_idx(channels, x) for x in ("Ax", "Ay", "Bx", "By")]
    xy_lambda = np.empty(len(q), dtype=float)
    for iq, m in enumerate(chi):
        block = np.asarray(m)[np.ix_(xy_indices, xy_indices)]
        xy_lambda[iq] = float(np.max(np.linalg.eigvalsh(0.5 * (block + block.conj().T))).real)

    izs = int(np.argmax(z_same))
    izo = int(np.argmax(z_opp))
    ixy = int(np.argmax(xy_lambda))

    row = {
        "file": str(path),
        "V": V,
        "Vprime": vp,
        "filling": filling,
        "T": T,
        "global_lambda": float(lead[ig]),
        "global_q1": float(q[ig, 0]),
        "global_q2": float(q[ig, 1]),
        "global_component": comp,
        "global_parity": parity,
        "global_component_weight": weight,
        "z_same_max": float(z_same[izs]),
        "z_same_q1": float(q[izs, 0]),
        "z_same_q2": float(q[izs, 1]),
        "z_opposite_max": float(z_opp[izo]),
        "z_opposite_q1": float(q[izo, 0]),
        "z_opposite_q2": float(q[izo, 1]),
        "xy_max": float(xy_lambda[ixy]),
        "xy_q1": float(q[ixy, 0]),
        "xy_q2": float(q[ixy, 1]),
        "z_same_gamma": float(z_same[gamma]) if gamma >= 0 else np.nan,
        "z_opposite_gamma": float(z_opp[gamma]) if gamma >= 0 else np.nan,
        "xy_gamma": float(xy_lambda[gamma]) if gamma >= 0 else np.nan,
    }
    return row
