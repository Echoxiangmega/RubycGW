"""Helpers for searching a spontaneous loop-current branch in the 18-site Ruby cell.

The main GW continuation follows a period-three charge-ordered solution once that
branch has been selected.  For an independent loop-current search we instead
construct a warm start with all primitive-translation-breaking self-energy
components projected out, then apply a finite uniform current source and remove
that source adiabatically.

The projection is exact in the 18-site supercell representation.  At fixed
supercell momentum k, primitive translation by a1 is represented by ``U(k)``.
A primitive-translation-invariant matrix obeys ``U M U^dagger = M``; averaging
over the three-element translation coset removes the folded +/-Q components.
Because T1=a1-a2 is already a supercell translation, invariance under a1 also
implies invariance under a2 up to the scalar Bloch phase.

Current conventions are the project conventions:

    ``opposite`` = eta_plus  = physical opposite circulation,
    ``same``     = eta_minus = physical same circulation.

The uniform q=0 current vertex is normalized as the harmonic basis used by the
cGW response code, K_q0=(K_s0+K_s1+K_s2)/sqrt(3).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .checkpoint import GWCheckpointSeed
from .grids import MatsubaraGrid
from .model import NSUB, eta_vertices
from .supercell import NSECTOR, NSUP


_PRIMITIVE_A1_WRAP_SC = np.array([2.0, 1.0], dtype=float)


def primitive_translation_a1_matrix(k_sc: np.ndarray) -> np.ndarray:
    """Return the 18x18 representation of primitive translation by ``a1``.

    With sector representatives R_s=(s,0), a1 maps s=0->1, 1->2, while
    s=2->0 crosses the supercell displacement 2*T1+T2.  The Bloch convention
    used by :func:`build_supercell_h0` gives the wrap phase
    exp[-2*pi*i*k_sc.(2,1)].
    """
    k = np.asarray(k_sc, dtype=float).reshape(2)
    phase = np.exp(-2j * np.pi * np.dot(k, _PRIMITIVE_A1_WRAP_SC))
    U = np.zeros((NSUP, NSUP), dtype=complex)
    eye6 = np.eye(NSUB, dtype=complex)
    U[NSUB : 2 * NSUB, 0:NSUB] = eye6
    U[2 * NSUB : 3 * NSUB, NSUB : 2 * NSUB] = eye6
    U[0:NSUB, 2 * NSUB : 3 * NSUB] = phase * eye6
    return U


def project_primitive_translation_invariant(
    matrix: np.ndarray,
    k_sc: np.ndarray,
) -> np.ndarray:
    """Project one or many 18x18 matrices onto the primitive q=0 sector."""
    arr = np.asarray(matrix, dtype=complex)
    if arr.shape[-2:] != (NSUP, NSUP):
        raise ValueError(f"matrix trailing shape {arr.shape[-2:]} != {(NSUP, NSUP)}")
    U = primitive_translation_a1_matrix(k_sc)
    U2 = U @ U
    Ud = U.conj().T
    U2d = U2.conj().T
    return (arr + U @ arr @ Ud + U2 @ arr @ U2d) / 3.0


def remove_charge_order_from_seed(
    seed: GWCheckpointSeed,
    grid: MatsubaraGrid,
) -> GWCheckpointSeed:
    """Remove all folded +/-Q self-energy components from a checkpoint seed.

    ``Sigma_H`` is projected at k_sc=0; for a Hartree matrix this simply makes
    symmetry-related sector onsite fields equal.  ``Sigma_GW(iw,k_sc)`` is
    projected with the correct k-dependent primitive-translation matrix.

    The operation changes only the *initial condition*.  Subsequent GW
    iterations are unconstrained and may regenerate charge order if the target
    free-energy basin prefers it.
    """
    sigma_h = np.asarray(seed.Sigma_H, dtype=complex)
    sigma_gw = np.asarray(seed.Sigma_GW, dtype=complex)
    expected_gw = (grid.nf, grid.nk1, grid.nk2, NSUP, NSUP)
    if sigma_h.shape != (NSUP, NSUP):
        raise ValueError(f"Sigma_H shape {sigma_h.shape} != {(NSUP, NSUP)}")
    if sigma_gw.shape != expected_gw:
        raise ValueError(f"Sigma_GW shape {sigma_gw.shape} != {expected_gw}")

    sigma_h_q0 = project_primitive_translation_invariant(
        sigma_h, np.array([0.0, 0.0])
    )
    sigma_h_q0 = 0.5 * (sigma_h_q0 + sigma_h_q0.conj().T)

    kmesh = grid.kmesh()
    sigma_gw_q0 = np.empty_like(sigma_gw)
    for ik1 in range(grid.nk1):
        for ik2 in range(grid.nk2):
            sigma_gw_q0[:, ik1, ik2] = project_primitive_translation_invariant(
                sigma_gw[:, ik1, ik2], kmesh[ik1, ik2]
            )

    return GWCheckpointSeed(
        Sigma_H=sigma_h_q0,
        Sigma_GW=sigma_gw_q0,
        mu=float(seed.mu),
    )


def current_vertex_q0(channel: str) -> np.ndarray:
    """Return the normalized uniform 18-site current vertex for one channel."""
    key = str(channel).strip().lower()
    _, _, kp, km = eta_vertices()
    if key == "opposite":
        k6 = kp
    elif key == "same":
        k6 = km
    else:
        raise ValueError("current channel must be 'same' or 'opposite'")

    K = np.zeros((NSUP, NSUP), dtype=complex)
    for s in range(NSECTOR):
        sl = slice(NSUB * s, NSUB * (s + 1))
        K[sl, sl] = k6 / np.sqrt(float(NSECTOR))
    return K


def add_current_source(
    h0: np.ndarray,
    strength: float,
    channel: str,
) -> np.ndarray:
    """Return ``h0 - h K_channel,q0`` without modifying the input array."""
    arr = np.asarray(h0, dtype=complex)
    if arr.shape[-2:] != (NSUP, NSUP):
        raise ValueError("unexpected supercell h0 shape")
    K = current_vertex_q0(channel)
    out = np.array(arr, copy=True)
    out -= float(strength) * K
    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


def current_expectation_q0(
    G: np.ndarray,
    grid: MatsubaraGrid,
    channel: str,
) -> complex:
    """Return <eta_channel,q0> from the Matsubara Green function.

    Since the current vertex has zero diagonal, the high-frequency integrand
    decays as 1/omega^2 and the finite Matsubara sum is absolutely convergent.
    With the source convention H_source=-h K, this definition is consistent
    with chi=-int Tr[K G Gamma G] used by the response code.
    """
    arr = np.asarray(G, dtype=complex)
    expected = (grid.nf, grid.nk1, grid.nk2, NSUP, NSUP)
    if arr.shape != expected:
        raise ValueError(f"G shape {arr.shape} != {expected}")
    K = current_vertex_q0(channel)
    value = (grid.T / grid.nk) * np.einsum(
        "ab,nxyba->", K, arr, optimize=True
    )
    return complex(value)


def current_diagnostics(G: np.ndarray, grid: MatsubaraGrid) -> dict[str, complex]:
    """Return uniform physical-same and physical-opposite current amplitudes."""
    return {
        "opposite_q0": current_expectation_q0(G, grid, "opposite"),
        "same_q0": current_expectation_q0(G, grid, "same"),
    }


@dataclass(frozen=True)
class SeedProjectionDiagnostics:
    sigma_h_removed_max: float
    sigma_gw_removed_max: float
    sigma_h_removed_rms: float
    sigma_gw_removed_rms: float


def seed_projection_diagnostics(
    original: GWCheckpointSeed,
    projected: GWCheckpointSeed,
) -> SeedProjectionDiagnostics:
    """Quantify how much primitive-Q content was removed from a warm start."""
    dh = np.asarray(original.Sigma_H) - np.asarray(projected.Sigma_H)
    dg = np.asarray(original.Sigma_GW) - np.asarray(projected.Sigma_GW)
    return SeedProjectionDiagnostics(
        sigma_h_removed_max=float(np.max(np.abs(dh))),
        sigma_gw_removed_max=float(np.max(np.abs(dg))),
        sigma_h_removed_rms=float(np.sqrt(np.mean(np.abs(dh) ** 2))),
        sigma_gw_removed_rms=float(np.sqrt(np.mean(np.abs(dg) ** 2))),
    )


__all__ = [
    "SeedProjectionDiagnostics",
    "add_current_source",
    "current_diagnostics",
    "current_expectation_q0",
    "current_vertex_q0",
    "primitive_translation_a1_matrix",
    "project_primitive_translation_invariant",
    "remove_charge_order_from_seed",
    "seed_projection_diagnostics",
]
