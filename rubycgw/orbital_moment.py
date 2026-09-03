"""Checkpoint post-processing for local orbital moments on the Ruby supercell.

This module deliberately separates two quantities:

1. the *local plaquette orbital moment* extracted from an already-computed
   interacting one-body Green function; and
2. the strict periodic bulk orbital magnetization / magnetic response, which
   requires an electromagnetic vertex (for example a covariant derivative with
   respect to a vector potential or plaquette flux).

The present implementation covers (1).  It reconstructs the zero-source
Matsubara Green function from an 18-site GW checkpoint, evaluates the
equal-time one-body density matrix, projects the three oriented bonds of each
elementary triangle onto a circulating current, and converts that loop current
to a plaquette moment.

For a directed hopping I -> J with Hamiltonian matrix element t_IJ, define the
phase-conjugate current

    j_phi(I->J) = < dH / dphi_IJ >
                = -2 Im[t_IJ <c_I^dagger c_J>].

It has units of the model energy.  For a closed loop p,

    m_p^(code) = j_loop * A_p,

where A_p is the signed plaquette area in the real-space embedding.  This is
equivalent to (1/2) sum_bond j_bond (r_i x r_j)_z when the bond currents form
an exactly conserved loop.

If the hopping-energy unit E0 and lattice constant a are supplied, the physical
moment is reported using the electron Peierls convention

    phi_ij = -(e/hbar) integral_i^j A.dl,

so that

    m_p = (e/hbar) E0 a^2 m_p^(code).

The repository model is spinless; ``spin_degeneracy=1`` is therefore the
default.

This local marker is not the Bianco-Resta bulk orbital-magnetization marker and
does not replace the interacting Green-function/vertex formula for a periodic
bulk magnetization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint import load_supercell_checkpoint, read_checkpoint_metadata
from .grids import MatsubaraGrid
from .model import RubyParameters
from .supercell import (
    NSUB,
    NSUP,
    NSECTOR,
    SUPERCELL_MATRIX,
    SUPERCELL_REPRESENTATIVES,
    build_supercell_h0,
    supercell_hoppings,
)
from .supercell_gw import dyson_from_sigma_matrix


ELEMENTARY_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
EV_J = ELEMENTARY_CHARGE_C
ANGSTROM_M = 1.0e-10
BOHR_MAGNETON_J_T = 9.2740100657e-24

TRIANGLE_LABELS = ("A", "B")
_TRIANGLE_LOCAL_ORIENTED = {
    "A": (0, 1, 2),
    "B": (3, 4, 5),
}


@dataclass
class OrbitalMomentResult:
    """Local orbital-moment analysis for one 18-site GW checkpoint.

    Array conventions
    -----------------
    ``phase_bond_currents``
        Shape ``(3, 2, 3)``.  The last axis follows the oriented triangle
        edges A: 0->1->2->0 and B: 3->4->5->3.
    ``loop_phase_currents``
        Shape ``(3, 2)``.  Mean oriented phase-conjugate current on each
        triangle.
    ``plaquette_moments_code``
        Shape ``(3, 2)`` in model-energy times lattice-length squared.
        The real-space orientation is included, so A and B automatically carry
        the correct physical handedness.
    """

    metadata: dict
    density_matrix: np.ndarray
    phase_bond_currents: np.ndarray
    loop_phase_currents: np.ndarray
    loop_current_spread: np.ndarray
    signed_triangle_areas: np.ndarray
    plaquette_moments_code: np.ndarray
    cell_net_moments_code: np.ndarray
    cell_staggered_moments_code: np.ndarray
    supercell_net_moment_code: float
    plaquette_moments_muB: np.ndarray | None = None
    cell_net_moments_muB: np.ndarray | None = None
    cell_staggered_moments_muB: np.ndarray | None = None
    supercell_net_moment_muB: float | None = None
    loop_charge_currents_A: np.ndarray | None = None


def primitive_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return primitive lattice vectors and six orbital fractional positions.

    This is exactly the real-space embedding used by
    ``plot_supercell_order_realspace.py``.
    """
    lat = np.array(
        [
            [np.sqrt(3.0) / 2.0, 0.5],
            [-np.sqrt(3.0) / 2.0, 0.5],
        ],
        dtype=float,
    )
    dis = 1.0 / 8.0
    orb = np.array(
        [
            [1.0 / 3.0 - dis, 2.0 / 3.0 + dis],
            [1.0 / 3.0 + 2.0 * dis, 2.0 / 3.0 + dis],
            [1.0 / 3.0 - dis, 2.0 / 3.0 - 2.0 * dis],
            [2.0 / 3.0 + dis, 1.0 / 3.0 - dis],
            [2.0 / 3.0 + dis, 1.0 / 3.0 + 2.0 * dis],
            [2.0 / 3.0 - 2.0 * dis, 1.0 / 3.0 - dis],
        ],
        dtype=float,
    )
    return lat, orb


def primitive_to_cart(frac: np.ndarray) -> np.ndarray:
    lat, _ = primitive_geometry()
    return np.asarray(frac, dtype=float) @ lat


def supercell_site_positions() -> np.ndarray:
    """Return 18 real-space positions in the ordering I=6*s+a."""
    _, orb = primitive_geometry()
    out = np.zeros((NSUP, 2), dtype=float)
    for s, Rs in enumerate(SUPERCELL_REPRESENTATIVES):
        for a in range(NSUB):
            out[NSUB * s + a] = primitive_to_cart(Rs + orb[a])
    return out


def supercell_vectors_cart() -> tuple[np.ndarray, np.ndarray]:
    """Return the two supercell translation vectors in Cartesian coordinates."""
    T1 = primitive_to_cart(SUPERCELL_MATRIX[:, 0])
    T2 = primitive_to_cart(SUPERCELL_MATRIX[:, 1])
    return T1, T2


def triangle_oriented_indices(sector: int, triangle: str) -> tuple[int, int, int]:
    triangle = str(triangle).upper()
    if triangle not in _TRIANGLE_LOCAL_ORIENTED:
        raise ValueError("triangle must be 'A' or 'B'")
    base = NSUB * int(sector)
    if not 0 <= int(sector) < NSECTOR:
        raise ValueError(f"sector must be in [0,{NSECTOR - 1}]")
    return tuple(base + a for a in _TRIANGLE_LOCAL_ORIENTED[triangle])


def signed_polygon_area(points: np.ndarray) -> float:
    """Signed 2D polygon area for vertices ordered around the loop."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("points must have shape (N>=3, 2)")
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def triangle_signed_area(
    sector: int,
    triangle: str,
    positions: np.ndarray | None = None,
) -> float:
    """Signed area following the project's eta orientation."""
    if positions is None:
        positions = supercell_site_positions()
    inds = triangle_oriented_indices(sector, triangle)
    return signed_polygon_area(np.asarray(positions, dtype=float)[list(inds)])


def grid_and_params_from_checkpoint_metadata(
    meta: dict,
) -> tuple[MatsubaraGrid, RubyParameters]:
    grid = MatsubaraGrid(
        nk1=int(meta["nk1"]),
        nk2=int(meta["nk2"]),
        nw=int(meta["nw"]),
        nOmega=int(meta["nOmega"]),
        T=float(meta["T"]),
    )
    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=float(meta["V"]),
    )
    return grid, params


def reconstruct_checkpoint_G(
    path: str | Path,
    *,
    require_converged: bool = True,
) -> tuple[dict, np.ndarray, MatsubaraGrid, RubyParameters, np.ndarray]:
    """Reconstruct the zero-source Matsubara G stored implicitly in a checkpoint.

    Returns
    -------
    metadata, G, grid, params, stored_density

    Notes
    -----
    A nonzero-source checkpoint is rejected because the current checkpoint
    metadata does not store the source operator/channel needed to rebuild H0.
    """
    path = Path(path)
    meta = read_checkpoint_metadata(path)

    if require_converged and not bool(meta.get("converged", False)):
        raise ValueError(
            f"Checkpoint {path} is marked nonconverged "
            f"(final_error={meta.get('final_error')!r}). "
            "Pass require_converged=False only for diagnostics."
        )

    source = float(meta.get("source", 0.0))
    if abs(source) > 1e-14:
        raise ValueError(
            "Orbital-moment reconstruction currently requires a zero-source "
            "checkpoint because the checkpoint does not store the source channel."
        )

    grid, params = grid_and_params_from_checkpoint_metadata(meta)
    seed, checked_meta, density = load_supercell_checkpoint(
        path,
        params,
        grid,
        float(meta["primitive_filling"]),
    )
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    G = dyson_from_sigma_matrix(
        h0,
        grid,
        float(seed.mu),
        np.asarray(seed.Sigma_H, dtype=complex),
        np.asarray(seed.Sigma_GW, dtype=complex),
    )
    return checked_meta, G, grid, params, np.asarray(density, dtype=float)


def equal_time_density_matrix(
    G: np.ndarray,
    grid: MatsubaraGrid,
    *,
    diagonal_density: np.ndarray | None = None,
) -> np.ndarray:
    """Return rho_ab=<c_b^dagger c_a> averaged over momentum.

    The off-diagonal elements relevant for bond currents are absolutely
    convergent because their high-frequency expansion starts at 1/omega^2.
    The diagonal 1/2 Matsubara term is included.  If a tail-corrected density
    from the GW checkpoint is supplied, it replaces the diagonal entries.
    """
    G = np.asarray(G, dtype=complex)
    if G.shape[:3] != (grid.nf, grid.nk1, grid.nk2):
        raise ValueError(
            "G has incompatible frequency/momentum shape: "
            f"{G.shape[:3]} != {(grid.nf, grid.nk1, grid.nk2)}"
        )
    norb = int(G.shape[-1])
    if G.shape[-2] != norb:
        raise ValueError("G must be square in its last two axes")

    rho = (grid.T / grid.nk) * np.sum(G, axis=(0, 1, 2))
    rho = np.asarray(rho, dtype=complex)
    rho[np.diag_indices(norb)] += 0.5

    # Enforce the exact Hermitian observable structure after the finite
    # Matsubara sum.  This only removes roundoff/finite-box anti-Hermitian noise.
    rho = 0.5 * (rho + rho.conj().T)

    if diagonal_density is not None:
        diagonal_density = np.asarray(diagonal_density, dtype=float)
        if diagonal_density.shape != (norb,):
            raise ValueError(
                f"diagonal_density shape {diagonal_density.shape} != {(norb,)}"
            )
        rho[np.diag_indices(norb)] = diagonal_density
    return rho


def _zero_shift_hopping_amplitude(
    params: RubyParameters,
    I: int,
    J: int,
) -> complex:
    amp = 0.0j
    for a, b, S, value in supercell_hoppings(params):
        if int(a) == int(I) and int(b) == int(J) and np.all(np.asarray(S) == 0):
            amp += complex(value)
    if abs(amp) < 1e-15:
        raise ValueError(f"no zero-shift hopping found for directed bond {I}->{J}")
    return complex(amp)


def phase_bond_current(
    rho: np.ndarray,
    hopping: complex,
    I: int,
    J: int,
) -> float:
    """Return <dH/dphi_IJ> for a directed Peierls phase on I -> J.

    ``rho_ab=<c_b^dagger c_a>``, hence ``rho[J,I]=<c_I^dagger c_J>``.
    """
    rho = np.asarray(rho, dtype=complex)
    z = complex(hopping) * complex(rho[int(J), int(I)])
    return float(-2.0 * np.imag(z))


def triangle_phase_bond_currents(
    rho: np.ndarray,
    params: RubyParameters,
    sector: int,
    triangle: str,
) -> np.ndarray:
    """Three phase-conjugate currents following the eta loop orientation."""
    inds = triangle_oriented_indices(sector, triangle)
    edges = ((inds[0], inds[1]), (inds[1], inds[2]), (inds[2], inds[0]))
    out = np.zeros(3, dtype=float)
    for ie, (I, J) in enumerate(edges):
        hopping = _zero_shift_hopping_amplitude(params, I, J)
        out[ie] = phase_bond_current(rho, hopping, I, J)
    return out


def loop_current_and_spread(oriented_bond_currents: np.ndarray) -> tuple[float, float]:
    """Project three oriented bond currents onto the conserved loop component."""
    vals = np.asarray(oriented_bond_currents, dtype=float).reshape(3)
    loop = float(np.mean(vals))
    spread = float(np.max(np.abs(vals - loop)))
    return loop, spread


def plaquette_moment_from_loop_current(loop_current: float, signed_area: float) -> float:
    """Local plaquette moment in model-energy times lattice-length squared."""
    return float(loop_current) * float(signed_area)


def moment_code_to_muB(
    moment_code: np.ndarray | float,
    *,
    energy_unit_ev: float,
    lattice_constant_angstrom: float,
    spin_degeneracy: float = 1.0,
) -> np.ndarray:
    """Convert code-unit moment E0*a^2 to Bohr magnetons."""
    energy_unit_ev = float(energy_unit_ev)
    lattice_constant_angstrom = float(lattice_constant_angstrom)
    spin_degeneracy = float(spin_degeneracy)
    if energy_unit_ev <= 0.0:
        raise ValueError("energy_unit_ev must be positive")
    if lattice_constant_angstrom <= 0.0:
        raise ValueError("lattice_constant_angstrom must be positive")
    if spin_degeneracy <= 0.0:
        raise ValueError("spin_degeneracy must be positive")

    prefactor = (
        spin_degeneracy
        * (ELEMENTARY_CHARGE_C / HBAR_J_S)
        * (energy_unit_ev * EV_J)
        * (lattice_constant_angstrom * ANGSTROM_M) ** 2
        / BOHR_MAGNETON_J_T
    )
    return prefactor * np.asarray(moment_code, dtype=float)


def phase_current_code_to_ampere(
    phase_current_code: np.ndarray | float,
    *,
    energy_unit_ev: float,
    spin_degeneracy: float = 1.0,
) -> np.ndarray:
    """Convert the phase-conjugate current scale to amperes.

    The sign follows the Peierls convention documented in the module docstring.
    """
    energy_unit_ev = float(energy_unit_ev)
    spin_degeneracy = float(spin_degeneracy)
    if energy_unit_ev <= 0.0:
        raise ValueError("energy_unit_ev must be positive")
    if spin_degeneracy <= 0.0:
        raise ValueError("spin_degeneracy must be positive")
    prefactor = (
        spin_degeneracy
        * (ELEMENTARY_CHARGE_C / HBAR_J_S)
        * (energy_unit_ev * EV_J)
    )
    return prefactor * np.asarray(phase_current_code, dtype=float)


def analyze_checkpoint_orbital_moments(
    path: str | Path,
    *,
    require_converged: bool = True,
    energy_unit_ev: float | None = None,
    lattice_constant_angstrom: float | None = None,
    spin_degeneracy: float = 1.0,
) -> OrbitalMomentResult:
    """Analyze triangle-resolved orbital moments from one GW checkpoint."""
    meta, G, grid, params, stored_density = reconstruct_checkpoint_G(
        path, require_converged=require_converged
    )
    rho = equal_time_density_matrix(
        G,
        grid,
        diagonal_density=stored_density,
    )
    positions = supercell_site_positions()

    phase_bond = np.zeros((NSECTOR, 2, 3), dtype=float)
    loop = np.zeros((NSECTOR, 2), dtype=float)
    spread = np.zeros((NSECTOR, 2), dtype=float)
    areas = np.zeros((NSECTOR, 2), dtype=float)
    moments = np.zeros((NSECTOR, 2), dtype=float)

    for s in range(NSECTOR):
        for itri, tri in enumerate(TRIANGLE_LABELS):
            vals = triangle_phase_bond_currents(rho, params, s, tri)
            jloop, jspread = loop_current_and_spread(vals)
            area = triangle_signed_area(s, tri, positions=positions)
            phase_bond[s, itri] = vals
            loop[s, itri] = jloop
            spread[s, itri] = jspread
            areas[s, itri] = area
            moments[s, itri] = plaquette_moment_from_loop_current(jloop, area)

    cell_net = moments[:, 0] + moments[:, 1]
    cell_staggered = moments[:, 0] - moments[:, 1]
    supercell_net = float(np.sum(cell_net))

    moments_muB = None
    cell_net_muB = None
    cell_stag_muB = None
    supercell_net_muB = None
    loop_A = None

    if (energy_unit_ev is None) != (lattice_constant_angstrom is None):
        raise ValueError(
            "physical moment conversion requires both energy_unit_ev and "
            "lattice_constant_angstrom"
        )

    if energy_unit_ev is not None:
        moments_muB = moment_code_to_muB(
            moments,
            energy_unit_ev=float(energy_unit_ev),
            lattice_constant_angstrom=float(lattice_constant_angstrom),
            spin_degeneracy=spin_degeneracy,
        )
        cell_net_muB = moments_muB[:, 0] + moments_muB[:, 1]
        cell_stag_muB = moments_muB[:, 0] - moments_muB[:, 1]
        supercell_net_muB = float(np.sum(cell_net_muB))
        loop_A = phase_current_code_to_ampere(
            loop,
            energy_unit_ev=float(energy_unit_ev),
            spin_degeneracy=spin_degeneracy,
        )

    return OrbitalMomentResult(
        metadata=dict(meta),
        density_matrix=rho,
        phase_bond_currents=phase_bond,
        loop_phase_currents=loop,
        loop_current_spread=spread,
        signed_triangle_areas=areas,
        plaquette_moments_code=moments,
        cell_net_moments_code=cell_net,
        cell_staggered_moments_code=cell_staggered,
        supercell_net_moment_code=supercell_net,
        plaquette_moments_muB=moments_muB,
        cell_net_moments_muB=cell_net_muB,
        cell_staggered_moments_muB=cell_stag_muB,
        supercell_net_moment_muB=supercell_net_muB,
        loop_charge_currents_A=loop_A,
    )
