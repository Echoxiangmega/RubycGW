"""Bulk orbital magnetization from an 18-site Ruby GW checkpoint.

This module implements the first Green-function/current-vertex term of
Nourafkan, Kotliar, and Tremblay, Phys. Rev. B 90, 125132 (2014), Eq. (2),

    M_z^(1) = (i e / 2 hbar) (T/N_k) sum_{k,n}
              Tr{ [H0-mu+Sigma/2] G J_x G J_y G - (x <-> y) },

with

    J_alpha = - d G^{-1}/d k_alpha
            = D_alpha H0 + D_alpha Sigma.

The derivative is a physical Cartesian derivative. The repository stores Bloch
matrices in a cell gauge where only the supercell translation S appears in
exp(2*pi*i*k.S). Therefore D_alpha must also include the intra-cell orbital
embedding. For any matrix field X,

    [D_alpha X]_{IJ}(k)
      = i sum_S [R_S + r_J-r_I]_alpha X_{IJ}(S) exp(2*pi*i*k.S).

For H0 the real-space coefficients are known exactly from the hopping list. For
the checkpoint self-energy they are obtained by a spectral inverse Fourier
transform of the sampled k mesh.

Important completeness note
---------------------------
The full interacting Eq. (2) also contains

    M_z^(2) = (1/2N beta) sum Tr{
                  [H0 + (i omega_n-mu) I] G
                  (d Sigma_tilde^(B) / d B_z)|_{B=0} G }.

This term vanishes in the noninteracting limit and in the local-DMFT example
used in the Nourafkan paper, but it is not guaranteed to vanish for nonlocal
GW. The checkpoint analyzer therefore never silently sets it to zero. It
reports M^(1) immediately and marks the total as incomplete unless a magnetic
self-energy derivative is supplied (or V=0 exactly).

The optional ``sigma_b`` input uses the dimensionless magnetic variable

    b = (e a^2 / hbar) B_z,

where ``a`` is the length unit used by :func:`primitive_geometry`. With this
choice both M^(1) and M^(2) share the physical conversion

    m = (e/hbar) E0 a^2 M_code.

The code uses the positive elementary-charge magnitude in that conversion,
matching :mod:`rubycgw.orbital_moment`. Overall electron-charge/sign conventions
should be checked against the chosen Peierls convention when comparing to an
external convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint import load_supercell_checkpoint, read_checkpoint_metadata
from .grids import MatsubaraGrid
from .model import RubyParameters
from .orbital_moment import (
    ANGSTROM_M,
    BOHR_MAGNETON_J_T,
    ELEMENTARY_CHARGE_C,
    EV_J,
    HBAR_J_S,
    grid_and_params_from_checkpoint_metadata,
    supercell_site_positions,
    supercell_vectors_cart,
)
from .supercell import NSECTOR, NSUP, build_supercell_h0, supercell_hoppings
from .supercell_gw import dyson_from_sigma_matrix


@dataclass
class BulkOrbitalMagnetizationResult:
    """Bulk orbital-magnetization analysis for one checkpoint."""

    metadata: dict
    main_term_code: float
    main_term_imag_residual: float
    main_term_per_primitive_code: float
    main_term_2d_density_code: float
    field_self_energy_term_code: float | None
    field_self_energy_term_imag_residual: float | None
    total_code: float | None
    total_per_primitive_code: float | None
    total_2d_density_code: float | None
    field_self_energy_status: str
    complete: bool
    supercell_area_code: float
    primitive_cell_area_code: float
    max_abs_Dx_H0: float
    max_abs_Dy_H0: float
    max_abs_Dx_Sigma: float
    max_abs_Dy_Sigma: float
    k_resolved_main_code: np.ndarray
    k_resolved_field_code: np.ndarray | None
    main_term_muB: float | None = None
    main_term_per_primitive_muB: float | None = None
    total_muB: float | None = None
    total_per_primitive_muB: float | None = None


def _validate_kmatrix(field: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Validate ``(...,nk1,nk2,norb,norb)`` and return axis metadata."""
    arr = np.asarray(field, dtype=complex)
    if arr.ndim < 4:
        raise ValueError("field must have shape (...,nk1,nk2,norb,norb)")
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError("field must be square in its last two axes")
    ax1 = arr.ndim - 4
    ax2 = arr.ndim - 3
    nk1 = int(arr.shape[ax1])
    nk2 = int(arr.shape[ax2])
    return arr, ax1, ax2, nk1, nk2


def _spectral_lattice_modes(n: int) -> np.ndarray:
    """Signed real-space harmonics represented by an N-point periodic mesh."""
    return np.fft.fftfreq(int(n)) * float(n)


def spectral_cartesian_covariant_derivatives(
    field: np.ndarray,
    positions: np.ndarray | None = None,
    supercell_vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical Cartesian ``D_x field`` and ``D_y field``.

    The sampled field follows the repository cell-gauge Fourier convention

        X(k) = sum_S X(S) exp(2*pi*i*k.S).

    A spectral inverse transform gives X(S) on the real-space harmonics
    represented by the finite k mesh. The physical derivative multiplies each
    matrix element by the complete displacement ``R_S + r_J-r_I``.
    """
    arr, ax1, ax2, nk1, nk2 = _validate_kmatrix(field)
    norb = int(arr.shape[-1])

    if positions is None:
        positions = supercell_site_positions()
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (norb, 2):
        raise ValueError(f"positions shape {positions.shape} != {(norb, 2)}")

    if supercell_vectors is None:
        T1, T2 = supercell_vectors_cart()
    else:
        T1, T2 = supercell_vectors
    T1 = np.asarray(T1, dtype=float).reshape(2)
    T2 = np.asarray(T2, dtype=float).reshape(2)

    nk = nk1 * nk2
    coeff = np.fft.fftn(arr, axes=(ax1, ax2)) / float(nk)

    s1 = _spectral_lattice_modes(nk1)
    s2 = _spectral_lattice_modes(nk2)
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    Rcart = S1[..., None] * T1[None, None, :] + S2[..., None] * T2[None, None, :]

    # Matrix element I,J carries the embedding displacement r_J-r_I.
    delta = positions[None, :, :] - positions[:, None, :]

    shape_r = [1] * arr.ndim
    shape_r[ax1] = nk1
    shape_r[ax2] = nk2
    shape_r[-2] = 1
    shape_r[-1] = 1

    shape_d = [1] * arr.ndim
    shape_d[-2] = norb
    shape_d[-1] = norb

    dx = Rcart[..., 0].reshape(shape_r) + delta[..., 0].reshape(shape_d)
    dy = Rcart[..., 1].reshape(shape_r) + delta[..., 1].reshape(shape_d)

    dcoeff_x = 1j * dx * coeff
    dcoeff_y = 1j * dy * coeff
    Dx = float(nk) * np.fft.ifftn(dcoeff_x, axes=(ax1, ax2))
    Dy = float(nk) * np.fft.ifftn(dcoeff_y, axes=(ax1, ax2))
    return np.asarray(Dx, dtype=complex), np.asarray(Dy, dtype=complex)


def supercell_h0_cartesian_derivatives(
    kpts: np.ndarray,
    params: RubyParameters,
    *,
    positions: np.ndarray | None = None,
    supercell_vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic physical derivatives of the 18-site Bloch Hamiltonian."""
    kpts = np.asarray(kpts, dtype=float)
    flat = kpts.reshape(-1, 2)

    if positions is None:
        positions = supercell_site_positions()
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (NSUP, 2):
        raise ValueError(f"positions shape {positions.shape} != {(NSUP, 2)}")

    if supercell_vectors is None:
        T1, T2 = supercell_vectors_cart()
    else:
        T1, T2 = supercell_vectors
    T1 = np.asarray(T1, dtype=float).reshape(2)
    T2 = np.asarray(T2, dtype=float).reshape(2)

    Dx = np.zeros((flat.shape[0], NSUP, NSUP), dtype=complex)
    Dy = np.zeros_like(Dx)
    hops = supercell_hoppings(params)

    for ik, k in enumerate(flat):
        for I, J, S, amp in hops:
            S = np.asarray(S, dtype=float)
            R = S[0] * T1 + S[1] * T2
            d = R + positions[int(J)] - positions[int(I)]
            phase = np.exp(2j * np.pi * np.dot(k, S))
            value = complex(amp) * phase
            Dx[ik, int(I), int(J)] += 1j * d[0] * value
            Dy[ik, int(I), int(J)] += 1j * d[1] * value

    Dx = 0.5 * (Dx + np.swapaxes(Dx.conj(), -1, -2))
    Dy = 0.5 * (Dy + np.swapaxes(Dy.conj(), -1, -2))
    shape = kpts.shape[:-1] + (NSUP, NSUP)
    return Dx.reshape(shape), Dy.reshape(shape)


def _matrix_chain_trace(E: np.ndarray, G: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Trace ``E G A G B G`` over the final two matrix axes."""
    chain = np.matmul(E, G)
    chain = np.matmul(chain, A)
    chain = np.matmul(chain, G)
    chain = np.matmul(chain, B)
    chain = np.matmul(chain, G)
    return np.trace(chain, axis1=-2, axis2=-1)


def _nourafkan_main_term(
    h0: np.ndarray,
    G: np.ndarray,
    sigma_total: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    Dx_H0: np.ndarray,
    Dy_H0: np.ndarray,
    Dx_Sigma: np.ndarray,
    Dy_Sigma: np.ndarray,
) -> tuple[complex, np.ndarray]:
    """Return the first term of Nourafkan Eq. (2) in code units."""
    eye = np.eye(int(G.shape[-1]), dtype=complex)
    E = h0[None, :, :, :, :] - float(mu) * eye[None, None, None, :, :] + 0.5 * sigma_total
    Jx = Dx_H0[None, :, :, :, :] + Dx_Sigma
    Jy = Dy_H0[None, :, :, :, :] + Dy_Sigma

    txy = _matrix_chain_trace(E, G, Jx, Jy)
    tyx = _matrix_chain_trace(E, G, Jy, Jx)
    k_resolved = 0.5j * float(grid.T) * np.sum(txy - tyx, axis=0)
    total = np.sum(k_resolved) / float(grid.nk)
    return complex(total), np.asarray(k_resolved, dtype=complex)


def _nourafkan_field_self_energy_term(
    h0: np.ndarray,
    G: np.ndarray,
    sigma_b: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
) -> tuple[complex, np.ndarray]:
    """Return Eq. (2)'s ``d Sigma_tilde^(B)/dB`` term in code units.

    ``sigma_b`` means ``d Sigma_tilde / db`` with
    ``b=(e a^2/hbar) B_z``. This normalization gives the same final physical
    conversion as the first term.
    """
    sigma_b = np.asarray(sigma_b, dtype=complex)
    if sigma_b.shape != G.shape:
        raise ValueError(f"sigma_b shape {sigma_b.shape} != G shape {G.shape}")

    norb = int(G.shape[-1])
    eye = np.eye(norb, dtype=complex)
    energy_vertex = h0[None, :, :, :, :] + (
        1j * grid.omega[:, None, None, None, None] - float(mu)
    ) * eye[None, None, None, :, :]
    chain = np.matmul(energy_vertex, G)
    chain = np.matmul(chain, sigma_b)
    chain = np.matmul(chain, G)
    tr = np.trace(chain, axis1=-2, axis2=-1)
    k_resolved = 0.5 * float(grid.T) * np.sum(tr, axis=0)
    total = np.sum(k_resolved) / float(grid.nk)
    return complex(total), np.asarray(k_resolved, dtype=complex)


def supercell_area_code() -> float:
    """Area of the 18-site supercell in squared lattice-length units."""
    T1, T2 = supercell_vectors_cart()
    return float(abs(T1[0] * T2[1] - T1[1] * T2[0]))


def moment_code_to_muB(
    value: float | np.ndarray,
    *,
    energy_unit_ev: float,
    lattice_constant_angstrom: float,
    spin_degeneracy: float = 1.0,
) -> np.ndarray:
    """Convert code moment to Bohr magnetons."""
    E0_J = float(energy_unit_ev) * EV_J
    a_m = float(lattice_constant_angstrom) * ANGSTROM_M
    pref = ELEMENTARY_CHARGE_C * E0_J * a_m**2 / HBAR_J_S / BOHR_MAGNETON_J_T
    return np.asarray(value, dtype=float) * pref * float(spin_degeneracy)


def bulk_orbital_magnetization_from_arrays(
    h0: np.ndarray,
    G: np.ndarray,
    sigma_total: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    params: RubyParameters,
    *,
    metadata: dict | None = None,
    sigma_b: np.ndarray | None = None,
    energy_unit_ev: float | None = None,
    lattice_constant_angstrom: float | None = None,
    spin_degeneracy: float = 1.0,
) -> BulkOrbitalMagnetizationResult:
    """Evaluate the checkpoint-compatible Nourafkan bulk-M expression."""
    h0 = np.asarray(h0, dtype=complex)
    G = np.asarray(G, dtype=complex)
    sigma_total = np.asarray(sigma_total, dtype=complex)

    expected_h0 = (grid.nk1, grid.nk2, NSUP, NSUP)
    expected_G = (grid.nf, grid.nk1, grid.nk2, NSUP, NSUP)
    if h0.shape != expected_h0:
        raise ValueError(f"h0 shape {h0.shape} != {expected_h0}")
    if G.shape != expected_G:
        raise ValueError(f"G shape {G.shape} != {expected_G}")
    if sigma_total.shape != expected_G:
        raise ValueError(f"sigma_total shape {sigma_total.shape} != {expected_G}")

    positions = supercell_site_positions()
    vectors = supercell_vectors_cart()
    Dx_H0, Dy_H0 = supercell_h0_cartesian_derivatives(
        grid.kmesh(), params, positions=positions, supercell_vectors=vectors
    )
    Dx_Sigma, Dy_Sigma = spectral_cartesian_covariant_derivatives(
        sigma_total, positions=positions, supercell_vectors=vectors
    )

    main_complex, k_main_complex = _nourafkan_main_term(
        h0, G, sigma_total, mu, grid, Dx_H0, Dy_H0, Dx_Sigma, Dy_Sigma
    )
    main = float(main_complex.real)

    field_complex = None
    k_field_complex = None
    if sigma_b is not None:
        field_complex, k_field_complex = _nourafkan_field_self_energy_term(
            h0, G, sigma_b, mu, grid
        )
        field = float(field_complex.real)
        status = "provided"
        complete = True
        total = main + field
    elif abs(float(params.V)) < 1e-15:
        field = 0.0
        field_complex = 0.0j
        k_field_complex = np.zeros((grid.nk1, grid.nk2), dtype=complex)
        status = "exact_zero_noninteracting"
        complete = True
        total = main
    else:
        field = None
        status = "missing_for_nonlocal_interacting_self_energy"
        complete = False
        total = None

    area_sc = supercell_area_code()
    area_primitive = area_sc / float(NSECTOR)
    main_per_primitive = main / float(NSECTOR)
    main_density = main / area_sc

    if total is None:
        total_per_primitive = None
        total_density = None
    else:
        total_per_primitive = total / float(NSECTOR)
        total_density = total / area_sc

    main_muB = None
    main_primitive_muB = None
    total_muB = None
    total_primitive_muB = None
    have_energy = energy_unit_ev is not None
    have_length = lattice_constant_angstrom is not None
    if have_energy != have_length:
        raise ValueError("energy_unit_ev and lattice_constant_angstrom must be supplied together")
    if have_energy:
        main_muB = float(moment_code_to_muB(
            main,
            energy_unit_ev=float(energy_unit_ev),
            lattice_constant_angstrom=float(lattice_constant_angstrom),
            spin_degeneracy=spin_degeneracy,
        ))
        main_primitive_muB = main_muB / float(NSECTOR)
        if total is not None:
            total_muB = float(moment_code_to_muB(
                total,
                energy_unit_ev=float(energy_unit_ev),
                lattice_constant_angstrom=float(lattice_constant_angstrom),
                spin_degeneracy=spin_degeneracy,
            ))
            total_primitive_muB = total_muB / float(NSECTOR)

    return BulkOrbitalMagnetizationResult(
        metadata={} if metadata is None else dict(metadata),
        main_term_code=main,
        main_term_imag_residual=float(main_complex.imag),
        main_term_per_primitive_code=main_per_primitive,
        main_term_2d_density_code=main_density,
        field_self_energy_term_code=field,
        field_self_energy_term_imag_residual=(
            None if field_complex is None else float(complex(field_complex).imag)
        ),
        total_code=total,
        total_per_primitive_code=total_per_primitive,
        total_2d_density_code=total_density,
        field_self_energy_status=status,
        complete=bool(complete),
        supercell_area_code=area_sc,
        primitive_cell_area_code=area_primitive,
        max_abs_Dx_H0=float(np.max(np.abs(Dx_H0))),
        max_abs_Dy_H0=float(np.max(np.abs(Dy_H0))),
        max_abs_Dx_Sigma=float(np.max(np.abs(Dx_Sigma))),
        max_abs_Dy_Sigma=float(np.max(np.abs(Dy_Sigma))),
        k_resolved_main_code=np.asarray(k_main_complex.real, dtype=float),
        k_resolved_field_code=(
            None if k_field_complex is None else np.asarray(k_field_complex.real, dtype=float)
        ),
        main_term_muB=main_muB,
        main_term_per_primitive_muB=main_primitive_muB,
        total_muB=total_muB,
        total_per_primitive_muB=total_primitive_muB,
    )


def analyze_checkpoint_bulk_orbital_magnetization(
    path: str | Path,
    *,
    sigma_b: np.ndarray | None = None,
    require_converged: bool = True,
    energy_unit_ev: float | None = None,
    lattice_constant_angstrom: float | None = None,
    spin_degeneracy: float = 1.0,
) -> BulkOrbitalMagnetizationResult:
    """Load an 18-site zero-source checkpoint and evaluate bulk orbital M."""
    path = Path(path)
    meta = read_checkpoint_metadata(path)
    if require_converged and not bool(meta.get("converged", False)):
        raise ValueError(
            f"Checkpoint {path} is marked nonconverged "
            f"(final_error={meta.get('final_error')!r})."
        )
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError(
            "Bulk-M reconstruction currently requires a zero-source checkpoint "
            "because the checkpoint metadata does not store the source operator."
        )

    grid, params = grid_and_params_from_checkpoint_metadata(meta)
    seed, checked_meta, _ = load_supercell_checkpoint(
        path, params, grid, float(meta["primitive_filling"])
    )
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    sigma_total = (
        np.asarray(seed.Sigma_GW, dtype=complex)
        + np.asarray(seed.Sigma_H, dtype=complex)[None, None, None, :, :]
    )
    G = dyson_from_sigma_matrix(
        h0,
        grid,
        float(seed.mu),
        np.asarray(seed.Sigma_H, dtype=complex),
        np.asarray(seed.Sigma_GW, dtype=complex),
    )
    return bulk_orbital_magnetization_from_arrays(
        h0,
        G,
        sigma_total,
        float(seed.mu),
        grid,
        params,
        metadata=checked_meta,
        sigma_b=sigma_b,
        energy_unit_ev=energy_unit_ev,
        lattice_constant_angstrom=lattice_constant_angstrom,
        spin_degeneracy=spin_degeneracy,
    )
