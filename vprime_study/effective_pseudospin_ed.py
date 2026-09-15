"""Finite-size quantum ED for the Ruby strong-coupling pseudospin model.

The model uses Pauli pseudospins (eigenvalues +/-1) on the triangle-center
honeycomb-like lattice,

    H = sum_<ij>_gamma [
        Jn tau_i^{n_gamma} tau_j^{n_gamma}
      + Jm tau_i^{m_gamma} tau_j^{m_gamma}
      + Jz tau_i^z tau_j^z
    ],

with theta_gamma=2*pi*gamma/3, n=(cos theta,sin theta), and
m=(-sin theta,cos theta). There are two triangle pseudospins (A/B) per
primitive cell and three A->B bond orientations with cell offsets
(0,0), (0,1), (-1,0).

The Hilbert space is split by the exact unitary parity prod_i tau_i^z, since all
x/y terms flip two pseudospins. This reduces a 3x3 torus (18 pseudospins) to
131072 states per parity sector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .cross_model import reference_pair_effective_couplings

CHANNELS = np.asarray(["Ax", "Ay", "Az", "Bx", "By", "Bz"])
BOND_OFFSETS = ((0, 0), (0, 1), (-1, 0))


@dataclass
class SectorHamiltonian:
    parity: int
    states: np.ndarray
    lookup: np.ndarray
    matrix: sp.csr_matrix


@dataclass
class SectorSpectrum:
    parity: int
    states: np.ndarray
    lookup: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray


@dataclass
class EffectiveEDResult:
    Lx: int
    Ly: int
    nspin: int
    Jn: float
    Jm: float
    Jz: float
    ground_energy: float
    gap: float
    ground_degeneracy: int
    ground_parities: np.ndarray
    energies_even: np.ndarray
    energies_odd: np.ndarray
    q_raw: np.ndarray
    q_centered: np.ndarray
    structure_factor: np.ndarray
    sf_eigenvalues: np.ndarray
    sf_eigenvectors: np.ndarray
    lambda_max: np.ndarray
    z_same: np.ndarray
    z_opposite: np.ndarray
    xy_max: np.ndarray


def effective_couplings(*, t: float, V: float, Vprime: float, Vcross: float):
    """Return the leading-order (Jn,Jm,Jz) used by the ED model."""
    return reference_pair_effective_couplings(
        t=float(t), V=float(V), Vprime=float(Vprime), Vcross=float(Vcross)
    )


def triangle_center_bonds(Lx: int, Ly: int):
    """Return (i,j,gamma) bonds for an Lx x Ly periodic triangle-center torus."""
    Lx, Ly = int(Lx), int(Ly)
    if Lx < 1 or Ly < 1:
        raise ValueError("Lx and Ly must be positive")
    out: list[tuple[int, int, int]] = []
    for y in range(Ly):
        for x in range(Lx):
            ia = 2 * (x + Lx * y)
            for gamma, (dx, dy) in enumerate(BOND_OFFSETS):
                xb = (x + dx) % Lx
                yb = (y + dy) % Ly
                ib = 2 * (xb + Lx * yb) + 1
                out.append((ia, ib, gamma))
    return tuple(out)


def _parity_bits(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states, dtype=np.uint64).copy()
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    return (x & np.uint64(1)).astype(np.uint8)


def parity_basis(nspin: int, parity: int) -> np.ndarray:
    nspin, parity = int(nspin), int(parity)
    if nspin < 1 or nspin > 30:
        raise ValueError("this ED implementation supports 1 <= nspin <= 30")
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    states = np.arange(1 << nspin, dtype=np.uint64)
    return states[_parity_bits(states) == parity]


def _lookup_for_basis(states: np.ndarray, nspin: int) -> np.ndarray:
    lookup = np.full(1 << int(nspin), -1, dtype=np.int32)
    lookup[np.asarray(states, dtype=np.int64)] = np.arange(states.size, dtype=np.int32)
    return lookup


def build_sector_hamiltonian(
    Lx: int,
    Ly: int,
    Jn: float,
    Jm: float,
    Jz: float,
    parity: int,
) -> SectorHamiltonian:
    """Build one prod(tau_z) parity block as a complex CSR matrix."""
    Lx, Ly = int(Lx), int(Ly)
    nspin = 2 * Lx * Ly
    states = parity_basis(nspin, parity)
    dim = int(states.size)
    lookup = _lookup_for_basis(states, nspin)
    src = np.arange(dim, dtype=np.int32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    diagonal = np.zeros(dim, dtype=float)

    for i, j, gamma in triangle_center_bonds(Lx, Ly):
        theta = 2.0 * np.pi * gamma / 3.0
        nx, ny = float(np.cos(theta)), float(np.sin(theta))
        mx, my = -ny, nx
        Jxx = float(Jn) * nx * nx + float(Jm) * mx * mx
        Jyy = float(Jn) * ny * ny + float(Jm) * my * my
        Jxy = float(Jn) * nx * ny + float(Jm) * mx * my

        bi = ((states >> np.uint64(i)) & np.uint64(1)).astype(np.int8)
        bj = ((states >> np.uint64(j)) & np.uint64(1)).astype(np.int8)
        zi = 1 - 2 * bi
        zj = 1 - 2 * bj
        diagonal += float(Jz) * zi * zj

        mask = (np.uint64(1) << np.uint64(i)) ^ (np.uint64(1) << np.uint64(j))
        target = lookup[states ^ mask]
        # sigma_x sigma_x -> 1
        # sigma_y sigma_y -> -zi*zj
        # sigma_x sigma_y + sigma_y sigma_x -> i*(zi+zj)
        amplitude = Jxx - Jyy * (zi * zj) + 1j * Jxy * (zi + zj)
        rows.append(target)
        cols.append(src)
        data.append(np.asarray(amplitude, dtype=complex))

    rows.append(src)
    cols.append(src)
    data.append(np.asarray(diagonal, dtype=complex))
    H = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
        dtype=complex,
    ).tocsr()
    H.sum_duplicates()
    return SectorHamiltonian(int(parity), states, lookup, H)


def solve_sector(
    block: SectorHamiltonian,
    *,
    nev: int = 4,
    tol: float = 1e-10,
    maxiter: int | None = None,
    v0: np.ndarray | None = None,
) -> SectorSpectrum:
    """Return the lowest states in one parity block."""
    H = block.matrix
    dim = H.shape[0]
    nev = max(1, int(nev))
    if dim <= max(32, nev + 1):
        values, vectors = np.linalg.eigh(H.toarray())
        take = min(nev, dim)
        values, vectors = values[:take], vectors[:, :take]
    else:
        k = min(nev, dim - 1)
        values, vectors = eigsh(
            H, k=k, which="SA", tol=float(tol), maxiter=maxiter, v0=v0
        )
        order = np.argsort(values.real)
        values = np.asarray(values[order].real, dtype=float)
        vectors = np.asarray(vectors[:, order], dtype=complex)
    return SectorSpectrum(
        parity=block.parity,
        states=block.states,
        lookup=block.lookup,
        energies=np.asarray(values, dtype=float),
        vectors=np.asarray(vectors, dtype=complex),
    )


def allowed_qmesh(Lx: int, Ly: int) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    for iy in range(int(Ly)):
        for ix in range(int(Lx)):
            raw.append((ix / float(Lx), iy / float(Ly)))
    qraw = np.asarray(raw, dtype=float)
    qcentered = (qraw + 0.5) % 1.0 - 0.5
    return qraw, qcentered


def _local_sum_vector(
    *,
    Lx: int,
    Ly: int,
    states: np.ndarray,
    psi: np.ndarray,
    q: np.ndarray,
    sublattice: int,
    component: str,
    target_lookup: np.ndarray | None = None,
    target_dim: int | None = None,
) -> np.ndarray:
    """Apply O_{a,mu}(q)=Ncell^-1/2 sum_R exp(-iqR) tau_{R,a}^mu."""
    ncell = int(Lx) * int(Ly)
    if component == "z":
        out = np.zeros(states.size, dtype=complex)
    else:
        if target_lookup is None or target_dim is None:
            raise ValueError("x/y operator needs the opposite-parity lookup")
        out = np.zeros(int(target_dim), dtype=complex)

    norm = np.sqrt(float(ncell))
    for y in range(int(Ly)):
        for x in range(int(Lx)):
            site = 2 * (x + int(Lx) * y) + int(sublattice)
            phase = np.exp(-2j * np.pi * (float(q[0]) * x + float(q[1]) * y)) / norm
            bit = ((states >> np.uint64(site)) & np.uint64(1)).astype(np.int8)
            z = 1 - 2 * bit
            if component == "z":
                out += phase * z * psi
            else:
                target = target_lookup[states ^ (np.uint64(1) << np.uint64(site))]
                amp = np.ones(states.size, dtype=complex)
                if component == "y":
                    amp = 1j * z
                out[target] += phase * amp * psi
    return out


def structure_factor_matrix(
    *,
    Lx: int,
    Ly: int,
    state: SectorSpectrum,
    vector_index: int,
    q: np.ndarray,
    opposite_states: np.ndarray,
    opposite_lookup: np.ndarray,
) -> np.ndarray:
    """Return the 6x6 equal-time structure-factor matrix at one q."""
    psi = np.asarray(state.vectors[:, int(vector_index)], dtype=complex)
    xy_vectors: dict[str, np.ndarray] = {}
    z_vectors: dict[str, np.ndarray] = {}
    for sub, prefix in ((0, "A"), (1, "B")):
        for comp in ("x", "y"):
            xy_vectors[prefix + comp] = _local_sum_vector(
                Lx=Lx, Ly=Ly, states=state.states, psi=psi, q=q,
                sublattice=sub, component=comp,
                target_lookup=opposite_lookup, target_dim=opposite_states.size,
            )
        z_vectors[prefix + "z"] = _local_sum_vector(
            Lx=Lx, Ly=Ly, states=state.states, psi=psi, q=q,
            sublattice=sub, component="z",
        )

    S = np.zeros((6, 6), dtype=complex)
    for ia, a in enumerate(CHANNELS.tolist()):
        va = z_vectors[a] if a.endswith("z") else xy_vectors[a]
        for ib, b in enumerate(CHANNELS.tolist()):
            if a.endswith("z") != b.endswith("z"):
                continue
            vb = z_vectors[b] if b.endswith("z") else xy_vectors[b]
            S[ia, ib] = np.vdot(va, vb)
    return 0.5 * (S + S.conj().T)


def _projection(channels: np.ndarray, comp: str, parity: str) -> np.ndarray:
    names = [str(x) for x in channels.tolist()]
    v = np.zeros(len(names), dtype=complex)
    ia, ib = names.index("A" + comp), names.index("B" + comp)
    v[ia] = 1.0 / np.sqrt(2.0)
    v[ib] = (1.0 if parity == "even" else -1.0) / np.sqrt(2.0)
    return v


def _expect(S: np.ndarray, v: np.ndarray) -> float:
    return float(np.real(np.vdot(v, S @ v)))


def solve_effective_pseudospin_ed(
    *,
    Lx: int,
    Ly: int,
    Jn: float,
    Jm: float,
    Jz: float,
    nev: int = 4,
    tol: float = 1e-10,
    maxiter: int | None = None,
    degeneracy_tol: float = 1e-7,
    v0_even: np.ndarray | None = None,
    v0_odd: np.ndarray | None = None,
) -> EffectiveEDResult:
    """Solve both parity sectors and compute all allowed-q structure factors."""
    Lx, Ly = int(Lx), int(Ly)
    blocks = {
        p: build_sector_hamiltonian(Lx, Ly, Jn, Jm, Jz, p) for p in (0, 1)
    }
    spectra = {
        0: solve_sector(blocks[0], nev=nev, tol=tol, maxiter=maxiter, v0=v0_even),
        1: solve_sector(blocks[1], nev=nev, tol=tol, maxiter=maxiter, v0=v0_odd),
    }

    levels = []
    for p in (0, 1):
        for k, energy in enumerate(spectra[p].energies):
            levels.append((float(energy), p, k))
    levels.sort(key=lambda x: x[0])
    e0 = levels[0][0]
    gs = [x for x in levels if abs(x[0] - e0) <= float(degeneracy_tol)]
    excited = [x[0] for x in levels if x[0] > e0 + float(degeneracy_tol)]
    gap = float(excited[0] - e0) if excited else np.nan

    qraw, qcentered = allowed_qmesh(Lx, Ly)
    nq = qraw.shape[0]
    Savg = np.zeros((nq, 6, 6), dtype=complex)
    for _energy, p, k in gs:
        other = spectra[p ^ 1]
        for iq, q in enumerate(qraw):
            Savg[iq] += structure_factor_matrix(
                Lx=Lx, Ly=Ly, state=spectra[p], vector_index=k, q=q,
                opposite_states=other.states, opposite_lookup=other.lookup,
            )
    Savg /= float(len(gs))

    evals = np.empty((nq, 6), dtype=float)
    evecs = np.empty((nq, 6, 6), dtype=complex)
    for iq, S in enumerate(Savg):
        w, u = np.linalg.eigh(0.5 * (S + S.conj().T))
        order = np.argsort(w.real)[::-1]
        evals[iq] = w[order].real
        evecs[iq] = u[:, order]

    zsame_v = _projection(CHANNELS, "z", "even")
    zopp_v = _projection(CHANNELS, "z", "odd")
    zsame = np.asarray([_expect(S, zsame_v) for S in Savg])
    zopp = np.asarray([_expect(S, zopp_v) for S in Savg])
    xy_idx = [0, 1, 3, 4]
    xymax = np.empty(nq, dtype=float)
    for iq, S in enumerate(Savg):
        block = S[np.ix_(xy_idx, xy_idx)]
        xymax[iq] = float(np.max(np.linalg.eigvalsh(block)).real)

    return EffectiveEDResult(
        Lx=Lx, Ly=Ly, nspin=2 * Lx * Ly,
        Jn=float(Jn), Jm=float(Jm), Jz=float(Jz),
        ground_energy=float(e0), gap=gap, ground_degeneracy=len(gs),
        ground_parities=np.asarray([p for _e, p, _k in gs], dtype=int),
        energies_even=np.asarray(spectra[0].energies),
        energies_odd=np.asarray(spectra[1].energies),
        q_raw=qraw, q_centered=qcentered, structure_factor=Savg,
        sf_eigenvalues=evals, sf_eigenvectors=evecs,
        lambda_max=evals[:, 0].copy(), z_same=zsame, z_opposite=zopp, xy_max=xymax,
    )


def leading_mode_label(vec: np.ndarray) -> tuple[str, str, float]:
    c = {name: complex(vec[i]) for i, name in enumerate(CHANNELS.tolist())}
    weights = {}
    parity_amp = {}
    for comp in "xyz":
        A, B = c["A" + comp], c["B" + comp]
        weights[comp] = abs(A) ** 2 + abs(B) ** 2
        parity_amp[(comp, "even")] = abs((A + B) / np.sqrt(2.0))
        parity_amp[(comp, "odd")] = abs((A - B) / np.sqrt(2.0))
    comp = max(weights, key=weights.get)
    parity = "even" if parity_amp[(comp, "even")] >= parity_amp[(comp, "odd")] else "odd"
    if comp == "z":
        parity = "same" if parity == "even" else "opposite"
    return comp, parity, float(weights[comp])
