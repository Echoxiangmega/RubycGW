"""Pseudo-arclength helpers for same-torus Fierz-channel GW.

The continuation state is restricted to the physical Matsubara symmetry
Sigma(-iw)=Sigma(iw)^dagger and a Hermitian static self-energy.  This halves the
large dynamic state and prevents Newton-Krylov steps from wandering into a
nonphysical complex subspace.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fierz_channel_gw import (
    ChannelGWResult,
    build_channel_definition,
    channel_static_self_energy,
    compute_channel_correlation_self_energy,
    compute_channel_polarization,
    compute_channel_screened_interaction,
)
from .grids import MatsubaraGrid
from .supercell_gw import dyson_from_sigma_matrix
from .supercell_gw_split import one_body_density_matrix_tail


@dataclass
class FierzResidualEvaluation:
    residual: np.ndarray
    G: np.ndarray
    P: np.ndarray
    W: np.ndarray
    sigma_static_out: np.ndarray
    sigma_c_out: np.ndarray
    rho: np.ndarray
    filling_error: float
    physical_residual: float
    smin: float
    soft_m: int
    soft_Omega: float
    soft_mode: np.ndarray


class FierzStateCodec:
    """Real scaled coordinates for (Hermitian Sigma_static, Sigma_c, mu)."""

    def __init__(self, norb: int, grid: MatsubaraGrid, target: float):
        self.norb = int(norb)
        self.grid = grid
        self.target = float(target)
        self.nstat = self.norb * self.norb
        self.ndyn = 2 * int(grid.nw) * self.norb * self.norb
        self.static_scale = np.sqrt(max(self.nstat, 1))
        self.dynamic_scale = np.sqrt(max(self.ndyn, 1))
        self.number_scale = max(abs(self.target), 1.0)
        self.size = self.nstat + self.ndyn + 1

    def _pack_hermitian_raw(self, matrix: np.ndarray) -> np.ndarray:
        A = 0.5 * (np.asarray(matrix, dtype=complex) + np.asarray(matrix, dtype=complex).conj().T)
        out = []
        out.extend(np.real(np.diag(A)).tolist())
        for i in range(self.norb):
            for j in range(i + 1, self.norb):
                out.append(float(A[i, j].real))
                out.append(float(A[i, j].imag))
        return np.asarray(out, dtype=float)

    def _unpack_hermitian_raw(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=float)
        A = np.zeros((self.norb, self.norb), dtype=complex)
        p = 0
        for i in range(self.norb):
            A[i, i] = v[p]
            p += 1
        for i in range(self.norb):
            for j in range(i + 1, self.norb):
                z = complex(v[p], v[p + 1])
                p += 2
                A[i, j] = z
                A[j, i] = z.conjugate()
        if p != self.nstat:
            raise RuntimeError("Hermitian codec size mismatch")
        return A

    def encode(self, sigma_static: np.ndarray, sigma_c: np.ndarray, mu: float) -> np.ndarray:
        stat = self._pack_hermitian_raw(sigma_static) / self.static_scale
        arr = np.asarray(sigma_c, dtype=complex)
        expected = (self.grid.nf, 1, 1, self.norb, self.norb)
        if arr.shape != expected:
            raise ValueError(f"sigma_c shape {arr.shape} != {expected}")
        pos = arr[self.grid.nw:, 0, 0].reshape(-1)
        dyn = np.concatenate([pos.real, pos.imag]) / self.dynamic_scale
        return np.concatenate([stat, dyn, [float(mu)]])

    def decode(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.size:
            raise ValueError(f"state size {x.size} != {self.size}")
        a = self.nstat
        b = a + self.ndyn
        sigma_static = self._unpack_hermitian_raw(x[:a] * self.static_scale)
        dyn = x[a:b] * self.dynamic_scale
        nhalf = self.grid.nw * self.norb * self.norb
        pos = (dyn[:nhalf] + 1j * dyn[nhalf:]).reshape(
            self.grid.nw, self.norb, self.norb
        )
        sigma_c = np.zeros(
            (self.grid.nf, 1, 1, self.norb, self.norb), dtype=complex
        )
        for k in range(self.grid.nw):
            sigma_c[self.grid.nw + k, 0, 0] = pos[k]
            sigma_c[self.grid.nw - 1 - k, 0, 0] = pos[k].conj().T
        return sigma_static, sigma_c, float(x[-1])

    def encode_result(self, result: ChannelGWResult) -> np.ndarray:
        return self.encode(result.Sigma_static, result.Sigma_c, result.mu)

    def encode_residual(
        self,
        r_static: np.ndarray,
        r_dynamic: np.ndarray,
        filling_error: float,
    ) -> np.ndarray:
        stat = self._pack_hermitian_raw(r_static) / self.static_scale
        arr = np.asarray(r_dynamic, dtype=complex)
        pos = arr[self.grid.nw:, 0, 0].reshape(-1)
        dyn = np.concatenate([pos.real, pos.imag]) / self.dynamic_scale
        return np.concatenate([
            stat,
            dyn,
            [float(filling_error) / self.number_scale],
        ])


def screening_soft_mode(P, definition, grid: MatsubaraGrid):
    g = np.asarray(definition.coupling, dtype=complex)
    eye = np.eye(g.shape[0], dtype=complex)
    best = None
    for im, Pm in enumerate(np.asarray(P)):
        M = eye - g @ Pm
        _, s, vh = np.linalg.svd(M, full_matrices=False)
        val = float(s[-1])
        if best is None or val < best[0]:
            mode = vh[-1].conj()
            nrm = float(np.linalg.norm(mode))
            if nrm > 0:
                mode = mode / nrm
            best = (val, im, mode)
    val, im, mode = best
    return val, int(grid.m_values[im]), float(grid.Omega[im]), np.asarray(mode)


def project_operator_to_channel(Kext: np.ndarray, definition) -> np.ndarray:
    A = np.asarray(definition.vertices, dtype=complex).reshape(len(definition.labels), -1)
    rhs = np.asarray(Kext, dtype=complex).reshape(-1)
    coeff, *_ = np.linalg.lstsq(A.T, rhs, rcond=None)
    return np.asarray(coeff, dtype=complex)


def soft_mode_overlap(Kext: np.ndarray, definition, soft_mode: np.ndarray) -> float:
    coeff = project_operator_to_channel(Kext, definition)
    nc = float(np.linalg.norm(coeff))
    nm = float(np.linalg.norm(soft_mode))
    if nc < 1e-14 or nm < 1e-14:
        return 0.0
    return float(abs(np.vdot(coeff, soft_mode)) / (nc * nm))


class FierzGWResidual:
    """Residual R=(Sigma-F[Sigma], N-Ntarget) at fixed Fierz mode and variable V."""

    def __init__(
        self,
        h0: np.ndarray,
        interaction_pairs,
        mode: str,
        grid: MatsubaraGrid,
        target: float,
    ):
        self.h0 = np.asarray(h0, dtype=complex)
        self.pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
        self.mode = str(mode)
        self.grid = grid
        self.target = float(target)
        self.norb = int(self.h0.shape[-1])
        self.codec = FierzStateCodec(self.norb, grid, target)

    def evaluate(self, x: np.ndarray, V: float) -> FierzResidualEvaluation:
        sigma_static, sigma_c, mu = self.codec.decode(x)
        definition = build_channel_definition(self.pairs, self.norb, float(V), self.mode)
        G = dyson_from_sigma_matrix(
            self.h0, self.grid, mu, sigma_static, sigma_c
        )
        rho = one_body_density_matrix_tail(
            G, self.grid, self.h0, mu, sigma_static
        )
        rho0 = np.asarray(rho[0, 0], dtype=complex)
        sigma_static_out, _, _ = channel_static_self_energy(rho0, definition)
        P = compute_channel_polarization(G, definition, self.grid)
        W = compute_channel_screened_interaction(P, definition)
        sigma_c_out = compute_channel_correlation_self_energy(
            G, W, definition, self.grid
        )
        r_static = sigma_static - sigma_static_out
        r_dynamic = sigma_c - sigma_c_out
        filling_error = float(np.trace(rho0).real - self.target)
        residual = self.codec.encode_residual(
            r_static, r_dynamic, filling_error
        )
        physical_residual = max(
            float(np.max(np.abs(r_static))),
            float(np.max(np.abs(r_dynamic))),
            abs(filling_error),
        )
        smin, soft_m, soft_Omega, soft_mode = screening_soft_mode(
            P, definition, self.grid
        )
        return FierzResidualEvaluation(
            residual=np.asarray(residual, dtype=float),
            G=np.asarray(G),
            P=np.asarray(P),
            W=np.asarray(W),
            sigma_static_out=np.asarray(sigma_static_out),
            sigma_c_out=np.asarray(sigma_c_out),
            rho=np.asarray(rho0),
            filling_error=filling_error,
            physical_residual=physical_residual,
            smin=smin,
            soft_m=soft_m,
            soft_Omega=soft_Omega,
            soft_mode=soft_mode,
        )

    def __call__(self, x: np.ndarray, V: float) -> np.ndarray:
        return self.evaluate(x, V).residual
