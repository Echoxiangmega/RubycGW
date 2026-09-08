"""Analytic high-frequency tail for covariant response calculations.

Production SC-GW evaluates the equal-time density matrix entering Hartree and
static bare-V Fock with the reference subtraction

    rho = rho_ref + T sum_n [G - G_ref],

where

    G_ref^{-1}(k,iw) = iw + mu - h0(k) - Sigma_H.

The historical cGW kernel instead used only the represented finite-frequency
sum of X=G Gamma G.  This module differentiates the *same tail-subtracted
production map*.  The derivative is slightly subtle because G_ref depends
explicitly on the external source through h0 and implicitly through Sigma_H.
For a transfer Q=(q,iOmega), define the positive tangent convention used by the
cGW code

    X(k;Q) = G(k+Q) Gamma(k;Q) G(k) = - dG/dh.

Then the equal-time response tangent entering H/F is

    R(k;Q) = T_box sum X + C_Q[K + Gamma_H],

with

    C_Q[S] = T_infty sum G_ref(k+Q) S G_ref(k)
             - T_box sum G_ref(k+Q) S G_ref(k).

The Hartree tangent obeys a small linear equation because Gamma_H appears in
its own reference tail.  That equation is solved exactly in orbital space.
The Fock tangent is then the usual static convolution of R with the bare V.

This is a numerical-consistency correction: MT/AL continue to differentiate
the explicitly represented retarded W-V part exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .supercell_gw import _fermi
from .supercell_gw_split import compute_static_fock_matrix


@dataclass(frozen=True)
class TailReference:
    """Static reference eigensystem and Green function used by production GW."""

    h0: np.ndarray
    mu: float
    sigma_h: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray
    gref: np.ndarray


def build_tail_reference(
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
    grid: MatsubaraGrid,
) -> TailReference:
    h0 = np.asarray(h0, dtype=complex)
    sigma_h = np.asarray(sigma_h, dtype=complex)
    norb = int(h0.shape[-1])
    if h0.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected h0 shape for tail reference")
    if sigma_h.shape != (norb, norb):
        raise ValueError("unexpected sigma_h shape for tail reference")

    href = h0 + sigma_h[None, None, :, :]
    href = 0.5 * (href + np.swapaxes(href.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(href)
    denom = (
        1j * grid.omega[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    gref = np.einsum(
        "xyia,nxya,xyja->nxyij",
        evecs,
        1.0 / denom,
        evecs.conj(),
        optimize=True,
    )
    return TailReference(
        h0=np.array(h0, copy=True),
        mu=float(mu),
        sigma_h=np.array(sigma_h, copy=True),
        evals=np.asarray(evals),
        evecs=np.asarray(evecs),
        gref=np.asarray(gref),
    )


def _normalize_q(q_index, grid: MatsubaraGrid) -> tuple[int, int]:
    arr = np.asarray(q_index, dtype=int).reshape(2)
    return int(arr[0] % grid.nk1), int(arr[1] % grid.nk2)


def _source_kfield(source: np.ndarray, grid: MatsubaraGrid, norb: int) -> np.ndarray:
    arr = np.asarray(source, dtype=complex)
    if arr.shape == (norb, norb):
        return np.broadcast_to(arr, (grid.nk1, grid.nk2, norb, norb)).copy()
    if arr.shape == (grid.nk1, grid.nk2, norb, norb):
        return np.array(arr, copy=True)
    raise ValueError(
        "tail source must be an orbital matrix or a static k-resolved matrix field"
    )


def reference_response_infinite(
    reference: TailReference,
    source: np.ndarray,
    grid: MatsubaraGrid,
    q_index=(0, 0),
    m_ext: int = 0,
) -> np.ndarray:
    """Return T*sum_{all iw} G_ref(k+Q) source G_ref(k) analytically.

    The result has shape ``(nk1,nk2,norb,norb)`` and follows the same positive
    tangent convention R=-d rho/dh used by the H/F cGW pieces.
    """
    ip1, ip2 = _normalize_q(q_index, grid)
    m_ext = int(m_ext)
    norb = int(reference.evals.shape[-1])
    S = _source_kfield(source, grid, norb)

    eval0 = reference.evals
    U0 = reference.evecs
    evalq = np.roll(eval0, shift=(-ip1, -ip2), axis=(0, 1))
    Uq = np.roll(U0, shift=(-ip1, -ip2), axis=(0, 1))

    f0 = _fermi(eval0 - float(reference.mu), grid.T)
    fq = _fermi(evalq - float(reference.mu), grid.T)
    omega_ext = 2.0 * np.pi * float(grid.T) * float(m_ext)

    # band indices: a belongs to k+q, b belongs to k
    numerator = f0[..., None, :] - fq[..., :, None]
    denominator = (
        1j * omega_ext
        + eval0[..., None, :]
        - evalq[..., :, None]
    )
    coeff = np.empty_like(denominator, dtype=complex)
    small = np.abs(denominator) < 1.0e-12
    coeff[~small] = numerator[~small] / denominator[~small]
    if np.any(small):
        if m_ext != 0:
            # Nonzero bosonic Matsubara frequency cannot be genuinely singular;
            # this branch is only a roundoff safeguard.
            coeff[small] = 0.0
        else:
            # lim_{eb->ea} [f(eb)-f(ea)]/(eb-ea) = f'(e)
            fmid = 0.5 * (
                np.broadcast_to(f0[..., None, :], denominator.shape)
                + np.broadcast_to(fq[..., :, None], denominator.shape)
            )
            deriv = -fmid * (1.0 - fmid) / float(grid.T)
            coeff[small] = deriv[small]

    Stilde = np.einsum(
        "xyia,xyij,xyjb->xyab",
        Uq.conj(),
        S,
        U0,
        optimize=True,
    )
    Reig = coeff * Stilde
    return np.einsum(
        "xyia,xyab,xyjb->xyij",
        Uq,
        Reig,
        U0.conj(),
        optimize=True,
    )


def reference_response_box(
    reference: TailReference,
    source: np.ndarray,
    grid: MatsubaraGrid,
    q_index=(0, 0),
    m_ext: int = 0,
) -> np.ndarray:
    """Return the represented finite-box reference bubble per base momentum."""
    ip1, ip2 = _normalize_q(q_index, grid)
    norb = int(reference.evals.shape[-1])
    S = _source_kfield(source, grid, norb)
    src, dst = frequency_shift_slices(grid.nf, int(m_ext))
    out = np.zeros((grid.nk1, grid.nk2, norb, norb), dtype=complex)
    if src.stop == src.start:
        return out
    Gq = roll_spatial(reference.gref[src], ip1, ip2)
    G0 = reference.gref[dst]
    Xref = np.einsum(
        "nxyia,xyab,nxybj->nxyij",
        Gq,
        S,
        G0,
        optimize=True,
    )
    return float(grid.T) * np.sum(Xref, axis=0)


def reference_tail_remainder(
    reference: TailReference,
    source: np.ndarray,
    grid: MatsubaraGrid,
    q_index=(0, 0),
    m_ext: int = 0,
) -> np.ndarray:
    """Analytic missing tail C_Q[source] = infinite reference bubble - box."""
    return reference_response_infinite(
        reference, source, grid, q_index=q_index, m_ext=m_ext
    ) - reference_response_box(
        reference, source, grid, q_index=q_index, m_ext=m_ext
    )


def _negative_q(q_index, grid: MatsubaraGrid) -> tuple[int, int]:
    ip1, ip2 = _normalize_q(q_index, grid)
    return (-ip1) % grid.nk1, (-ip2) % grid.nk2


def _diag_average(response_k: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    diag = np.diagonal(np.asarray(response_k), axis1=-2, axis2=-1)
    return np.sum(diag, axis=(0, 1)) / float(grid.nk)


def _masked_static_field(
    static: np.ndarray,
    template: np.ndarray,
    grid: MatsubaraGrid,
    m_ext: int,
) -> np.ndarray:
    arr = np.asarray(static, dtype=complex)
    norb = int(template.shape[-1])
    if arr.shape == (norb, norb):
        kfield = np.broadcast_to(arr, (grid.nk1, grid.nk2, norb, norb))
    elif arr.shape == (grid.nk1, grid.nk2, norb, norb):
        kfield = arr
    else:
        raise ValueError("unexpected static response field shape")
    out = np.zeros_like(template, dtype=complex)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        out[dst] = np.broadcast_to(kfield, out[dst].shape)
    return out


@dataclass
class TailHFContext:
    """Precomputed affine H/F tail tangent for one external source and Q."""

    reference: TailReference
    source_k: np.ndarray
    Vq: np.ndarray
    grid: MatsubaraGrid
    q_index: tuple[int, int]
    m_ext: int
    backend: str
    include_hartree: bool
    include_fock: bool
    C_source: np.ndarray
    C_diag: np.ndarray
    hartree_lhs: np.ndarray
    hartree_b_source: np.ndarray
    H_const: np.ndarray
    F_const: np.ndarray

    def _hartree_map(self, response_k: np.ndarray) -> np.ndarray:
        im1, im2 = _negative_q(self.q_index, self.grid)
        return np.asarray(self.Vq[im1, im2]) @ _diag_average(response_k, self.grid)

    def total_static_parts(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total (H_matrix, F_k, R_k) including source-tail constants."""
        R_num = float(self.grid.T) * np.sum(np.asarray(X), axis=0)
        norb = int(R_num.shape[-1])

        if self.include_hartree:
            rhs = self._hartree_map(R_num) + self.hartree_b_source
            try:
                hvec = np.linalg.solve(self.hartree_lhs, rhs)
            except np.linalg.LinAlgError:
                hvec = np.linalg.lstsq(self.hartree_lhs, rhs, rcond=None)[0]
            H = np.zeros((norb, norb), dtype=complex)
            H[np.diag_indices(norb)] = hvec
            R = R_num + self.C_source + np.einsum(
                "a,axyij->xyij", hvec, self.C_diag, optimize=True
            )
        else:
            H = np.zeros((norb, norb), dtype=complex)
            R = R_num + self.C_source

        if self.include_fock:
            F = compute_static_fock_matrix(
                R, self.Vq, self.grid, backend=self.backend
            )
        else:
            F = np.zeros_like(R)
        return H, F, R

    def linear_vertex_parts(self, X: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return only the Gamma-dependent H/F pieces for a linear solver."""
        H, F, _ = self.total_static_parts(X)
        Hlin = H - self.H_const
        Flin = F - self.F_const
        return (
            _masked_static_field(Hlin, template, self.grid, self.m_ext),
            _masked_static_field(Flin, template, self.grid, self.m_ext),
        )

    def constant_vertex_parts(self, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            _masked_static_field(self.H_const, template, self.grid, self.m_ext),
            _masked_static_field(self.F_const, template, self.grid, self.m_ext),
        )

    def total_vertex_parts(self, X: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, F, _ = self.total_static_parts(X)
        return (
            _masked_static_field(H, template, self.grid, self.m_ext),
            _masked_static_field(F, template, self.grid, self.m_ext),
        )


def build_tail_hf_context(
    reference: TailReference,
    source: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    q_index=(0, 0),
    m_ext: int = 0,
    backend: str = "fft",
    include_hartree: bool = True,
    include_fock: bool = True,
) -> TailHFContext:
    q_index = _normalize_q(q_index, grid)
    norb = int(reference.evals.shape[-1])
    source_k = _source_kfield(source, grid, norb)
    Vq = np.asarray(Vq, dtype=complex)
    C_source = reference_tail_remainder(
        reference, source_k, grid, q_index=q_index, m_ext=m_ext
    )

    if include_hartree:
        C_diag = np.empty(
            (norb, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
        A = np.empty((norb, norb), dtype=complex)
        im1, im2 = _negative_q(q_index, grid)
        Vminus = Vq[im1, im2]
        for j in range(norb):
            Ej = np.zeros((norb, norb), dtype=complex)
            Ej[j, j] = 1.0
            Cj = reference_tail_remainder(
                reference, Ej, grid, q_index=q_index, m_ext=m_ext
            )
            C_diag[j] = Cj
            A[:, j] = Vminus @ _diag_average(Cj, grid)
        lhs = np.eye(norb, dtype=complex) - A
        b_source = Vminus @ _diag_average(C_source, grid)
        try:
            h0 = np.linalg.solve(lhs, b_source)
        except np.linalg.LinAlgError:
            h0 = np.linalg.lstsq(lhs, b_source, rcond=None)[0]
        H_const = np.zeros((norb, norb), dtype=complex)
        H_const[np.diag_indices(norb)] = h0
        R_const = C_source + np.einsum(
            "a,axyij->xyij", h0, C_diag, optimize=True
        )
    else:
        C_diag = np.zeros(
            (norb, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
        lhs = np.eye(norb, dtype=complex)
        b_source = np.zeros(norb, dtype=complex)
        H_const = np.zeros((norb, norb), dtype=complex)
        R_const = C_source

    if include_fock:
        F_const = compute_static_fock_matrix(
            R_const, Vq, grid, backend=backend
        )
    else:
        F_const = np.zeros((grid.nk1, grid.nk2, norb, norb), dtype=complex)

    return TailHFContext(
        reference=reference,
        source_k=source_k,
        Vq=Vq,
        grid=grid,
        q_index=q_index,
        m_ext=int(m_ext),
        backend=str(backend),
        include_hartree=bool(include_hartree),
        include_fock=bool(include_fock),
        C_source=C_source,
        C_diag=C_diag,
        hartree_lhs=lhs,
        hartree_b_source=b_source,
        H_const=H_const,
        F_const=F_const,
    )


__all__ = [
    "TailReference",
    "TailHFContext",
    "build_tail_reference",
    "reference_response_infinite",
    "reference_response_box",
    "reference_tail_remainder",
    "build_tail_hf_context",
]
