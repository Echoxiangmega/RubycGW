"""Exact bond-space reduction of the F-only response on a one-point cluster.

The production tail map is affine: Gamma = K + c[K] + L Gamma.
Only directed interacting bonds carry the static Fock correction. No projection
onto a selected current pattern and no Hermitian symmetrization are performed.
"""
import numpy as np
from .dynamic_cgw import _bare_vertex_field_iomega, _x_field_iomega
from .response_tail import build_tail_hf_context


def fock_bond_diagnostic(G, Vq, K, m, grid, reference, scales=(0., .25, .5, .75, 1.)):
    if grid.nk != 1:
        raise ValueError('bond reduction requires a single cluster momentum')
    ctx = build_tail_hf_context(reference, K, Vq, grid, m_ext=m,
                                include_hartree=False, include_fock=True, backend='direct')
    rows, cols = np.nonzero(np.abs(Vq[0, 0]) > 0)
    n = len(rows)

    def box(S):
        field = _bare_vertex_field_iomega(S, G, m, grid)
        return grid.T * _x_field_iomega(G, field, m, grid).sum(axis=0)[0, 0]

    def flinear(S):
        return (-Vq[0, 0] * box(S))[rows, cols]

    A = np.zeros((n, n), complex)
    for j, (a, b) in enumerate(zip(rows, cols)):
        E = np.zeros_like(K, dtype=complex)
        E[a, b] = 1
        A[:, j] = flinear(E)
    c = ctx.F_const[0, 0, rows, cols]
    b = c + flinear(K)

    # chi is measured with the same represented external bubble as production.
    def measure(S):
        return -np.trace(K @ box(S))

    weights = np.empty(n, complex)
    for j, (a, bcol) in enumerate(zip(rows, cols)):
        E = np.zeros_like(K, dtype=complex)
        E[a, bcol] = 1
        weights[j] = measure(E)
    bubble = measure(K)
    responses, residuals, gammas = [], [], []
    for alpha in scales:
        f = np.linalg.solve(np.eye(n) - alpha*A, alpha*b)
        gamma = np.array(K, dtype=complex, copy=True)
        gamma[rows, cols] += f
        responses.append(bubble + weights @ f)
        residuals.append(np.linalg.norm(f-alpha*(b+A@f)))
        gammas.append(gamma)
    eig, U = np.linalg.eig(A)
    modal_residue = (weights @ U) * np.linalg.solve(U, b)
    modal_response = modal_residue / (1 - eig)

    return dict(
        kernel=A, eigenvalues=eig, modal_residue=modal_residue,
        modal_response=modal_response,
        spectral_radius=np.max(np.abs(eig), initial=0),
        sigma_min=np.linalg.svd(np.eye(n)-A, compute_uv=False)[-1] if n else 1.,
        source=b, tail_source=c, rows=rows, cols=cols,
        bubble=bubble, first=weights@b, second=weights@A@b,
        tail_first=weights@c, scales=np.asarray(scales),
        response=np.asarray(responses), residual=np.asarray(residuals),
        gamma=np.asarray(gammas),
    )
