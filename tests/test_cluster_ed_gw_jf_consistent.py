from types import SimpleNamespace

import numpy as np

import rubycgw.cluster_ed_gw_jf as jf
from rubycgw.cluster_ed_gw_jf_consistent import (
    _impurity_vertex,
    _static_net_field,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.response_tail import (
    build_tail_hf_context,
    build_tail_reference,
    reference_response_infinite,
)


def test_consistency_layer_is_installed_package_wide():
    assert getattr(jf, "_tail_consistent_cluster_jf_installed", False)


def test_tail_context_recovers_exact_reference_response_outside_box():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=1, nOmega=1, T=0.23)
    h0 = np.diag([-0.7, -0.2, 0.15, 0.5, 0.9, 1.3]).astype(complex)[None, None]
    sigma_h = np.zeros((6, 6), dtype=complex)
    ref = build_tail_reference(h0, mu=0.11, sigma_h=sigma_h, grid=grid)

    K = np.zeros((6, 6), dtype=complex)
    K[0, 0], K[1, 1], K[2, 2] = 2.0, -1.0, -1.0
    ctx = build_tail_hf_context(
        ref,
        K,
        np.zeros((1, 1, 6, 6), dtype=complex),
        grid,
        q_index=(0, 0),
        include_hartree=False,
        include_fock=False,
    )
    X = np.einsum(
        "nxyia,ab,nxybj->nxyij",
        ref.gref,
        K,
        ref.gref,
        optimize=True,
    )
    _, _, R = ctx.total_static_parts(X)
    exact = reference_response_infinite(ref, K, grid, q_index=(0, 0))
    assert np.allclose(R, exact, rtol=2e-12, atol=2e-12)


def test_local_ruby_static_hf_cancels_cluster_double_counting_with_tail():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.2)
    h0 = np.zeros((2, 2, 6, 6), dtype=complex)
    for i in range(6):
        h0[..., i, i] = -0.4 + 0.17 * i
    sigma_h = np.zeros((6, 6), dtype=complex)
    ref = build_tail_reference(h0, mu=0.05, sigma_h=sigma_h, grid=grid)

    # Primitive Ruby density interactions are cell-local, hence V(q) is
    # independent of q.  Use a generic symmetric local interaction matrix.
    Vc = np.zeros((6, 6), dtype=complex)
    for i in range(6):
        Vc[i, (i + 1) % 6] = 0.4
        Vc[(i + 1) % 6, i] = 0.4
    Vq = np.broadcast_to(Vc, (2, 2, 6, 6)).copy()

    K = np.diag([2.0, -1.0, -1.0, 0.0, 0.0, 0.0]).astype(complex)
    ctx = build_tail_hf_context(
        ref,
        K,
        Vq,
        grid,
        q_index=(1, 0),
        include_hartree=True,
        include_fock=True,
    )

    rng = np.random.default_rng(4)
    X = 1e-2 * (
        rng.normal(size=(grid.nf, 2, 2, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 2, 2, 6, 6))
    )
    op = SimpleNamespace(G=np.zeros_like(X), Vc=Vc)
    field, _ = _static_net_field(op, X, ctx)
    assert np.max(np.abs(field)) < 2e-11


def test_source_dependent_impurity_affine_part_drops_out_of_linear_kernel():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=1, nOmega=1, T=0.3)
    eye = np.eye(6, dtype=complex)

    class FakeTangent:
        def __init__(self):
            self.Gc_inv = np.broadcast_to(eye, (grid.nf, 6, 6)).copy()
            self.inner_matrix = np.asarray([[1.7]])
            self.sigma_modes = np.zeros((1, grid.nf, 6, 6), dtype=complex)
            self.sigma_modes[0, :, 0, 0] = 0.31

        def coefficients_from_dg0_inv(self, d):
            # Any real-linear functional is sufficient for this affine test.
            return np.asarray([np.real(np.sum(d)) / 19.0])

    op = SimpleNamespace(bath_tangent=FakeTangent())
    rng = np.random.default_rng(7)
    dgc = 1e-3 * (
        rng.normal(size=(grid.nf, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 6, 6))
    )
    K = np.diag([2.0, -1.0, -1.0, 0.0, 0.0, 0.0]).astype(complex)
    direct = np.zeros_like(dgc)
    direct[:, 1, 1] = 0.13

    total = _impurity_vertex(op, dgc, K, direct)
    const = _impurity_vertex(op, np.zeros_like(dgc), K, direct)
    linear = _impurity_vertex(
        op, dgc, np.zeros_like(K), np.zeros_like(direct)
    )
    assert np.allclose(total - const, linear, rtol=1e-13, atol=1e-13)
