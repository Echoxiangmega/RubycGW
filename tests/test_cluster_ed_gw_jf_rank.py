import numpy as np

from rubycgw.cluster_ed_gw_jf import BathTangentModel
from rubycgw.cluster_ed_gw_jf_rank import (
    effective_bath_tangent_rank,
    truncate_bath_tangent_model,
)


def _model(rank=4):
    rng = np.random.default_rng(1234)
    nf, norb, nfit = 3, 2, 2
    nparam = 9
    packed_fit = 2 * nfit * norb * norb

    # Orthonormal columns are not required for the truncation algebra tested
    # here; the model only needs internally consistent retained coordinates.
    Ufit = rng.normal(size=(packed_fit, rank))
    singular = np.linspace(2.0, 0.8, rank)
    modes = rng.normal(size=(nparam, rank))
    sigma_modes = rng.normal(size=(rank, nf, norb, norb)) + 1j * rng.normal(
        size=(rank, nf, norb, norb)
    )
    inner = np.eye(rank) + 0.03 * rng.normal(size=(rank, rank))
    Gc_inv = np.broadcast_to(np.eye(norb), (nf, norb, norb)).astype(complex).copy()
    return BathTangentModel(
        theta0=np.zeros(nparam),
        mode_vectors=modes,
        singular_values=singular,
        Ufit=Ufit,
        sigma_modes=sigma_modes,
        inner_matrix=inner,
        fit_indices=np.arange(nfit),
        fit_weights=np.ones(nfit),
        fit_metric="delta",
        model_g0_fit=np.broadcast_to(np.eye(norb), (nfit, norb, norb)).astype(complex),
        Gc_inv=Gc_inv,
        rank=rank,
        condition_number=float(np.linalg.cond(inner)),
        build_seconds=12.3,
        fd_step=2e-4,
    )


def test_nonpositive_rank_means_full_and_large_rank_clips():
    model = _model(4)
    assert effective_bath_tangent_rank(model, 0) == 4
    assert effective_bath_tangent_rank(model, -3) == 4
    assert effective_bath_tangent_rank(model, 99) == 4
    assert truncate_bath_tangent_model(model, 0) is model
    assert truncate_bath_tangent_model(model, 99) is model


def test_truncated_coordinates_are_leading_full_coordinates():
    model = _model(4)
    small = truncate_bath_tangent_model(model, 2)
    rng = np.random.default_rng(55)
    dg0 = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))

    np.testing.assert_allclose(
        small.coefficients_from_dg0_inv(dg0),
        model.coefficients_from_dg0_inv(dg0)[:2],
    )
    np.testing.assert_allclose(small.inner_matrix, model.inner_matrix[:2, :2])
    np.testing.assert_allclose(small.sigma_modes, model.sigma_modes[:2])
    assert small.rank == 2
    assert np.isclose(small.condition_number, np.linalg.cond(model.inner_matrix[:2, :2]))


def test_truncated_impurity_response_matches_manual_projected_solve():
    model = _model(4)
    small = truncate_bath_tangent_model(model, 2)
    rng = np.random.default_rng(77)
    dgc = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))

    dg0 = -np.einsum(
        "nab,nbc,ncd->nad",
        model.Gc_inv,
        dgc,
        model.Gc_inv,
        optimize=True,
    )
    rhs = model.coefficients_from_dg0_inv(dg0)[:2]
    coeff = np.linalg.solve(model.inner_matrix[:2, :2], rhs)
    expected = np.einsum("r,rnab->nab", coeff, model.sigma_modes[:2], optimize=True)

    np.testing.assert_allclose(small.impurity_sigma_from_delta_gc(dgc), expected)
