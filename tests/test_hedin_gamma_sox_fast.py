import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.hedin_gamma_fast import GammaPFeedbackOptions
from rubycgw.hedin_gamma_sox_fast import solve_matrix_gw_gamma_sox_feedback
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast


def test_gamma_p_sox_zero_interaction_reduces_to_free_background():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=1, T=.23)
    h = np.array([[.12, .17 + .01j], [.17 - .01j, -.09]], dtype=complex)
    h0 = h[None, None]
    Vq = np.zeros_like(h0)

    gw_opts = GWOptions(
        mu=.02,
        target_filling=None,
        max_iter=4,
        tol=1e-11,
        verbose=False,
        momentum_backend="direct",
    )
    bg = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    vopts = SupercellVertexOptions(
        max_iter=40,
        tol=1e-10,
        solver="gmres",
        gmres_restart=8,
        verbose=False,
        momentum_backend="direct",
    )
    out = solve_matrix_gw_gamma_sox_feedback(
        h0,
        Vq,
        grid,
        gw_opts=gw_opts,
        vertex_opts=vopts,
        feedback_opts=GammaPFeedbackOptions(
            max_iter=3,
            tol=1e-9,
            mixing_method="pulay",
            verbose=False,
        ),
        sox_opts=SOXOptions(n_quad=16, tail_complete=True),
        background=bg,
    )

    assert out.converged
    assert out.iterations == 1
    np.testing.assert_allclose(out.G, bg.G, rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(out.W, 0.0, atol=1e-14)
    np.testing.assert_allclose(out.Sigma_SOX, 0.0, atol=1e-14)
    np.testing.assert_allclose(out.Sigma_corr, bg.Sigma_GW, rtol=5e-12, atol=5e-12)
    assert not out.include_sox_vertex


def test_gamma_p_sox_result_exposes_resolved_self_energy_blocks():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=.25)
    h0 = np.array([[[[.10, .14], [.14, -.07]]]], dtype=complex).reshape(1, 1, 2, 2)
    Vq = np.zeros_like(h0)
    gw_opts = GWOptions(mu=.0, target_filling=None, max_iter=3, tol=1e-10,
                        verbose=False, momentum_backend="direct")
    bg = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    out = solve_matrix_gw_gamma_sox_feedback(
        h0, Vq, grid,
        gw_opts=gw_opts,
        vertex_opts=SupercellVertexOptions(max_iter=30, tol=1e-9, solver="gmres",
                                           gmres_restart=6, verbose=False,
                                           momentum_backend="direct"),
        feedback_opts=GammaPFeedbackOptions(max_iter=2, tol=1e-8, verbose=False),
        sox_opts=SOXOptions(n_quad=16),
        background=bg,
    )
    assert out.Sigma_GW.shape == out.G.shape
    assert out.Sigma_SOX.shape == out.G.shape
    assert out.Sigma_corr.shape == out.G.shape
    assert hasattr(out, "timing_totals")
    assert "sox" in out.timing_totals
