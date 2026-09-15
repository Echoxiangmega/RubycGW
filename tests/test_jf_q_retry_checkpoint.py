from types import SimpleNamespace

import numpy as np

import scan_cluster_ed_gw_jf_q as scan
from rubycgw.cluster_ed_gw_jf import ClusterJFOptions
from rubycgw.grids import MatsubaraGrid


class _FakeOperator:
    def __init__(self):
        self.opts = ClusterJFOptions(
            solver="gcrotmk",
            maxiter=80,
            restart=24,
            recycle_dim=12,
            verbose=False,
        )


def _retry_args(**updates):
    data = dict(
        no_rhs_recycle=False,
        no_auto_retry=False,
        retry_maxiter=300,
        retry_krylov_m=48,
        no_gmres_fallback=False,
    )
    data.update(updates)
    return SimpleNamespace(**data)


def _success(nvertex):
    chi = np.eye(nvertex, dtype=complex)
    results = [SimpleNamespace(Gamma=np.zeros((1,)), iterations=1, final_error=1e-12)] * nvertex
    return chi, results


def test_q_retry_uses_fresh_enlarged_krylov_and_restores_options(monkeypatch):
    op = _FakeOperator()
    original = op.opts
    vertices = np.zeros((2, 6, 6), dtype=complex)
    calls = []

    def fake_response(operator, _vertices, _q, *, initial_gammas, recycle):
        calls.append(
            (
                operator.opts.solver,
                operator.opts.maxiter,
                operator.opts.restart,
                list(initial_gammas),
                recycle,
            )
        )
        if len(calls) == 1:
            raise RuntimeError("soft primary")
        return _success(len(_vertices))

    monkeypatch.setattr(scan, "response_matrix", fake_response)
    initial = [np.ones((1,)), np.ones((1,))]
    chi, _results, nretry, solver = scan._solve_q_with_retry(
        op, vertices, (0, 1), initial, _retry_args()
    )

    assert np.allclose(chi, np.eye(2))
    assert nretry == 1
    assert solver == "gcrotmk"
    assert calls[0][0:3] == ("gcrotmk", 80, 24)
    assert calls[0][4] is True
    assert calls[1][0:3] == ("gcrotmk", 300, 48)
    assert calls[1][4] is False
    assert all(x is None for x in calls[1][3])
    assert op.opts is original


def test_q_retry_falls_back_to_fresh_gmres(monkeypatch):
    op = _FakeOperator()
    original = op.opts
    vertices = np.zeros((1, 6, 6), dtype=complex)
    calls = []

    def fake_response(operator, _vertices, _q, *, initial_gammas, recycle):
        calls.append((operator.opts.solver, operator.opts.maxiter, operator.opts.restart, recycle))
        if len(calls) < 3:
            raise RuntimeError(f"fail {len(calls)}")
        return _success(len(_vertices))

    monkeypatch.setattr(scan, "response_matrix", fake_response)
    _chi, _results, nretry, solver = scan._solve_q_with_retry(
        op, vertices, (0, 1), [None], _retry_args()
    )

    assert nretry == 2
    assert solver == "gmres"
    assert calls == [
        ("gcrotmk", 80, 24, True),
        ("gcrotmk", 300, 48, False),
        ("gmres", 300, 48, False),
    ]
    assert op.opts is original


def test_partial_analysis_keeps_unfinished_q_as_nan():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=1, nOmega=0, T=0.1)
    q_points = [(0, 0), (0, 1)]
    chi = np.full((2, 2, 2), np.nan + 1j * np.nan, dtype=complex)
    chi[0] = np.diag([2.0, 1.0])
    completed = np.asarray([True, False])

    chi_h, pair_complete, eig, _vec, leading, _proj = scan._analyse_completed(
        chi, completed, q_points, grid, ["Ax", "Ay"]
    )

    assert np.allclose(chi_h[0], chi[0])
    assert pair_complete[0]
    assert leading[0] == 2.0
    assert np.all(np.isnan(eig[1]))
    assert np.isnan(leading[1])


def test_partial_checkpoint_name():
    assert scan._partial_path("results/test.npz").name == "test.partial.npz"
    assert scan._partial_path("results/test").name == "test.partial.npz"
