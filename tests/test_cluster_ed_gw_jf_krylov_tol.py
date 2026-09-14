import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

import rubycgw.cluster_ed_gw_jf as jfmod
from rubycgw.grids import MatsubaraGrid


def _dummy_operator(solver: str, tol: float):
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=1, nOmega=1, T=0.5)
    op = object.__new__(jfmod.ClusterEmbeddedJacobian)
    op.G = np.zeros((grid.nf, 1, 1, 6, 6), dtype=complex)
    op.grid = grid
    op.opts = jfmod.ClusterJFOptions(
        solver=solver,
        tol=tol,
        maxiter=8,
        restart=4,
        recycle_dim=2,
        verbose=False,
    )

    n = 2 * int(np.prod(op.G.shape))
    op.linear_operator = lambda q: LinearOperator(
        (n, n), matvec=lambda x: np.asarray(x, dtype=float), dtype=float
    )
    op.apply_A = lambda gamma, q: np.asarray(gamma, dtype=complex)
    return op


def _fake_converged_vector(b: np.ndarray, relative_tol: float):
    """Return an info=0 vector at half the solver's Euclidean threshold."""
    threshold = float(relative_tol) * float(np.linalg.norm(b))
    residual = np.zeros_like(b, dtype=float)
    residual[0] = 0.5 * threshold
    return np.asarray(b, dtype=float) - residual, 0


@pytest.mark.parametrize("solver_name, symbol", [("gcrotmk", "gcrotmk"), ("gmres", "gmres")])
def test_modern_scipy_krylov_tolerance_implies_final_maxabs(monkeypatch, solver_name, symbol):
    tol = 1.0e-8
    op = _dummy_operator(solver_name, tol)
    K = np.eye(6, dtype=complex)
    b = jfmod._pack_complex(np.broadcast_to(K, op.G.shape))
    bnorm = float(np.linalg.norm(b))
    assert bnorm > 2.0  # old rtol=tol would allow a max residual above tol here

    seen = {}

    def fake_solver(A, rhs, **kwargs):
        rel = float(kwargs["rtol"])
        seen["rtol"] = rel
        return _fake_converged_vector(rhs, rel)

    monkeypatch.setattr(jfmod, symbol, fake_solver)
    result = op.solve(K, (0, 0))

    assert result.solver_info == 0
    assert result.converged
    assert result.final_error <= tol
    assert np.isclose(seen["rtol"] * bnorm, tol, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("solver_name, symbol", [("gcrotmk", "gcrotmk"), ("gmres", "gmres")])
def test_legacy_scipy_tol_fallback_uses_same_maxabs_scaling(monkeypatch, solver_name, symbol):
    tol = 1.0e-8
    op = _dummy_operator(solver_name, tol)
    K = np.eye(6, dtype=complex)
    b = jfmod._pack_complex(np.broadcast_to(K, op.G.shape))
    bnorm = float(np.linalg.norm(b))

    seen = {"modern_calls": 0, "legacy_calls": 0}

    def fake_legacy_solver(A, rhs, **kwargs):
        if "rtol" in kwargs:
            seen["modern_calls"] += 1
            raise TypeError("simulate an older SciPy API that only accepts tol=")
        rel = float(kwargs["tol"])
        seen["legacy_calls"] += 1
        seen["tol"] = rel
        return _fake_converged_vector(rhs, rel)

    monkeypatch.setattr(jfmod, symbol, fake_legacy_solver)
    result = op.solve(K, (0, 0))

    assert seen["modern_calls"] == 1
    assert seen["legacy_calls"] == 1
    assert result.solver_info == 0
    assert result.converged
    assert result.final_error <= tol
    assert np.isclose(seen["tol"] * bnorm, tol, rtol=1e-13, atol=0.0)
