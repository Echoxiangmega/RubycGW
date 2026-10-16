"""Regression checks for the 36-form-factor V-only rank diagnostic."""
import numpy as np

from research.diagnose_v_only_vertex_rank import (
    effective_rank_diagnostic, embed_q0, embed_real_q, hermitian_form_factors,
    target_vectors,
)
from rubycgw.model import RubyParameters, ruby_interaction_bonds


def test_original_model_has_only_six_intratriangle_interactions():
    bonds = ruby_interaction_bonds(RubyParameters(V=0.7))
    pairs = {tuple(sorted((a, b))) for a, b, _, _ in bonds}
    assert pairs == {
        (0, 1), (0, 2), (1, 2),
        (3, 4), (3, 5), (4, 5),
    }
    assert all(tuple(R) == (0, 0) and V == 0.7 for _, _, R, V in bonds)


def test_form_factors_are_orthonormal_and_have_correct_physical_labels():
    mats, labels, kinds = hermitian_form_factors()
    assert mats.shape == (36, 6, 6)
    assert (sum(kinds == "density"), sum(kinds == "real_bond"),
            sum(kinds == "imag_bond")) == (6, 15, 15)
    gram = np.einsum("aij,bji->ab", mats, mats)
    np.testing.assert_allclose(gram, np.eye(36), atol=1e-14)
    q0 = embed_q0(mats, cells=2)
    assert q0.shape == (36, 12, 12)
    np.testing.assert_allclose(
        np.einsum("aij,bji->ab", q0, q0), np.eye(36), atol=1e-14
    )
    refs = target_vectors(mats)
    np.testing.assert_allclose(
        np.dot(refs["LC_same"], refs["LC_opposite"]), 0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.dot(refs["LC_same"], refs["CO_AminusB"]), 0, atol=1e-14
    )


def test_rank_two_correction_reconstructs_static_target():
    n = 36
    chi_cgw = np.eye(n) * 1.4
    # Here the GW bubble is identity so the metric does not rotate the modes.
    delta = np.zeros((n, n))
    delta[0, 0] = .14
    delta[1, 1] = -.08
    chi_ed = np.linalg.inv(np.linalg.inv(chi_cgw) - delta)
    result = effective_rank_diagnostic(
        chi_ed, chi_cgw, np.eye(n), np.eye(n)
    )
    assert result["capture"][1] > 1 - 1e-12
    np.testing.assert_allclose(
        result["chi_reconstructed"][0], chi_cgw, atol=1e-12
    )
    np.testing.assert_allclose(
        result["chi_reconstructed"][2], chi_ed, atol=1e-12
    )


def test_driver_invocation_from_checkout_without_editable_install(tmp_path):
    """An isolated Python process cannot rely on cwd or PYTHONPATH for imports."""
    import subprocess
    import sys
    from pathlib import Path

    driver = Path(__file__).resolve().parents[1] / "research" / "diagnose_v_only_vertex_rank.py"
    proc = subprocess.run(
        [sys.executable, "-I", str(driver), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--filling" in proc.stdout


def test_commensurate_q_pi_preserves_orthonormal_hermitian_basis():
    mats, _, _ = hermitian_form_factors()
    coord = np.array([[0, 0], [1, 0]], dtype=int)
    q_zero = embed_real_q(mats, coord, (0.0, 0.0))
    q_pi = embed_real_q(mats, coord, (0.5, 0.0))
    np.testing.assert_allclose(q_zero, embed_q0(mats, 2))
    for operators in (q_zero, q_pi):
        np.testing.assert_allclose(operators, operators.conj().swapaxes(-1, -2))
        np.testing.assert_allclose(
            np.einsum("aij,bji->ab", operators, operators),
            np.eye(36), atol=1e-13
        )
    # q=0 and q=pi operators are orthogonal on the two-cell torus.
    np.testing.assert_allclose(
        np.einsum("aij,bji->ab", q_zero, q_pi),
        np.zeros((36, 36)), atol=1e-13
    )
