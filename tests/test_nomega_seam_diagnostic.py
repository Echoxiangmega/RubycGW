import numpy as np

from diagnose_nomega_seam import _expected_seam_abs_omega, _seam_distance


def test_expected_seam_frequency_matches_fermionic_index_definition():
    T = 0.05
    for nOmega in (8, 10, 12, 14):
        expected = (2 * nOmega + 1) * np.pi * T
        assert np.isclose(_expected_seam_abs_omega(nOmega, T), expected)


def test_seam_distance_identifies_both_adjacent_fermion_indices():
    nOmega = 10
    assert _seam_distance(10, nOmega) == 0
    assert _seam_distance(-11, nOmega) == 0
    assert _seam_distance(9, nOmega) == 1
    assert _seam_distance(-12, nOmega) == 1
