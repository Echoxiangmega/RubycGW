from pathlib import Path
import sys

import numpy as np

NATIVE = Path(__file__).resolve().parents[1] / "ed_benchmark" / "native_ed"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

import analyze_strong_coupling as asc
from core import make_states
from geometry import RubyPBC


def _local_chiral_doublet(states, ns, sites, physical_b=False):
    w = np.exp(2j * np.pi / 3.0)
    plus = np.asarray([1.0, w, w**2], complex) / np.sqrt(3.0)
    minus = np.asarray([1.0, w**2, w], complex) / np.sqrt(3.0)
    coeffs = (minus, plus) if physical_b else (plus, minus)
    U = np.zeros((len(states), 2), complex)
    for col, coeff in enumerate(coeffs):
        for site, z in zip(sites, coeff):
            bit = np.uint64(1 << (ns - 1 - site))
            i = int(np.searchsorted(states, bit))
            assert states[i] == bit
            U[i, col] = z
    return U


def _project_local_ops(g, states, U, cell, sub):
    cols = []
    cache = None
    for psi in U.T:
        actions, cache = asc.pseudospin_actions(g, states, psi, cache)
        cols.append(actions)
    loops, cells, subs, _ = cache
    p = next(i for i, (c, s) in enumerate(zip(cells, subs))
             if int(c) == cell and int(s) == sub)
    mats = []
    for a in range(3):
        M = np.column_stack([z[:, 3*p+a] for z in cols])
        mats.append(U.conj().T @ M)
    return mats


def test_microscopic_triangle_proxies_project_to_physical_paulis_A_and_B():
    g = RubyPBC("ring3", t=0.4, t1=0.2, t2=-0.2)
    states = make_states(g.Ns, 1)
    px = np.array([[0, 1], [1, 0]], complex)
    py = np.array([[0, -1j], [1j, 0]], complex)
    pz = np.array([[1, 0], [0, -1]], complex)

    Ua = _local_chiral_doublet(states, g.Ns, [0, 1, 2], physical_b=False)
    Ub = _local_chiral_doublet(states, g.Ns, [3, 4, 5], physical_b=True)

    for got, want in zip(_project_local_ops(g, states, Ua, 0, 0), (px, py, pz)):
        np.testing.assert_allclose(got, want, atol=1e-12)
    for got, want in zip(_project_local_ops(g, states, Ub, 0, 1), (px, py, pz)):
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_spectrum_plateau_fit_recovers_constant_scaled_gap():
    rows = []
    for V in (2.0, 3.0, 4.0):
        rows.append({
            "V": V,
            "gap_E1": 2.5/V,
            "V_gap_E1": 2.5,
        })
    fit = asc._plateau_diagnostics(rows, 2.0)
    assert len(fit) == 1
    assert abs(fit[0]["plateau_intercept_1overV_to_0"] - 2.5) < 1e-12
    assert fit[0]["relative_scaled_spread"] < 1e-12
