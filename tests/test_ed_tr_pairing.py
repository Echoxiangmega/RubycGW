"""Regression tests for time-reversed symmetry-block recycling in ED."""
from pathlib import Path
import sys

import numpy as np
import pytest

NATIVE = Path(__file__).resolve().parents[1] / "ed_benchmark" / "native_ed"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from geometry import RubyPBC
from scan_susceptibility import VScanSolver, _symmetry_label_key, _time_reversed_block_key
from symmetry import generators_trans, sector_labels


@pytest.mark.parametrize("cluster,nf", [
    ("ring3", 2), ("ring3", 3),
    ("2x2", 2), ("2x2", 3),
])
def test_tr_pairing_matches_original_symmetry_sector_spectrum(cluster, nf):
    g=RubyPBC(cluster,t=0.4,t1=0.2,t2=-0.2)
    old=VScanSolver(g,nf,symmetry="point",low=3,max_basis=6000,
                    time_reversal_pairs=False)
    fast=VScanSolver(g,nf,symmetry="point",low=3,max_basis=6000,
                     time_reversal_pairs=True)
    Eref = None
    for V in (0.4, 2.0):
        H0,e0,psis0,rows0,incomplete0,res0=old.solve(V,1e-7)
        H1,e1,psis1,rows1,incomplete1,res1=fast.solve(V,1e-7)
        np.testing.assert_allclose(e1,e0,atol=1e-9,rtol=0)
        assert incomplete0==incomplete1
        assert len(rows0)==len(rows1)
        for a,b in zip(rows0,rows1):
            assert a["labels"]==b["labels"]
            np.testing.assert_allclose(a["energies"],b["energies"],atol=1e-8,rtol=0)
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(psis1.conj().T@psis1)),
                                   np.ones(psis1.shape[1]),atol=1e-8)
        assert res1<2e-6
    assert fast.last_tr_reused>0


def test_tr_partner_is_involution_for_odd_signed_rotation():
    g=RubyPBC("2x2",t=0.4,t1=0.2,t2=-0.2)
    orders=tuple(x[2] for x in generators_trans(g))
    labels=list(sector_labels(g,"point",use_signed=True,nf=3))
    keys={_symmetry_label_key(lab) for lab in labels}
    for lab in labels:
        mate=_time_reversed_block_key(lab,orders,3)
        assert mate in keys
        K,R,mirror,signed=mate
        reflected={"K":list(K),"rotation":None if R is None else list(R),
                   "mirror":mirror,"signed_rotation":signed}
        assert _time_reversed_block_key(reflected,orders,3)==_symmetry_label_key(lab)
