import numpy as np

import geometry


def test_ring3_1x1_momenta_convention():
    g=geometry.make_ring3_supercell(1,1)
    qs=np.asarray(g.permitted_momenta(),float)
    got={tuple(np.round(q,10)) for q in qs}
    want={(0.0,0.0),(1/3,1/3),(2/3,2/3)}
    want={tuple(np.round(q,10)) for q in want}
    assert got==want


def test_rectangular_ring3_permitted_momenta():
    for L1,L2 in [(2,3),(3,4),(4,5),(5,4)]:
        g=geometry.make_ring3_supercell(L1,L2)
        qs=np.asarray(g.permitted_momenta(),float)
        assert len(qs)==3*L1*L2
        assert np.all(qs>=0.0)
        assert np.all(qs<1.0)

        # Every allowed q obeys the finite-torus boundary condition M^T q in Z^2.
        mtq=qs @ np.asarray(g.M,float)
        assert np.allclose(mtq,np.rint(mtq),atol=1e-9,rtol=0.0)

        # No duplicate reciprocal classes remain after canonicalisation.
        keys={tuple(np.round(q,10)) for q in qs}
        assert len(keys)==len(qs)
