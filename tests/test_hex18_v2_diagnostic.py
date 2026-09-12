import json
import warnings
import numpy as np

from rubycgw.hex18_ed import Hex18Solver, canonical_ring_mode
from rubycgw.model import RubyParameters


def test_hex18_v2_hscan_diagnostic():
    solver = Hex18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0), primitive_filling=2.0)
    spec = solver.solve(2.0, n_eigs=8, tol=1e-10, maxiter=5000)
    u0 = canonical_ring_mode(0).real
    rows = []
    for h in (2e-3, 1e-3, 5e-4, 2e-4, 1e-4):
        r = solver.source_response(2.0, u0, h=h, tol=1e-10, maxiter=5000,
                                   v0=spec.eigenvectors[:, 0])
        rows.append({"h": float(h), "chi": float(r["chi"]), "Mplus": float(r["M_plus"])})
    warnings.warn("HEX18_V2_HSCAN_RESULT " + json.dumps({
        "gap_full": float(spec.gap_above_manifold),
        "rows": rows,
    }, sort_keys=True))
    assert all(np.isfinite(x["chi"]) for x in rows)
