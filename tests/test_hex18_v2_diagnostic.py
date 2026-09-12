import json
import warnings
import numpy as np

from rubycgw.hex18_ed import Hex18Solver, canonical_ring_mode
from rubycgw.model import RubyParameters


def test_hex18_v2_diagnostic():
    solver = Hex18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0), primitive_filling=2.0)
    spec = solver.solve(2.0, n_eigs=8, tol=1e-9, maxiter=5000)
    cur = solver.current_structure(spec)
    Sell = [
        float(np.vdot(canonical_ring_mode(l), cur.matrix @ canonical_ring_mode(l)).real)
        for l in range(6)
    ]
    u0 = canonical_ring_mode(0).real
    u3 = canonical_ring_mode(3).real
    r0 = solver.source_response(2.0, u0, h=1e-3, tol=1e-9, maxiter=5000,
                                v0=spec.eigenvectors[:, 0])
    r3 = solver.source_response(2.0, u3, h=1e-3, tol=1e-9, maxiter=5000,
                                v0=spec.eigenvectors[:, 0])
    principal = np.asarray(cur.eigenvectors[:, 0], dtype=complex)
    imax = int(np.argmax(np.abs(principal)))
    principal *= np.exp(-1j * np.angle(principal[imax]))
    payload = {
        "E": [float(x) for x in spec.energies[:8]],
        "gap": float(spec.gap_above_manifold),
        "gs_mult": int(spec.ground_multiplicity),
        "S_eigs": [float(x) for x in cur.eigenvalues],
        "S_ell": Sell,
        "principal_real": [float(x) for x in principal.real],
        "principal_imag_max": float(np.max(np.abs(principal.imag))),
        "circulant_error": float(cur.circulant_error),
        "mean_J_max": float(np.max(np.abs(cur.means))),
        "chi_uniform": float(r0["chi"]),
        "M_uniform_plus": float(r0["M_plus"]),
        "chi_alternating": float(r3["chi"]),
        "M_alternating_plus": float(r3["M_plus"]),
        "J_uniform_plus": [float(x) for x in r0["J_plus"]],
        "J_alternating_plus": [float(x) for x in r3["J_plus"]],
    }
    warnings.warn("HEX18_V2_RESULT " + json.dumps(payload, sort_keys=True))
    assert np.isfinite(cur.eigenvalues[0])
    assert np.isfinite(r0["chi"])
    assert np.isfinite(r3["chi"])
