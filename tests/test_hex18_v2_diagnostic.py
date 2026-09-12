import json
import warnings
import numpy as np

from rubycgw.hex18_ed import Hex18Solver, canonical_ring_mode
from rubycgw.model import RubyParameters


def test_hex18_v_scan_diagnostic():
    solver = Hex18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0), primitive_filling=2.0)
    u0 = canonical_ring_mode(0).real
    payload = []
    v0 = None
    for V in (0.0, 0.5, 1.0, 1.5, 2.0):
        spec = solver.solve(V, n_eigs=5, tol=1e-9, maxiter=5000, v0=v0)
        v0 = spec.eigenvectors[:, 0]
        cur = solver.current_structure(spec)
        Sell = [
            float(np.vdot(canonical_ring_mode(l), cur.matrix @ canonical_ring_mode(l)).real)
            for l in range(6)
        ]
        r0 = solver.source_response(V, u0, h=1e-3, tol=1e-9, maxiter=5000, v0=v0)
        principal = np.asarray(cur.eigenvectors[:, 0], dtype=complex)
        imax = int(np.argmax(np.abs(principal)))
        principal *= np.exp(-1j * np.angle(principal[imax]))
        payload.append({
            "V": float(V),
            "E0": float(spec.energies[0]),
            "gap": float(spec.gap_above_manifold),
            "Smax": float(cur.eigenvalues[0]),
            "S_ell": Sell,
            "principal_real": [float(x) for x in principal.real],
            "principal_imag_max": float(np.max(np.abs(principal.imag))),
            "circulant_error": float(cur.circulant_error),
            "mean_J_max": float(np.max(np.abs(cur.means))),
            "chi_uniform": float(r0["chi"]),
            "M_uniform_plus": float(r0["M_plus"]),
        })
    warnings.warn("HEX18_VSCAN_RESULT " + json.dumps(payload, sort_keys=True))
    assert all(np.isfinite(row["chi_uniform"]) for row in payload)
