import json
import warnings
import numpy as np

import analyze_hex18_low_energy_modes as diag
from rubycgw.hex18_ed import Hex18Solver
from rubycgw.model import RubyParameters


def test_hex18_low_energy_mode_diagnostic():
    solver = Hex18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0), primitive_filling=2.0)
    families = diag._manybody_local_families(solver)
    comps = ["x", "y", "z", "rho"]
    payload = []
    for V in (1.0, 2.0, 3.0, 4.0):
        spec = solver.solve(V, n_eigs=12, tol=1e-9, maxiter=8000, v0=None)
        E = np.asarray(spec.energies, dtype=float)
        vecs = np.asarray(spec.eigenvectors, dtype=complex)
        amp, w = diag._fingerprint(vecs, families)
        flat = w[1].reshape(-1)
        ib = int(np.argmax(flat))
        ia, ell = np.unravel_index(ib, w[1].shape)
        top = []
        for n in range(1, 6):
            ibn = int(np.argmax(w[n].reshape(-1)))
            ian, elln = np.unravel_index(ibn, w[n].shape)
            top.append({
                "n": n,
                "gap": float(E[n]-E[0]),
                "best_comp": comps[ian],
                "best_ell": int(elln),
                "best_weight": float(w[n,ian,elln]),
                "w_z0": float(w[n,2,0]),
            })
        proj = []
        for M in (2,4,6,8):
            jvals, _, rows = diag._projected_chiral_states(E, vecs, solver.current_ops, M)
            rmin, rmax = rows
            proj.append({
                "M": M,
                "Jmin": float(rmin[0]), "Jmax": float(rmax[0]),
                "Emin_gap": float(rmin[1]-E[0]), "Emax_gap": float(rmax[1]-E[0]),
                "sigmaE_min": float(rmin[2]), "sigmaE_max": float(rmax[2]),
                "localJ_min": [float(x) for x in rmin[3]],
                "localJ_max": [float(x) for x in rmax[3]],
                "prob_min": [float(x) for x in rmin[4]],
                "prob_max": [float(x) for x in rmax[4]],
            })
        payload.append({
            "V": V,
            "E0": float(E[0]),
            "gap1": float(E[1]-E[0]),
            "n1_best_comp": comps[ia],
            "n1_best_ell": int(ell),
            "n1_best_weight": float(w[1,ia,ell]),
            "states": top,
            "projected": proj,
        })
    warnings.warn("HEX18_LOW_MODE_RESULT " + json.dumps(payload, sort_keys=True))
    assert all(np.isfinite(row["gap1"]) for row in payload)
