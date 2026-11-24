"""Common exact observables for the Hex18 ED/VMC benchmark.

The same routines are applied to (i) exact ED ground-state vectors and
(ii) the fully enumerated optimized VMC wavefunction. Therefore differences
in the final benchmark are not Monte-Carlo measurement noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from order_diagnostics import order_diagnostics
from itertools import product
import math
import numpy as np


@dataclass
class ObservableResult:
    charge_qmax: float
    charge_q1: float
    charge_q2: float
    charge_qmax_connected: float
    bow_star_mean: float
    bow_star_S: float
    bow_star_S_connected: float
    lc_same_mean: float
    lc_opposite_mean: float
    lc_same_S: float
    lc_opposite_S: float
    lc_same_S_connected: float
    lc_opposite_S_connected: float
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "CO_qmax": self.charge_qmax,
            "CO_q1": self.charge_q1,
            "CO_q2": self.charge_q2,
            "CO_qmax_connected": self.charge_qmax_connected,
            "B_star_mean": self.bow_star_mean,
            "S_BOW_star": self.bow_star_S,
            "S_BOW_star_connected": self.bow_star_S_connected,
            "J_same_mean": self.lc_same_mean,
            "J_opposite_mean": self.lc_opposite_mean,
            "S_LC_same": self.lc_same_S,
            "S_LC_opposite": self.lc_opposite_S,
            "S_LC_same_connected": self.lc_same_S_connected,
            "S_LC_opposite_connected": self.lc_opposite_S_connected,
            **self.diagnostics,
        }


def occupations_from_native_states(states: np.ndarray, ns: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint64)
    shifts = np.arange(ns - 1, -1, -1, dtype=np.uint64)
    return ((states[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float64)


def charge_momenta(conn: np.ndarray, reps, momenta) -> list[dict]:
    nc = len(reps)
    ns = 6 * nc
    conn = np.asarray(conn, complex)
    if conn.shape != (ns, ns):
        raise ValueError("Wrong charge covariance shape")
    ret = []
    for k in momenta:
        F = np.zeros((ns, 6), dtype=complex)
        for cell, R in enumerate(reps):
            phi = 2 * np.pi * float(np.dot(k, R))
            for alpha in range(6):
                F[6 * cell + alpha, alpha] = np.exp(1j * phi) / np.sqrt(nc)
        block = F.conj().T @ conn @ F
        block = (block + block.conj().T) / 2
        eig = np.linalg.eigvalsh(block)
        ret.append({"k": tuple(float(x) for x in k), "value": float(eig[-1].real)})
    return ret


def ruby_star_formfactor(g, center=None) -> np.ndarray:
    """Ruby SD/ISD-like real-bond form factor used by the existing HF code."""
    if center is None:
        center = np.array([0.0, -2.0 - math.sqrt(3.0) / 2.0])
    center = np.asarray(center, float)
    f = np.empty(len(g.edges), dtype=float)
    for ib, e in enumerate(g.edges):
        p0 = np.asarray(g.positions[e.i], float)
        p1 = np.asarray(g.positions[e.j], float) + np.asarray(g.B, float) @ np.asarray(e.winding, int)
        mid = 0.5 * (p0 + p1)
        dist = min(
            np.linalg.norm(mid + np.asarray(g.B) @ np.asarray(w, int) - center)
            for w in product(range(-2, 3), repeat=2)
        )
        star = (e.kind == "t" and dist < 1.7) or (e.kind != "t" and dist < 1.05)
        f[ib] = 1.0 if star else -1.0
    f -= f.mean()
    scale = np.max(np.abs(f))
    if scale < 1e-14:
        raise RuntimeError("Degenerate Ruby star form factor")
    return f / scale


def _bow_terms(g, f: np.ndarray):
    if len(f) != len(g.edges):
        raise ValueError("Bond form-factor length mismatch")
    terms = []
    norm = float(len(g.edges))
    for coeff, e in zip(f, g.edges):
        a = float(coeff) / norm
        terms.append((a, [("+", e.i), ("-", e.j)]))
        terms.append((a, [("+", e.j), ("-", e.i)]))
    return terms


def _normalized_columns(psi_vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(psi_vectors, complex)
    if x.ndim == 1:
        x = x[:, None]
    norms = np.linalg.norm(x, axis=0)
    if np.any(norms < 1e-14):
        raise ValueError("Zero-norm state supplied")
    return x / norms[None, :]


def benchmark_observables(g, states, psi_vectors, native_observables_module, weights=None, *, ground_manifold=False) -> ObservableResult:
    """Compute common charge, BOW, and LC diagnostics.

    Several ED ground vectors are interpreted as an incoherent ensemble,
    which makes the result basis-independent inside an exactly degenerate
    finite-size ground manifold.
    """
    states = np.asarray(states, dtype=np.uint64)
    psi = _normalized_columns(psi_vectors)
    ng = psi.shape[1]
    if weights is None:
        weights = np.full(ng, 1.0 / ng)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()

    if ground_manifold:
        if not np.allclose(psi.conj().T@psi,np.eye(ng),atol=1e-8) or not np.allclose(weights,1/ng):
            raise ValueError("Ground diagnostics require an orthonormal, equally weighted complete manifold")
    ns = g.Ns
    occ = occupations_from_native_states(states, ns)
    density = np.zeros(ns, float)
    nn = np.zeros((ns, ns), float)

    loops = g.cycles()
    tri = [loop for loop in loops if len(loop) == 3]
    if len(tri) != 2 * len(g.reps):
        raise AssertionError("Unexpected number of elementary triangles")
    ntri = len(tri)
    J_mean = np.zeros(ntri, complex)
    J_second = np.zeros((ntri, ntri), complex)
    current_projected = np.zeros((ntri,ng,ng),complex) if ground_manifold else None

    fstar = ruby_star_formfactor(g)
    bow_terms = _bow_terms(g, fstar)
    bow_mean = 0.0 + 0.0j
    bow_second = 0.0

    for a in range(ng):
        v = psi[:, a]
        w = float(weights[a])
        prob = np.abs(v) ** 2
        d = prob @ occ
        raw_nn = occ.T @ (prob[:, None] * occ)
        density += w * d
        nn += w * raw_nn

        jvec = []
        jm = []
        for loop in tri:
            x = native_observables_module.op_sum(
                v, states, ns, native_observables_module.current_terms(g, loop)
            )
            jvec.append(x)
            jm.append(np.vdot(v, x))
        jvec = np.asarray(jvec)
        jm = np.asarray(jm)
        rawJ = jvec.conj() @ jvec.T
        J_mean += w * jm
        J_second += w * rawJ
        if ground_manifold:
            current_projected[:,:,a] = (psi.conj().T@jvec.T).T

        bx = native_observables_module.op_sum(v, states, ns, bow_terms)
        bm = np.vdot(v, bx)
        bow_mean += w * bm
        bow_second += w * float(np.vdot(bx, bx).real)

    conn_n = nn - np.outer(density, density)
    nbar = float(np.mean(density))
    raw_delta_n = nn - nbar * (density[:, None] + density[None, :]) + nbar**2
    qdata = charge_momenta(raw_delta_n, g.reps, g.permitted_momenta())
    qmax = max(qdata, key=lambda q: q["value"])
    qdata_conn = charge_momenta(conn_n, g.reps, g.permitted_momenta())
    qmax_conn = max(qdata_conn, key=lambda q: q["value"])

    connJ = J_second - np.outer(J_mean.conj(), J_mean)
    connJ = 0.5 * (connJ + connJ.conj().T)
    ab_sign = np.array([1.0 if loop[0] % 6 < 3 else -1.0 for loop in tri])
    u_same = np.ones(ntri, complex) / ntri
    u_opp = ab_sign.astype(complex) / ntri

    same_mean = np.vdot(u_same, J_mean)
    opp_mean = np.vdot(u_opp, J_mean)
    same_S_raw = float(np.vdot(u_same, J_second @ u_same).real)
    opp_S_raw = float(np.vdot(u_opp, J_second @ u_opp).real)
    same_S = float(np.vdot(u_same, connJ @ u_same).real)
    opp_S = float(np.vdot(u_opp, connJ @ u_opp).real)

    bow_raw = float(bow_second)
    bow_conn = float(bow_second - abs(bow_mean) ** 2)
    if bow_conn < 0 and abs(bow_conn) < 1e-12:
        bow_conn = 0.0
    if same_S < 0 and abs(same_S) < 1e-12:
        same_S = 0.0
    if opp_S < 0 and abs(opp_S) < 1e-12:
        opp_S = 0.0

    diagnostics=order_diagnostics(g,occ,psi,weights,raw_delta_n,J_second,tri,
                                  current_projected,ground_manifold=ground_manifold)
    return ObservableResult(
        charge_qmax=float(qmax["value"]),
        charge_q1=float(qmax["k"][0]),
        charge_q2=float(qmax["k"][1]),
        charge_qmax_connected=float(qmax_conn["value"]),
        bow_star_mean=float(np.real_if_close(bow_mean).real),
        bow_star_S=bow_raw,
        bow_star_S_connected=bow_conn,
        lc_same_mean=float(np.real_if_close(same_mean).real),
        lc_opposite_mean=float(np.real_if_close(opp_mean).real),
        lc_same_S=same_S_raw,
        lc_opposite_S=opp_S_raw,
        lc_same_S_connected=same_S,
        lc_opposite_S_connected=opp_S,
        diagnostics=diagnostics,
    )
