"""Jacobian-free response of the three-orientation symmetrized embedding.

This module linearizes the *unprojected* symmetrized functional around a
C3/time-reversal-constrained parent checkpoint,

    Phi_sym = Phi_GW^lat
            + (1/3) sum_r [Phi_ED^(r) - Phi_GW,C^(r)].

The nonlinear C3/TR projectors used to hold the parent saddle are deliberately
NOT differentiated here.  Consequently C3-breaking charge-order and
time-reversal-breaking loop-current fluctuations remain in the response space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gcrotmk, gmres

from . import cluster_ed_gw_jf as jf
from . import cluster_ed_gw_jf_consistent as consistent
from .cluster_ed_gw import BathParameters, cluster_interaction_matrix
from .cluster_orientation import (
    build_oriented_lattice_fields,
    gauge_transform_lattice,
    gauge_transform_response_field,
    local_response_vertex_in_orientation,
    orientation_b_shift,
)
from .finite_q_cgw import normalize_q_index
from .grids import MatsubaraGrid
from .models.ruby import physical_pair_cluster_interactions
from .response_tail import build_tail_reference
from .supercell_gw import hartree_self_energy_matrix
from .supercell_gw_split import one_body_density_matrix_tail


@dataclass
class SymmetrizedJFBuild:
    operator: "SymmetrizedEmbeddedJacobian"
    tangents: tuple[jf.BathTangentModel, jf.BathTangentModel, jf.BathTangentModel]


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _build_one_orientation_operator(
    *,
    orientation: int,
    G: np.ndarray,
    h0: np.ndarray,
    Vq: np.ndarray,
    bath: BathParameters,
    h_impurity: np.ndarray,
    params,
    grid: MatsubaraGrid,
    mu: float,
    rho_cluster: np.ndarray,
    sigma_h: np.ndarray,
    bath_opts: jf.BathTangentOptions,
    jf_opts: jf.ClusterJFOptions,
):
    """Build one tail-consistent physical-pair operator without global leakage."""
    r = int(orientation)
    builder = lambda p: physical_pair_cluster_interactions(p, r)

    old_jf = jf.ruby_cluster_interactions
    old_consistent = consistent.ruby_cluster_interactions
    jf.ruby_cluster_interactions = builder
    consistent.ruby_cluster_interactions = builder
    try:
        op, tangent = jf.build_embedded_jacobian(
            G,
            Vq,
            bath,
            h_impurity,
            params,
            grid,
            mu,
            rho_cluster,
            bath_opts=bath_opts,
            jf_opts=jf_opts,
        )
    finally:
        jf.ruby_cluster_interactions = old_jf
        consistent.ruby_cluster_interactions = old_consistent

    # The generic compatibility layer historically reconstructs canonical h0
    # and a cluster Hartree reference.  A physical-pair orientation checkpoint
    # instead needs the actual oriented lattice h0 and the full-lattice Hartree
    # used by the nonlinear background.
    op._jf_tail_reference = build_tail_reference(
        np.asarray(h0, dtype=complex),
        float(mu),
        np.asarray(sigma_h, dtype=complex),
        grid,
    )
    op._jf_h_cluster = np.asarray(h_impurity, dtype=complex).copy()
    op._jf_interactions = builder(params)
    op.Vc = cluster_interaction_matrix(op._jf_interactions, 6)
    op._Vc4 = op.Vc[None, None]
    return op, tangent


class SymmetrizedEmbeddedJacobian:
    """Matrix-free JF operator for the equal-weight three-cut functional."""

    def __init__(self, ops):
        if len(ops) != 3:
            raise ValueError("exactly three orientation operators are required")
        self.ops = tuple(ops)
        self.G = np.asarray(self.ops[0].G, dtype=complex)
        self.grid = self.ops[0].grid
        self.opts = self.ops[0].opts
        for op in self.ops[1:]:
            if op.grid != self.grid:
                raise ValueError("orientation JF grids differ")
            if op.G.shape != self.G.shape:
                raise ValueError("orientation JF G shapes differ")

    def _to_orientation(self, field, r: int, p):
        return gauge_transform_response_field(
            np.asarray(field, dtype=complex),
            orientation_b_shift(r),
            p,
        )

    def _to_common(self, field, r: int, p):
        return gauge_transform_response_field(
            np.asarray(field, dtype=complex),
            -orientation_b_shift(r),
            p,
        )

    def _source_data(self, K: np.ndarray, p):
        data = []
        for r, op in enumerate(self.ops):
            Kr = local_response_vertex_in_orientation(
                K,
                r,
                p,
                nk1=self.grid.nk1,
                nk2=self.grid.nk2,
            )
            Kfield = np.broadcast_to(Kr, op.G.shape).copy()
            ctx = consistent._tail_context(op, Kr, p)
            direct = consistent._source_direct_sigma(op, Kr)

            zero_X = np.zeros_like(op.G)
            static_const, _ = consistent._static_net_field(op, zero_X, ctx)
            zero_gc = np.zeros_like(op.Gc)
            imp_const = consistent._impurity_vertex(
                op, zero_gc, Kr, direct
            )
            rhs_r = (
                Kfield
                + static_const
                + imp_const[:, None, None, :, :]
            )
            data.append(
                dict(
                    K=Kr,
                    ctx=ctx,
                    direct=direct,
                    static_const=static_const,
                    imp_const=imp_const,
                    rhs=rhs_r,
                )
            )
        return data

    def rhs(self, K: np.ndarray, q_index) -> tuple[np.ndarray, list[dict]]:
        p = normalize_q_index(q_index, self.grid)
        data = self._source_data(np.asarray(K, dtype=complex), p)
        out = np.zeros_like(self.G)
        for r, d in enumerate(data):
            out += self._to_common(d["rhs"], r, p)
        return out / 3.0, data

    def linear_kernel(self, Gamma: np.ndarray, q_index, source_data) -> np.ndarray:
        p = normalize_q_index(q_index, self.grid)
        common = np.asarray(Gamma, dtype=complex)
        out = np.zeros_like(common)

        for r, (op, d) in enumerate(zip(self.ops, source_data)):
            field = self._to_orientation(common, r, p)
            X = op._x_field(field, p)

            static_total, _ = consistent._static_net_field(
                op, X, d["ctx"]
            )
            static_linear = static_total - d["static_const"]
            lat_dyn = consistent._dynamic_lattice(op, field, p)

            delta_gc = np.mean(X, axis=(1, 2))
            imp_total = consistent._impurity_vertex(
                op, delta_gc, d["K"], d["direct"]
            )
            imp_linear = imp_total - d["imp_const"]
            c_dyn = consistent._cluster_dynamic_tangent(op, delta_gc)

            kernel_r = (
                static_linear
                + lat_dyn
                + (imp_linear - c_dyn)[:, None, None, :, :]
            )
            out += self._to_common(kernel_r, r, p)
        return out / 3.0

    def solve(
        self,
        K: np.ndarray,
        q_index,
        *,
        initial_gamma=None,
        recycle=None,
    ) -> jf.ClusterJFResult:
        p = normalize_q_index(q_index, self.grid)
        K = np.asarray(K, dtype=complex)
        if K.shape != (6, 6):
            raise ValueError("bare vertex must be 6x6")

        rhs, source_data = self.rhs(K, p)
        gamma0 = (
            rhs
            if initial_gamma is None
            or np.asarray(initial_gamma).shape != self.G.shape
            else np.asarray(initial_gamma, dtype=complex)
        )
        shape = self.G.shape
        n = 2 * int(np.prod(shape))

        def matvec(x):
            gamma = jf._unpack_complex(x, shape)
            y = gamma - self.linear_kernel(gamma, p, source_data)
            return jf._pack_complex(y)

        A = LinearOperator((n, n), matvec=matvec, dtype=float)
        b = jf._pack_complex(rhs)
        x0 = jf._pack_complex(gamma0)
        rtol = jf._krylov_relative_tol_for_maxabs(b, float(self.opts.tol))
        count = [0]

        def callback(_):
            count[0] += 1

        solver = str(self.opts.solver).lower()
        if solver == "gcrotmk":
            CU = recycle if recycle is not None else []
            kwargs = dict(
                x0=x0,
                atol=0.0,
                maxiter=int(self.opts.maxiter),
                m=int(self.opts.restart),
                k=int(self.opts.recycle_dim),
                CU=CU,
                callback=callback,
            )
            try:
                x, info = gcrotmk(A, b, rtol=rtol, **kwargs)
            except TypeError:
                x, info = gcrotmk(A, b, tol=rtol, **kwargs)
        elif solver == "gmres":
            kwargs = dict(
                x0=x0,
                atol=0.0,
                maxiter=int(self.opts.maxiter),
                restart=int(self.opts.restart),
                callback=callback,
            )
            try:
                x, info = gmres(
                    A, b, rtol=rtol, callback_type="pr_norm", **kwargs
                )
            except TypeError:
                x, info = gmres(A, b, tol=rtol, **kwargs)
        else:
            raise ValueError("solver must be 'gcrotmk' or 'gmres'")

        Gamma = jf._unpack_complex(x, shape)
        residual = (
            rhs
            + self.linear_kernel(Gamma, p, source_data)
            - Gamma
        )
        err = _maxabs(residual)
        converged = bool(
            int(info) == 0
            and np.isfinite(err)
            and err <= float(self.opts.tol)
        )
        if self.opts.verbose:
            print(
                f"[cluster-JF:sym] q={p}, solver={solver}, "
                f"it={count[0]}, info={info}, residual={err:.3e}",
                flush=True,
            )
        return jf.ClusterJFResult(
            Gamma=np.asarray(Gamma),
            converged=converged,
            iterations=int(count[0]),
            final_error=float(err),
            solver_info=int(info),
            q_index=p,
        )


def build_symmetrized_embedded_jacobian(
    G: np.ndarray,
    h0: np.ndarray,
    Vq: np.ndarray,
    baths,
    static_shifts: np.ndarray,
    params,
    grid: MatsubaraGrid,
    mu: float,
    sigma_h: np.ndarray,
    *,
    bath_opts: jf.BathTangentOptions = jf.BathTangentOptions(),
    jf_opts: jf.ClusterJFOptions = jf.ClusterJFOptions(),
) -> SymmetrizedJFBuild:
    """Build the three orientation tangents around one symmetric parent G."""
    if len(baths) != 3:
        raise ValueError("three saved baths are required")
    shifts = np.asarray(static_shifts, dtype=complex)
    if shifts.shape != (3, 6, 6):
        raise ValueError("static_shifts must have shape (3,6,6)")

    G = np.asarray(G, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    rho_k = one_body_density_matrix_tail(
        G, grid, h0, float(mu), np.asarray(sigma_h, dtype=complex)
    )

    ops = []
    tangents = []
    for r in range(3):
        hr, Vr = build_oriented_lattice_fields(h0, Vq, r)
        Gr = gauge_transform_lattice(G, orientation_b_shift(r))
        rr = gauge_transform_lattice(rho_k, orientation_b_shift(r))
        rho_cr = np.mean(rr, axis=(0, 1))
        hc = np.mean(hr, axis=(0, 1))
        hc = 0.5 * (hc + hc.conj().T)
        h_imp = hc + shifts[r]

        op, tangent = _build_one_orientation_operator(
            orientation=r,
            G=Gr,
            h0=hr,
            Vq=Vr,
            bath=baths[r],
            h_impurity=h_imp,
            params=params,
            grid=grid,
            mu=float(mu),
            rho_cluster=rho_cr,
            sigma_h=np.asarray(sigma_h, dtype=complex),
            bath_opts=bath_opts,
            jf_opts=jf_opts,
        )
        ops.append(op)
        tangents.append(tangent)

    return SymmetrizedJFBuild(
        operator=SymmetrizedEmbeddedJacobian(ops),
        tangents=tuple(tangents),
    )


def response_matrix_symmetrized(
    operator: SymmetrizedEmbeddedJacobian,
    vertices: np.ndarray,
    q_index,
    *,
    initial_gammas=None,
    recycle: bool = True,
):
    """Solve all source channels and evaluate the physical lattice response."""
    K = np.asarray(vertices, dtype=complex)
    if K.ndim != 3 or K.shape[1:] != (6, 6):
        raise ValueError("vertices must have shape (nchannel,6,6)")
    if initial_gammas is None:
        initial_gammas = [None] * len(K)
    if len(initial_gammas) != len(K):
        raise ValueError("initial_gammas length mismatch")

    CU = [] if recycle and str(operator.opts.solver).lower() == "gcrotmk" else None
    results = []
    for j, vertex in enumerate(K):
        result = operator.solve(
            vertex,
            q_index,
            initial_gamma=initial_gammas[j],
            recycle=CU,
        )
        if not result.converged:
            raise RuntimeError(
                f"symmetrized JF channel {j} did not converge: "
                f"residual={result.final_error:.3e}"
            )
        results.append(result)

    # Observable response is a lattice one-body quantity.  Evaluate it in the
    # common ori0 gauge using the same tail-completed density map.
    op0 = operator.ops[0]
    p = normalize_q_index(q_index, operator.grid)
    chi = np.zeros((len(K), len(K)), dtype=complex)
    for b, result in enumerate(results):
        ctx = consistent._tail_context(op0, K[b], p)
        X = op0._x_field(result.Gamma, p)
        _, _, Rk = ctx.total_static_parts(X)
        chi[:, b] = -(1.0 / float(operator.grid.nk)) * np.einsum(
            "aij,xyji->a", K, Rk, optimize=True
        )
    return chi, results


__all__ = [
    "SymmetrizedJFBuild",
    "SymmetrizedEmbeddedJacobian",
    "build_symmetrized_embedded_jacobian",
    "response_matrix_symmetrized",
]
