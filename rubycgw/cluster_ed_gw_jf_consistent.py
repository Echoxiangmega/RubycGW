"""Numerical-consistency layer for the cluster-ED+GW JF response.

The production cluster-ED+GW map evaluates the equal-time density matrix with
an analytic static-reference tail subtraction.  The original JF prototype used
finite-box H/F contractions and a finite-box susceptibility observable.  It
also linearized only bath-parameter changes of the impurity map, while a source
``h0 -> h0-hK`` changes the correlated-cluster one-body block explicitly.

This module installs three consistency corrections without changing the public
JF API:

1. H/F response uses :mod:`rubycgw.response_tail`, i.e. the derivative of the
   same tail-subtracted density map used by the nonlinear embedding.
2. The cluster-GW static double counting is differentiated from that same
   tail-completed local density response; for the local Ruby interaction this
   cancels the lattice static H/F pieces exactly up to roundoff, as it should.
3. The finite-bath impurity tangent includes the explicit local source
   derivative at fixed bath, in addition to the low-rank bath-fit tangent.

The installation is idempotent and is invoked from ``rubycgw.__init__`` so all
standalone drivers see the corrected implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from . import cluster_ed_gw_jf as jf
from .cluster_ed_gw import BathParameters, cluster_interaction_matrix, ruby_cluster_interactions
from .finite_q_cgw import normalize_q_index, vertex_corrections_finite_q
from .model import NSUB, build_h0
from .response_tail import build_tail_hf_context, build_tail_reference
from .supercell_cgw import SupercellVertexOptions, vertex_corrections_q0
from .supercell_gw import hartree_self_energy_matrix


def _key_matrix(a: np.ndarray) -> bytes:
    z = np.ascontiguousarray(np.asarray(a, dtype=np.complex128))
    return z.tobytes()


def _source_direct_sigma(operator, source: np.ndarray) -> np.ndarray:
    """Return the explicit impurity vertex ``-dSigma/dh`` at fixed bath.

    The source convention is ``h_cluster -> h_cluster - h K``.  Therefore the
    desired vertex contribution equals the derivative of Sigma with respect to
    ``h_cluster`` along ``+K``.  A centered finite difference is used once per
    independent source and cached across all q points.
    """
    cache = operator._jf_source_sigma_cache
    key = _key_matrix(source)
    if key in cache:
        return cache[key]

    step = float(operator._jf_source_fd_step)
    K = np.asarray(source, dtype=complex)
    h = np.asarray(operator._jf_h_cluster, dtype=complex)
    bath = operator._jf_bath
    interactions = operator._jf_interactions
    grid = operator.grid
    mu = float(operator._jf_mu)
    discard = float(operator._jf_discard_weight_tol)

    sig_plus = jf._impurity_sigma_for_bath(
        bath, h + step * K, interactions, grid, mu, discard
    )
    sig_minus = jf._impurity_sigma_for_bath(
        bath, h - step * K, interactions, grid, mu, discard
    )
    direct = (sig_plus - sig_minus) / (2.0 * step)
    cache[key] = np.asarray(direct, dtype=complex)
    if operator.opts.verbose:
        print(
            f"[JF impurity] explicit source tangent cached: step={step:.2e}, "
            f"max|dSigma/dh_local|={jf._maxabs(direct):.3e}",
            flush=True,
        )
    return cache[key]


def _impurity_vertex(operator, delta_gc, source, direct_sigma):
    """Source-aware low-rank impurity vertex in the JF sign convention."""
    tangent = operator.bath_tangent
    dgc = np.asarray(delta_gc, dtype=complex)
    # X=-dG/dh, so -d(Gc^{-1})/dh = -Gc^{-1} X Gc^{-1}.
    drive = -np.einsum(
        "nab,nbc,ncd->nad",
        tangent.Gc_inv,
        dgc,
        tangent.Gc_inv,
        optimize=True,
    )
    # For h_cluster -> h_cluster-hK, the bath-fit tangent is driven by
    # A-K-s_direct.  coefficients_from_dg0_inv applies an extra minus sign to
    # its argument, hence add K+s_direct to the stored -A field here.
    drive = (
        drive
        + np.asarray(source, dtype=complex)[None, :, :]
        + np.asarray(direct_sigma, dtype=complex)
    )
    rhs = tangent.coefficients_from_dg0_inv(drive)
    coeff = np.linalg.solve(tangent.inner_matrix, rhs)
    bath_part = np.einsum(
        "r,rnab->nab", coeff, tangent.sigma_modes, optimize=True
    )
    return np.asarray(direct_sigma, dtype=complex) + bath_part


def _cluster_dynamic_tangent(operator, delta_gc: np.ndarray) -> np.ndarray:
    """Cluster-GW double-counting tangent excluding static H/F pieces."""
    invg = np.linalg.inv(operator.Gc)
    gamma_c = np.einsum(
        "nab,nbc,ncd->nad", invg, delta_gc, invg, optimize=True
    )
    parts = vertex_corrections_q0(
        operator._Gc5,
        operator._Wc5,
        operator._Vc4,
        gamma_c[:, None, None],
        operator.cluster_grid,
        operator._jf_dyn_cl_opts,
    )
    return np.asarray(parts[2] + parts[3] + parts[4], dtype=complex)[:, 0, 0]


def _tail_context(operator, source: np.ndarray, q_index):
    p = normalize_q_index(q_index, operator.grid)
    key = (p, _key_matrix(source))
    cache = operator._jf_tail_context_cache
    if key not in cache:
        cache[key] = build_tail_hf_context(
            operator._jf_tail_reference,
            np.asarray(source, dtype=complex),
            operator.Vq,
            operator.grid,
            q_index=p,
            m_ext=0,
            backend=str(operator.opts.momentum_backend),
            include_hartree=bool(operator.opts.include_hartree),
            include_fock=bool(operator.opts.include_fock),
        )
    return cache[key]


def _static_net_field(operator, X: np.ndarray, ctx) -> tuple[np.ndarray, np.ndarray]:
    """Return embedded static H/F vertex and tail-completed density tangent."""
    Hlat, Flat, Rk = ctx.total_static_parts(X)
    Rc = np.mean(np.asarray(Rk, dtype=complex), axis=(0, 1))

    # cluster_gw_self_energy uses the same rho_c for its static Hartree/Fock
    # double counting.  Differentiate that exact map instead of forming a
    # finite-box cluster bubble.
    Hc = np.zeros((NSUB, NSUB), dtype=complex)
    Hc[np.diag_indices(NSUB)] = operator.Vc @ np.diag(Rc)
    Fc = -Rc * operator.Vc.T

    static_k = (
        (np.asarray(Hlat, dtype=complex) - Hc)[None, None, :, :]
        + np.asarray(Flat, dtype=complex)
        - Fc[None, None, :, :]
    )
    field = np.broadcast_to(static_k[None, ...], operator.G.shape).copy()
    return field, np.asarray(Rk, dtype=complex)


def _dynamic_lattice(operator, Gamma: np.ndarray, p) -> np.ndarray:
    parts = vertex_corrections_finite_q(
        operator.G,
        operator.W,
        operator.Vq,
        Gamma,
        p,
        operator.grid,
        operator._jf_dyn_lat_opts,
    )
    return np.asarray(parts[2] + parts[3] + parts[4], dtype=complex)


def _solve_consistent(operator, K, q_index, *, initial_gamma=None, recycle=None):
    p = normalize_q_index(q_index, operator.grid)
    K = np.asarray(K, dtype=complex)
    if K.shape != (NSUB, NSUB):
        raise ValueError("bare vertex must be a 6x6 primitive-cell matrix")

    Kfield = np.broadcast_to(K, operator.G.shape).copy()
    gamma0 = (
        Kfield
        if initial_gamma is None or np.asarray(initial_gamma).shape != operator.G.shape
        else np.asarray(initial_gamma, dtype=complex)
    )
    ctx = _tail_context(operator, K, p)
    direct = _source_direct_sigma(operator, K)

    zero_X = np.zeros_like(operator.G)
    static_const, _ = _static_net_field(operator, zero_X, ctx)
    zero_gc = np.zeros_like(operator.Gc)
    imp_const = _impurity_vertex(operator, zero_gc, K, direct)
    rhs_field = Kfield + static_const + imp_const[:, None, None, :, :]

    def linear_kernel(field):
        field = np.asarray(field, dtype=complex)
        X = operator._x_field(field, p)
        static_total, _ = _static_net_field(operator, X, ctx)
        static_linear = static_total - static_const
        lat_dyn = _dynamic_lattice(operator, field, p)
        delta_gc = np.mean(X, axis=(1, 2))
        imp_total = _impurity_vertex(operator, delta_gc, K, direct)
        imp_linear = imp_total - imp_const
        c_dyn = _cluster_dynamic_tangent(operator, delta_gc)
        return (
            static_linear
            + lat_dyn
            + (imp_linear - c_dyn)[:, None, None, :, :]
        )

    shape = operator.G.shape
    n = 2 * int(np.prod(shape))

    def matvec(x):
        gamma = jf._unpack_complex(x, shape)
        return jf._pack_complex(gamma - linear_kernel(gamma))

    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    b = jf._pack_complex(rhs_field)
    x0 = jf._pack_complex(gamma0)
    krylov_rtol = jf._krylov_relative_tol_for_maxabs(b, float(operator.opts.tol))
    count = [0]

    def callback(_):
        count[0] += 1

    solver = str(operator.opts.solver).lower()
    if solver == "gcrotmk":
        CU = recycle if recycle is not None else []
        kwargs = dict(
            x0=x0,
            atol=0.0,
            maxiter=int(operator.opts.maxiter),
            m=int(operator.opts.restart),
            k=int(operator.opts.recycle_dim),
            CU=CU,
            callback=callback,
        )
        try:
            x, info = jf.gcrotmk(A, b, rtol=krylov_rtol, **kwargs)
        except TypeError:
            x, info = jf.gcrotmk(A, b, tol=krylov_rtol, **kwargs)
    elif solver == "gmres":
        kwargs = dict(
            x0=x0,
            atol=0.0,
            maxiter=int(operator.opts.maxiter),
            restart=int(operator.opts.restart),
            callback=callback,
        )
        try:
            x, info = jf.gmres(
                A, b, rtol=krylov_rtol, callback_type="pr_norm", **kwargs
            )
        except TypeError:
            x, info = jf.gmres(A, b, tol=krylov_rtol, **kwargs)
    else:
        raise ValueError("solver must be 'gcrotmk' or 'gmres'")

    Gamma = jf._unpack_complex(x, shape)
    residual = rhs_field + linear_kernel(Gamma) - Gamma
    err = jf._maxabs(residual)
    converged = bool(
        int(info) == 0 and np.isfinite(err) and err <= float(operator.opts.tol)
    )
    if operator.opts.verbose:
        print(
            f"[cluster-JF] q={p}, solver={solver}, it={count[0]}, "
            f"info={info}, residual={err:.3e}, tail=on, impurity-source=on",
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


def _response_matrix_consistent(
    operator,
    vertices,
    q_index,
    *,
    initial_gammas=None,
    recycle=True,
):
    K = np.asarray(vertices, dtype=complex)
    if K.ndim != 3 or K.shape[1:] != (NSUB, NSUB):
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
        results.append(result)
        if not result.converged:
            raise RuntimeError(
                f"embedded JF channel {j} at q={q_index} did not converge: "
                f"residual={result.final_error:.3e}, info={result.solver_info}"
            )

    chi = np.zeros((len(K), len(K)), dtype=complex)
    p = normalize_q_index(q_index, operator.grid)
    for b, result in enumerate(results):
        ctx = _tail_context(operator, K[b], p)
        X = operator._x_field(result.Gamma, p)
        _, _, Rk = ctx.total_static_parts(X)
        chi[:, b] = -(1.0 / float(operator.grid.nk)) * np.einsum(
            "aij,xyji->a", K, Rk, optimize=True
        )
    return chi, results


def install_tail_consistent_cluster_jf() -> None:
    """Install the tail/source-consistent JF path package-wide."""
    if getattr(jf, "_tail_consistent_cluster_jf_installed", False):
        return

    original_build = jf.build_embedded_jacobian
    original_solve = jf.ClusterEmbeddedJacobian.solve
    original_response = jf.response_matrix

    def build_embedded_jacobian(
        G,
        Vq,
        bath,
        h_cluster,
        params,
        grid,
        mu,
        rho_cluster,
        *,
        bath_opts=jf.BathTangentOptions(),
        jf_opts=jf.ClusterJFOptions(),
    ):
        op, tangent = original_build(
            G,
            Vq,
            bath,
            h_cluster,
            params,
            grid,
            mu,
            rho_cluster,
            bath_opts=bath_opts,
            jf_opts=jf_opts,
        )
        h0 = build_h0(grid.kmesh(), params)
        Vc = cluster_interaction_matrix(ruby_cluster_interactions(params), 6)
        density = np.real(np.diag(np.asarray(rho_cluster, dtype=complex)))
        sigma_h = hartree_self_energy_matrix(density, Vc)
        op._jf_tail_reference = build_tail_reference(h0, float(mu), sigma_h, grid)
        op._jf_h_cluster = np.asarray(h_cluster, dtype=complex).copy()
        op._jf_bath = BathParameters(
            np.asarray(bath.energies, dtype=float).copy(),
            np.asarray(bath.couplings, dtype=complex).copy(),
            float(bath.fit_error),
            int(bath.nfev),
        )
        op._jf_params = params
        op._jf_interactions = ruby_cluster_interactions(params)
        op._jf_mu = float(mu)
        op._jf_discard_weight_tol = float(bath_opts.discard_weight_tol)
        op._jf_source_fd_step = float(bath_opts.fd_step)
        op._jf_source_sigma_cache = {}
        op._jf_tail_context_cache = {}
        op._jf_dyn_lat_opts = SupercellVertexOptions(
            include_hartree=False,
            include_fock=False,
            include_mt=bool(jf_opts.include_mt),
            include_al=bool(jf_opts.include_al),
            momentum_backend=str(jf_opts.momentum_backend),
            verbose=False,
        )
        op._jf_dyn_cl_opts = SupercellVertexOptions(
            include_hartree=False,
            include_fock=False,
            include_mt=bool(jf_opts.include_mt),
            include_al=bool(jf_opts.include_al),
            momentum_backend="direct",
            verbose=False,
        )
        return op, tangent

    def solve(self, K, q_index, *, initial_gamma=None, recycle=None):
        if not hasattr(self, "_jf_tail_reference"):
            return original_solve(
                self,
                K,
                q_index,
                initial_gamma=initial_gamma,
                recycle=recycle,
            )
        return _solve_consistent(
            self,
            K,
            q_index,
            initial_gamma=initial_gamma,
            recycle=recycle,
        )

    def response_matrix(
        operator,
        vertices,
        q_index,
        *,
        initial_gammas=None,
        recycle=True,
    ):
        if not hasattr(operator, "_jf_tail_reference"):
            return original_response(
                operator,
                vertices,
                q_index,
                initial_gammas=initial_gammas,
                recycle=recycle,
            )
        return _response_matrix_consistent(
            operator,
            vertices,
            q_index,
            initial_gammas=initial_gammas,
            recycle=recycle,
        )

    jf.build_embedded_jacobian = build_embedded_jacobian
    jf.ClusterEmbeddedJacobian.solve = solve
    jf.response_matrix = response_matrix
    jf._tail_consistent_cluster_jf_installed = True


__all__ = ["install_tail_consistent_cluster_jf"]
