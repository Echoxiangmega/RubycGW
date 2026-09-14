"""Jacobian-free finite-q response of the self-consistent cluster-ED+GW embedding.

The converged embedding used in :mod:`rubycgw.cluster_ed_gw_fast` is

    Sigma_tot(k) = Sigma_GW^lat(k) + Sigma_imp^C - Sigma_GW^C,

with the lattice Hartree term included in the full GW tangent.  The ordinary
finite-q cGW code differentiates only the lattice GW functional.  This module
adds the response of the *embedded cluster correction* without constructing a
huge analytic impurity vertex.

The key numerical idea is to linearize the finite-bath fit once.  A change of
the impurity Weiss field is projected onto the retained singular directions of
the bath-fit Jacobian.  The derivative of the ED impurity self-energy along
those bath directions is precomputed by finite differences.  Afterwards every
Krylov matrix-vector product is cheap: it contains FFT GW response, a 6-site
cluster-GW tangent and a small dense solve in the retained bath tangent space,
but no new ED diagonalization.

For an external momentum p,

    X(k;p) = G(k+p) Gamma(k;p) G(k),
    dG_C(p) = <X(k;p)>_k,

and the embedded linear equation is

    [I - L_GW^lat(p) - L_imp^C + L_GW^C] Gamma = K.

The impurity contribution is implicit because

    G0_C^{-1} = G_C^{-1} + Sigma_imp.

If F is the linearized finite-bath impurity map dSigma_imp = F[dG0^{-1}],
then

    dSigma_imp = F[-G_C^{-1} dG_C G_C^{-1} + dSigma_imp].

The low-rank bath representation reduces this local implicit equation to a
small dense system that is factorized once and reused for all q and channels.

Important scope
---------------
* The response is a tangent of the *finite-bath* embedding.  Increasing nbath
  changes the approximation itself, not only the solver tolerance.
* The bath tangent uses a Gauss-Newton linearization of the least-squares fit.
  It is most reliable when the converged bath fit error is already small.
* The outer operator is real-linear (bath parameters are real even when their
  couplings are complex), so Krylov solves are performed on stacked real and
  imaginary parts.  This avoids assuming complex linearity of the bath fit.
* The q=0 pseudospin channels tau_x,tau_y,tau_z are traceless non-density
  irreps on the symmetric background, so fixed-mu and fixed-filling linear
  responses coincide by symmetry.  If that symmetry is already broken, a
  separate density/chemical-potential constraint must be included.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, gcrotmk, gmres

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_gw_self_energy,
    ruby_cluster_interactions,
)
from .finite_q_cgw import (
    normalize_q_index,
    susceptibility_matrix_finite_q,
    vertex_corrections_finite_q,
)
from .grids import MatsubaraGrid, roll_spatial
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .supercell_cgw import SupercellVertexOptions, vertex_corrections_q0


def _pack_complex(a: np.ndarray) -> np.ndarray:
    z = np.asarray(a, dtype=complex)
    return np.concatenate([z.real.ravel(), z.imag.ravel()])


def _unpack_complex(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(np.prod(shape))
    if x.size != 2 * n:
        raise ValueError("packed complex-vector size mismatch")
    return x[:n].reshape(shape) + 1j * x[n:].reshape(shape)


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _bath_theta(bath: BathParameters) -> np.ndarray:
    eps = np.asarray(bath.energies, dtype=float).reshape(-1)
    hyb = np.asarray(bath.couplings, dtype=complex)
    return np.concatenate([eps, hyb.real.ravel(), hyb.imag.ravel()])


def _bath_from_theta(theta: np.ndarray, norb: int, nbath: int) -> BathParameters:
    t = np.asarray(theta, dtype=float).reshape(-1)
    nvc = int(norb) * int(nbath)
    if t.size != int(nbath) + 2 * nvc:
        raise ValueError("bath tangent parameter size mismatch")
    eps = t[:nbath]
    re = t[nbath:nbath + nvc].reshape(norb, nbath)
    im = t[nbath + nvc:].reshape(norb, nbath)
    return BathParameters(
        np.asarray(eps, dtype=float),
        np.asarray(re + 1j * im, dtype=complex),
        np.nan,
        0,
    )


def _complex_delta_derivatives(
    omega: np.ndarray,
    mu: float,
    bath: BathParameters,
):
    """Yield dDelta/dtheta for [eps, Re(V), Im(V)] real bath parameters."""
    w = np.asarray(omega, dtype=float).reshape(-1)
    eps = np.asarray(bath.energies, dtype=float).reshape(-1)
    hyb = np.asarray(bath.couplings, dtype=complex)
    norb, nbath = hyb.shape
    den = 1.0 / (1j * w[:, None] + float(mu) - eps[None, :])

    for p in range(nbath):
        outer = np.outer(hyb[:, p], hyb[:, p].conj())
        yield (den[:, p] ** 2)[:, None, None] * outer[None, :, :]

    for a in range(norb):
        for p in range(nbath):
            d = np.zeros((len(w), norb, norb), dtype=complex)
            vp = hyb[:, p]
            d[:, a, :] += den[:, p, None] * vp.conj()[None, :]
            d[:, :, a] += den[:, p, None] * vp[None, :]
            yield d

    for a in range(norb):
        for p in range(nbath):
            d = np.zeros((len(w), norb, norb), dtype=complex)
            vp = hyb[:, p]
            d[:, a, :] += 1j * den[:, p, None] * vp.conj()[None, :]
            d[:, :, a] += -1j * den[:, p, None] * vp[None, :]
            yield d


@dataclass(frozen=True)
class BathTangentOptions:
    """Controls the one-time finite-bath tangent construction."""

    fit_metric: str = "delta"  # must match the converged embedding fit
    nfit: int = 12
    svd_rcond: float = 1.0e-7
    max_rank: int | None = 24
    fd_step: float = 2.0e-4
    fd_scheme: str = "centered"  # centered or forward
    discard_weight_tol: float = 1.0e-11
    verbose: bool = True


@dataclass
class BathTangentModel:
    """Low-rank real-linear tangent of the finite-bath impurity map."""

    theta0: np.ndarray
    mode_vectors: np.ndarray
    singular_values: np.ndarray
    Ufit: np.ndarray
    sigma_modes: np.ndarray
    inner_matrix: np.ndarray
    fit_indices: np.ndarray
    fit_weights: np.ndarray
    fit_metric: str
    model_g0_fit: np.ndarray
    Gc_inv: np.ndarray
    rank: int
    condition_number: float
    build_seconds: float
    fd_step: float

    def _target_vector(self, dg0_inv: np.ndarray) -> np.ndarray:
        d = np.asarray(dg0_inv, dtype=complex)[self.fit_indices]
        if self.fit_metric == "g0":
            # dG0 = -G0 dG0^{-1} G0.
            target = -np.einsum(
                "nab,nbc,ncd->nad",
                self.model_g0_fit,
                d,
                self.model_g0_fit,
                optimize=True,
            )
        else:
            # G0^{-1}=iw+mu-h-Delta, hence dDelta=-dG0^{-1}.
            target = -d * self.fit_weights[:, None, None]
        return _pack_complex(target)

    def coefficients_from_dg0_inv(self, dg0_inv: np.ndarray) -> np.ndarray:
        y = self._target_vector(dg0_inv)
        # J = U diag(s) V^T; bath mode coordinates are alpha=U^T y / s.
        return (self.Ufit.T @ y) / self.singular_values

    def impurity_sigma_from_delta_gc(self, delta_gc: np.ndarray) -> np.ndarray:
        dgc = np.asarray(delta_gc, dtype=complex)
        dg0_from_gc = -np.einsum(
            "nab,nbc,ncd->nad",
            self.Gc_inv,
            dgc,
            self.Gc_inv,
            optimize=True,
        )
        rhs = self.coefficients_from_dg0_inv(dg0_from_gc)
        coeff = np.linalg.solve(self.inner_matrix, rhs)
        return np.einsum("r,rnab->nab", coeff, self.sigma_modes, optimize=True)


def _impurity_sigma_for_bath(
    bath: BathParameters,
    h_cluster: np.ndarray,
    interactions,
    grid: MatsubaraGrid,
    mu: float,
    discard_weight_tol: float,
) -> np.ndarray:
    himp = build_impurity_one_body(h_cluster, bath)
    impurity = FiniteBathImpurityED(
        himp,
        interactions,
        correlated_orbitals=tuple(range(NSUB)),
    )
    impurity.diagonalize()
    Gimp, _ = impurity.green_iomega(
        1j * grid.omega,
        float(mu),
        float(grid.T),
        orbitals=tuple(range(NSUB)),
        discard_weight_tol=float(discard_weight_tol),
    )
    delta = bath_hybridization(
        grid.omega,
        float(mu),
        bath.energies,
        bath.couplings,
    )
    eye = np.eye(NSUB, dtype=complex)
    g0_inv = (
        (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
        - np.asarray(h_cluster, dtype=complex)[None, :, :]
        - delta
    )
    return g0_inv - np.linalg.inv(Gimp)


def build_bath_tangent_model(
    bath: BathParameters,
    G_cluster: np.ndarray,
    h_cluster: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    mu: float,
    opts: BathTangentOptions = BathTangentOptions(),
) -> BathTangentModel:
    """Precompute a low-rank tangent of the finite-bath impurity self-energy.

    The expensive ED finite differences are performed only here.  Subsequent
    Jacobian-vector products contain no ED diagonalization.
    """
    t0 = perf_counter()
    metric = str(opts.fit_metric).lower()
    if metric not in ("delta", "g0"):
        raise ValueError("fit_metric must be 'delta' or 'g0'")
    scheme = str(opts.fd_scheme).lower()
    if scheme not in ("centered", "forward"):
        raise ValueError("fd_scheme must be 'centered' or 'forward'")
    if float(opts.fd_step) <= 0.0:
        raise ValueError("fd_step must be positive")

    Gc = np.asarray(G_cluster, dtype=complex)
    h = np.asarray(h_cluster, dtype=complex)
    if Gc.shape != (grid.nf, NSUB, NSUB):
        raise ValueError("G_cluster shape mismatch")
    if h.shape != (NSUB, NSUB):
        raise ValueError("h_cluster shape mismatch")

    theta0 = _bath_theta(bath)
    nbath = len(np.asarray(bath.energies).reshape(-1))
    if NSUB + nbath > 16:
        raise ValueError(
            "dense tangent precomputation supports at most 16 impurity orbitals; "
            "reduce nbath or add the Lanczos impurity backend"
        )

    pos = np.flatnonzero(np.asarray(grid.omega) > 0.0)
    pos = pos[np.argsort(np.asarray(grid.omega)[pos])[: min(int(opts.nfit), len(pos))]]
    if pos.size == 0:
        raise ValueError("bath tangent needs at least one positive Matsubara point")
    wfit = np.asarray(grid.omega)[pos]
    delta_fit = bath_hybridization(wfit, mu, bath.energies, bath.couplings)
    eye = np.eye(NSUB, dtype=complex)
    g0_fit = np.linalg.inv(
        (1j * wfit[:, None, None] + float(mu)) * eye[None, :, :]
        - h[None, :, :]
        - delta_fit
    )
    w0 = max(float(wfit[0]), 1.0e-12)
    weights = 1.0 / np.sqrt(wfit * wfit + w0 * w0)
    weights /= float(np.max(weights))

    columns = []
    for ddelta in _complex_delta_derivatives(wfit, mu, bath):
        if metric == "g0":
            field = np.einsum(
                "nab,nbc,ncd->nad", g0_fit, ddelta, g0_fit, optimize=True
            )
        else:
            field = ddelta * weights[:, None, None]
        columns.append(_pack_complex(field))
    J = np.stack(columns, axis=1)
    U, s, Vh = np.linalg.svd(J, full_matrices=False)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        raise RuntimeError("bath-fit Jacobian has no finite singular directions")
    keep = np.flatnonzero(s >= float(opts.svd_rcond) * float(s[0]))
    if opts.max_rank is not None:
        keep = keep[: max(int(opts.max_rank), 1)]
    if keep.size == 0:
        keep = np.asarray([0], dtype=int)
    rank = int(len(keep))
    Uret = np.asarray(U[:, keep], dtype=float)
    sret = np.asarray(s[keep], dtype=float)
    modes = np.asarray(Vh[keep].T, dtype=float)

    interactions = ruby_cluster_interactions(params)
    sigma0 = None
    if scheme == "forward":
        sigma0 = _impurity_sigma_for_bath(
            bath, h, interactions, grid, mu, opts.discard_weight_tol
        )

    sigma_modes = np.empty((rank, grid.nf, NSUB, NSUB), dtype=complex)
    step = float(opts.fd_step)
    for j in range(rank):
        direction = modes[:, j]
        if opts.verbose:
            print(
                f"[JF bath] impurity tangent mode {j+1}/{rank}: "
                f"s/s0={sret[j]/sret[0]:.3e}",
                flush=True,
            )
        plus = _bath_from_theta(theta0 + step * direction, NSUB, nbath)
        sig_plus = _impurity_sigma_for_bath(
            plus, h, interactions, grid, mu, opts.discard_weight_tol
        )
        if scheme == "centered":
            minus = _bath_from_theta(theta0 - step * direction, NSUB, nbath)
            sig_minus = _impurity_sigma_for_bath(
                minus, h, interactions, grid, mu, opts.discard_weight_tol
            )
            sigma_modes[j] = (sig_plus - sig_minus) / (2.0 * step)
        else:
            sigma_modes[j] = (sig_plus - sigma0) / step

    Gc_inv = np.linalg.inv(Gc)

    # F[dg0_inv] = sum_r sigma_modes[r] * a_r(dg0_inv).  Because dSigma_imp
    # itself enters dg0_inv, solve the local implicit equation in the retained
    # bath-mode coordinates: (I-M) a = rhs.
    tmp = BathTangentModel(
        theta0=theta0,
        mode_vectors=modes,
        singular_values=sret,
        Ufit=Uret,
        sigma_modes=sigma_modes,
        inner_matrix=np.eye(rank),
        fit_indices=np.asarray(pos, dtype=int),
        fit_weights=np.asarray(weights, dtype=float),
        fit_metric=metric,
        model_g0_fit=np.asarray(g0_fit),
        Gc_inv=Gc_inv,
        rank=rank,
        condition_number=np.nan,
        build_seconds=np.nan,
        fd_step=step,
    )
    M = np.empty((rank, rank), dtype=float)
    for j in range(rank):
        M[:, j] = tmp.coefficients_from_dg0_inv(sigma_modes[j])
    inner = np.eye(rank) - M
    cond = float(np.linalg.cond(inner))
    elapsed = perf_counter() - t0
    if opts.verbose:
        print(
            f"[JF bath] retained rank={rank}/{len(s)}, "
            f"s_min/s_max={sret[-1]/sret[0]:.3e}, "
            f"cond(I-M)={cond:.3e}, build={elapsed:.1f}s",
            flush=True,
        )
    return BathTangentModel(
        theta0=theta0,
        mode_vectors=modes,
        singular_values=sret,
        Ufit=Uret,
        sigma_modes=sigma_modes,
        inner_matrix=inner,
        fit_indices=np.asarray(pos, dtype=int),
        fit_weights=np.asarray(weights, dtype=float),
        fit_metric=metric,
        model_g0_fit=np.asarray(g0_fit),
        Gc_inv=Gc_inv,
        rank=rank,
        condition_number=cond,
        build_seconds=float(elapsed),
        fd_step=step,
    )


@dataclass(frozen=True)
class ClusterJFOptions:
    solver: str = "gcrotmk"  # gcrotmk or gmres
    tol: float = 1.0e-8
    maxiter: int = 80
    restart: int = 24
    recycle_dim: int = 12
    include_hartree: bool = True
    include_fock: bool = True
    include_mt: bool = True
    include_al: bool = True
    momentum_backend: str = "fft"
    verbose: bool = True


@dataclass
class ClusterJFResult:
    Gamma: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    solver_info: int
    q_index: tuple[int, int]


class ClusterEmbeddedJacobian:
    """Matrix-free embedded response operator at a fixed converged background."""

    def __init__(
        self,
        G: np.ndarray,
        W: np.ndarray,
        Vq: np.ndarray,
        G_cluster: np.ndarray,
        W_cluster: np.ndarray,
        V_cluster: np.ndarray,
        grid: MatsubaraGrid,
        bath_tangent: BathTangentModel,
        opts: ClusterJFOptions = ClusterJFOptions(),
    ):
        self.G = np.asarray(G, dtype=complex)
        self.W = np.asarray(W, dtype=complex)
        self.Vq = np.asarray(Vq, dtype=complex)
        self.Gc = np.asarray(G_cluster, dtype=complex)
        self.Wc = np.asarray(W_cluster, dtype=complex)
        self.Vc = np.asarray(V_cluster, dtype=complex)
        self.grid = grid
        self.bath_tangent = bath_tangent
        self.opts = opts
        if self.G.shape != (grid.nf, grid.nk1, grid.nk2, NSUB, NSUB):
            raise ValueError("embedded lattice G shape mismatch")
        if self.W.shape != (grid.nb, grid.nk1, grid.nk2, NSUB, NSUB):
            raise ValueError("embedded lattice W shape mismatch")
        if self.Gc.shape != (grid.nf, NSUB, NSUB):
            raise ValueError("cluster G shape mismatch")
        if self.Wc.shape != (grid.nb, NSUB, NSUB):
            raise ValueError("cluster W shape mismatch")

        self.lat_opts = SupercellVertexOptions(
            include_hartree=bool(opts.include_hartree),
            include_fock=bool(opts.include_fock),
            include_mt=bool(opts.include_mt),
            include_al=bool(opts.include_al),
            momentum_backend=str(opts.momentum_backend),
            verbose=False,
        )
        self.cl_opts = SupercellVertexOptions(
            include_hartree=bool(opts.include_hartree),
            include_fock=bool(opts.include_fock),
            include_mt=bool(opts.include_mt),
            include_al=bool(opts.include_al),
            momentum_backend="direct",
            verbose=False,
        )
        self.cluster_grid = MatsubaraGrid(
            nk1=1,
            nk2=1,
            nw=grid.nw,
            nOmega=grid.nOmega,
            T=grid.T,
        )
        self._Gc5 = self.Gc[:, None, None]
        self._Wc5 = self.Wc[:, None, None]
        self._Vc4 = self.Vc[None, None]

    def _x_field(self, Gamma: np.ndarray, q_index: tuple[int, int]) -> np.ndarray:
        p = normalize_q_index(q_index, self.grid)
        Gp = roll_spatial(self.G, p[0], p[1])
        return np.einsum(
            "...ab,...bc,...cd->...ad", Gp, Gamma, self.G, optimize=True
        )

    def _cluster_gw_tangent(self, delta_gc: np.ndarray) -> np.ndarray:
        invg = np.linalg.inv(self.Gc)
        gamma_c = np.einsum(
            "nab,nbc,ncd->nad", invg, delta_gc, invg, optimize=True
        )
        parts = vertex_corrections_q0(
            self._Gc5,
            self._Wc5,
            self._Vc4,
            gamma_c[:, None, None],
            self.cluster_grid,
            self.cl_opts,
        )
        total = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
        return np.asarray(total[:, 0, 0], dtype=complex)

    def kernel(self, Gamma: np.ndarray, q_index: tuple[int, int]) -> np.ndarray:
        p = normalize_q_index(q_index, self.grid)
        parts = vertex_corrections_finite_q(
            self.G,
            self.W,
            self.Vq,
            Gamma,
            p,
            self.grid,
            self.lat_opts,
        )
        lat = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
        X = self._x_field(Gamma, p)
        delta_gc = np.mean(X, axis=(1, 2))
        d_sigma_imp = self.bath_tangent.impurity_sigma_from_delta_gc(delta_gc)
        d_sigma_cgw = self._cluster_gw_tangent(delta_gc)
        local = d_sigma_imp - d_sigma_cgw
        return lat + local[:, None, None, :, :]

    def apply_A(self, Gamma: np.ndarray, q_index: tuple[int, int]) -> np.ndarray:
        return np.asarray(Gamma, dtype=complex) - self.kernel(Gamma, q_index)

    def linear_operator(self, q_index: tuple[int, int]) -> LinearOperator:
        shape = self.G.shape
        n = 2 * int(np.prod(shape))

        def matvec(x):
            gamma = _unpack_complex(x, shape)
            return _pack_complex(self.apply_A(gamma, q_index))

        return LinearOperator((n, n), matvec=matvec, dtype=float)

    def solve(
        self,
        K: np.ndarray,
        q_index: tuple[int, int],
        *,
        initial_gamma: np.ndarray | None = None,
        recycle: list | None = None,
    ) -> ClusterJFResult:
        p = normalize_q_index(q_index, self.grid)
        K = np.asarray(K, dtype=complex)
        if K.shape != (NSUB, NSUB):
            raise ValueError("bare vertex must be a 6x6 primitive-cell matrix")
        Kfield = np.broadcast_to(K, self.G.shape).copy()
        if initial_gamma is None or np.asarray(initial_gamma).shape != self.G.shape:
            gamma0 = Kfield
        else:
            gamma0 = np.asarray(initial_gamma, dtype=complex)
        A = self.linear_operator(p)
        b = _pack_complex(Kfield)
        x0 = _pack_complex(gamma0)
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
                x, info = gcrotmk(A, b, rtol=float(self.opts.tol), **kwargs)
            except TypeError:
                x, info = gcrotmk(A, b, tol=float(self.opts.tol), **kwargs)
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
                    A, b, rtol=float(self.opts.tol), callback_type="pr_norm", **kwargs
                )
            except TypeError:
                x, info = gmres(A, b, tol=float(self.opts.tol), **kwargs)
        else:
            raise ValueError("solver must be 'gcrotmk' or 'gmres'")

        Gamma = _unpack_complex(x, self.G.shape)
        residual = self.apply_A(Gamma, p) - Kfield
        err = _maxabs(residual)
        converged = bool(int(info) == 0 and np.isfinite(err) and err < float(self.opts.tol))
        if self.opts.verbose:
            print(
                f"[cluster-JF] q={p}, solver={solver}, it={count[0]}, "
                f"info={info}, residual={err:.3e}",
                flush=True,
            )
        return ClusterJFResult(
            Gamma=np.asarray(Gamma),
            converged=converged,
            iterations=int(count[0]),
            final_error=float(err),
            solver_info=int(info),
            q_index=p,
        )


def build_embedded_jacobian(
    G: np.ndarray,
    Vq: np.ndarray,
    bath: BathParameters,
    h_cluster: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    mu: float,
    rho_cluster: np.ndarray,
    *,
    bath_opts: BathTangentOptions = BathTangentOptions(),
    jf_opts: ClusterJFOptions = ClusterJFOptions(),
) -> tuple[ClusterEmbeddedJacobian, BathTangentModel]:
    """Construct the complete embedded Jacobian from a converged lattice G."""
    from .supercell_gw import compute_polarization_matrix, compute_screened_interaction_matrix

    G = np.asarray(G, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    P = compute_polarization_matrix(G, grid, backend=str(jf_opts.momentum_backend))
    W = compute_screened_interaction_matrix(P, Vq)
    Gc = np.mean(G, axis=(1, 2))
    Vc = np.asarray(Vq[0, 0], dtype=complex)
    _, _, Wc = cluster_gw_self_energy(
        Gc,
        np.asarray(rho_cluster, dtype=complex),
        Vc,
        grid,
    )
    tangent = build_bath_tangent_model(
        bath,
        Gc,
        np.asarray(h_cluster, dtype=complex),
        params,
        grid,
        mu,
        bath_opts,
    )
    op = ClusterEmbeddedJacobian(
        G,
        W,
        Vq,
        Gc,
        Wc,
        Vc,
        grid,
        tangent,
        jf_opts,
    )
    return op, tangent


def response_matrix(
    operator: ClusterEmbeddedJacobian,
    vertices: np.ndarray,
    q_index: tuple[int, int],
    *,
    initial_gammas: list[np.ndarray | None] | None = None,
    recycle: bool = True,
) -> tuple[np.ndarray, list[ClusterJFResult]]:
    """Solve all driven channels and return their static susceptibility matrix."""
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
    chi = susceptibility_matrix_finite_q(
        operator.G,
        K,
        [r.Gamma for r in results],
        q_index,
        operator.grid,
    )
    return np.asarray(chi), results


__all__ = [
    "BathTangentOptions",
    "BathTangentModel",
    "ClusterJFOptions",
    "ClusterJFResult",
    "ClusterEmbeddedJacobian",
    "build_bath_tangent_model",
    "build_embedded_jacobian",
    "response_matrix",
]
