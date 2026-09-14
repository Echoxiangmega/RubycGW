"""Jacobian-free finite-q response of the cluster-ED + lattice-GW embedding.

The zero-field embedding used in :mod:`rubycgw.cluster_ed_gw_fast` is

    Sigma_tot = Sigma_H^lat + Sigma_GW^lat + Sigma_imp - Sigma_GW^C.

For one static external momentum q we solve directly for the full response
vertex ``Gamma(k;q)`` and the impurity self-energy tangent ``S_imp(iw;q)``.
The lattice GW and cluster-GW tangents are evaluated analytically with the
existing cGW kernels. Only the finite-bath ED impurity map is differentiated
numerically, so the expensive finite-difference work remains local.

The linear system is

    Gamma - L_GW,q[Gamma] - S_imp + L_GW,C[dG_C] + dmu I = K,
    S_imp - D Sigma_imp[g0_C^{-1}] [d g0_C^{-1}] = 0,

with

    dG(k;q) = G(k+q) Gamma(k;q) G(k),
    dG_C     = <dG(k;q)>_k,
    d g0_C^{-1} = -G_C^{-1} dG_C G_C^{-1} + S_imp.

At q=0 an additional fixed-filling equation enforces ``dN=0`` and determines
``dmu``. At q!=0 the chemical-potential tangent is constrained to zero.

The impurity derivative is Jacobian-free. Two bath-derivative modes are
provided:

``linearized``
    Differentiate the finite-bath least-squares fit with a Gauss-Newton
    pseudoinverse, then finite-difference only the ED solve. This is the fast
    production mode and is smooth enough for Krylov iteration when the base
    bath fit is good.

``refit``
    Refit the nonlinear bath at every +/- directional perturbation. This is
    much slower but is useful as a validation of the linearized-bath tangent.

Because the finite-bath parametrization is a real manifold, the Krylov system
is represented as a real vector containing real and imaginary parts of all
complex fields. This avoids incorrectly assuming that the bath projection is
complex-linear.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, gcrotmk

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_gw_self_energy,
)
from .cluster_ed_gw_covariant import fit_finite_bath_complex
from .finite_q_cgw import (
    normalize_q_index,
    susceptibility_matrix_finite_q,
    vertex_corrections_finite_q,
)
from .grids import MatsubaraGrid, roll_spatial
from .impurity_ed import FiniteBathImpurityED
from .supercell_cgw import SupercellVertexOptions, vertex_corrections_q0


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _fro(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a).ravel()))


@dataclass(frozen=True)
class ClusterJFOptions:
    """Numerical controls for the finite-q embedded Jacobian solve."""

    rtol: float = 2.0e-5
    atol: float = 0.0
    maxiter: int = 80
    gcrot_m: int = 20
    gcrot_k: int = 8
    preconditioner_order: int = 1

    bath_derivative_mode: str = "linearized"  # linearized | refit
    difference_scheme: str = "forward"        # forward | central
    fd_rel_step: float = 2.0e-5
    fd_min_step: float = 1.0e-8
    fd_max_step: float = 5.0e-3

    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 400
    bath_fit_xtol: float = 1.0e-10
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_svd_rcond: float = 1.0e-9
    bath_tikhonov: float = 0.0

    discard_weight_tol: float = 1.0e-11
    verbose: bool = True

    def __post_init__(self):
        mode = str(self.bath_derivative_mode).lower()
        if mode not in ("linearized", "refit"):
            raise ValueError("bath_derivative_mode must be 'linearized' or 'refit'")
        scheme = str(self.difference_scheme).lower()
        if scheme not in ("forward", "central"):
            raise ValueError("difference_scheme must be 'forward' or 'central'")
        if int(self.gcrot_m) < 1 or int(self.gcrot_k) < 0:
            raise ValueError("invalid GCROT dimensions")
        if int(self.preconditioner_order) < 0:
            raise ValueError("preconditioner_order must be nonnegative")
        if not (0.0 < float(self.fd_min_step) <= float(self.fd_max_step)):
            raise ValueError("invalid finite-difference step bounds")


@dataclass
class BathLinearizationDiagnostics:
    rank: int
    n_parameters: int
    condition_number: float
    smallest_kept_singular: float
    largest_singular: float


class LinearizedBathFit:
    """Gauss-Newton derivative of the complex finite-bath least-squares fit."""

    def __init__(
        self,
        omega: np.ndarray,
        mu: float,
        bath: BathParameters,
        *,
        nfit: int = 12,
        rcond: float = 1.0e-9,
        tikhonov: float = 0.0,
    ):
        self.omega = np.asarray(omega, dtype=float).reshape(-1)
        self.mu = float(mu)
        self.bath = BathParameters(
            np.asarray(bath.energies, dtype=float).copy(),
            np.asarray(bath.couplings, dtype=complex).copy(),
            float(bath.fit_error),
            int(bath.nfev),
        )
        self.norb, self.nbath = self.bath.couplings.shape
        pos = np.flatnonzero(self.omega > 0.0)
        if pos.size == 0:
            raise ValueError("bath tangent requires positive Matsubara frequencies")
        pos = pos[np.argsort(self.omega[pos])[: min(int(nfit), len(pos))]]
        self.fit_indices = np.asarray(pos, dtype=int)
        self.wfit = self.omega[self.fit_indices]
        w0 = max(float(self.wfit[0]), 1.0e-12)
        self.weights = 1.0 / np.sqrt(self.wfit * self.wfit + w0 * w0)
        self.weights /= float(np.max(self.weights))
        self.rcond = float(rcond)
        self.tikhonov = float(tikhonov)

        J = self._build_real_jacobian()
        U, s, Vh = np.linalg.svd(J, full_matrices=False)
        smax = float(s[0]) if len(s) else 0.0
        cutoff = float(self.rcond) * max(smax, np.finfo(float).tiny)
        keep = s > cutoff
        if not np.any(keep):
            raise RuntimeError("finite-bath fit Jacobian has zero numerical rank")
        sk = s[keep]
        if self.tikhonov > 0.0:
            invs = sk / (sk * sk + self.tikhonov * self.tikhonov)
        else:
            invs = 1.0 / sk
        self._pinv = (Vh[keep].T * invs[None, :]) @ U[:, keep].T
        smin = float(sk[-1])
        self.diagnostics = BathLinearizationDiagnostics(
            rank=int(np.count_nonzero(keep)),
            n_parameters=int(J.shape[1]),
            condition_number=(smax / smin if smin > 0.0 else np.inf),
            smallest_kept_singular=smin,
            largest_singular=smax,
        )

    @property
    def n_parameters(self) -> int:
        return int(self.nbath + 2 * self.norb * self.nbath)

    def _build_real_jacobian(self) -> np.ndarray:
        eps = np.asarray(self.bath.energies, dtype=float)
        V = np.asarray(self.bath.couplings, dtype=complex)
        nw = len(self.wfit)
        npar = self.n_parameters
        D = 1j * self.wfit[:, None] + self.mu - eps[None, :]
        invD = 1.0 / D
        deriv = np.zeros((nw, self.norb, self.norb, npar), dtype=complex)

        for p in range(self.nbath):
            outer = np.outer(V[:, p], V[:, p].conj())
            deriv[:, :, :, p] = outer[None, :, :] * (invD[:, p] ** 2)[:, None, None]

        off_re = self.nbath
        off_im = self.nbath + self.norb * self.nbath
        for a in range(self.norb):
            e = np.zeros(self.norb, dtype=complex)
            e[a] = 1.0
            for p in range(self.nbath):
                v = V[:, p]
                d_re = np.outer(e, v.conj()) + np.outer(v, e.conj())
                d_im = 1j * np.outer(e, v.conj()) - 1j * np.outer(v, e.conj())
                col = a * self.nbath + p
                deriv[:, :, :, off_re + col] = d_re[None, :, :] * invD[:, p, None, None]
                deriv[:, :, :, off_im + col] = d_im[None, :, :] * invD[:, p, None, None]

        weighted = deriv * self.weights[:, None, None, None]
        flat = weighted.reshape(-1, npar)
        return np.concatenate([flat.real, flat.imag], axis=0)

    def parameter_direction(self, ddelta: np.ndarray) -> np.ndarray:
        arr = np.asarray(ddelta, dtype=complex)
        if arr.shape != (len(self.omega), self.norb, self.norb):
            raise ValueError("ddelta shape mismatch")
        rhs_c = arr[self.fit_indices] * self.weights[:, None, None]
        rhs = np.concatenate([rhs_c.real.ravel(), rhs_c.imag.ravel()])
        return np.asarray(self._pinv @ rhs, dtype=float)

    def theta(self) -> np.ndarray:
        V = np.asarray(self.bath.couplings, dtype=complex)
        return np.concatenate([
            np.asarray(self.bath.energies, dtype=float),
            V.real.ravel(),
            V.imag.ravel(),
        ])

    def bath_from_theta(self, theta: np.ndarray) -> BathParameters:
        x = np.asarray(theta, dtype=float).reshape(-1)
        if len(x) != self.n_parameters:
            raise ValueError("bath theta size mismatch")
        nb = self.nbath
        nc = self.norb * self.nbath
        eps = x[:nb]
        re = x[nb:nb + nc].reshape(self.norb, self.nbath)
        im = x[nb + nc:].reshape(self.norb, self.nbath)
        return BathParameters(
            np.asarray(eps, dtype=float),
            np.asarray(re + 1j * im, dtype=complex),
            float(self.bath.fit_error),
            0,
        )


@dataclass
class ImpurityJFDiagnostics:
    calls: int = 0
    ed_solves: int = 0
    bath_refits: int = 0
    last_step: float = 0.0
    last_direction_norm: float = 0.0
    last_bath_error_plus: float = np.nan
    last_bath_error_minus: float = np.nan


class ImpuritySelfEnergyJF:
    """Jacobian-vector products for the finite-bath ED self-energy map."""

    def __init__(
        self,
        grid: MatsubaraGrid,
        mu: float,
        h_cluster: np.ndarray,
        interactions,
        bath: BathParameters,
        base_g0_inv: np.ndarray,
        opts: ClusterJFOptions,
    ):
        self.grid = grid
        self.mu = float(mu)
        self.h_cluster = np.asarray(h_cluster, dtype=complex)
        self.interactions = tuple(interactions)
        self.opts = opts
        self.eye = np.eye(self.h_cluster.shape[0], dtype=complex)
        self.base_g0_inv = np.asarray(base_g0_inv, dtype=complex)
        self.bath = BathParameters(
            np.asarray(bath.energies, dtype=float).copy(),
            np.asarray(bath.couplings, dtype=complex).copy(),
            float(bath.fit_error),
            int(bath.nfev),
        )
        self.diag = ImpurityJFDiagnostics()

        if str(opts.bath_derivative_mode).lower() == "refit":
            target_delta = self._target_delta(self.base_g0_inv)
            self.bath = fit_finite_bath_complex(
                target_delta,
                grid.omega,
                self.mu,
                nbath=len(self.bath.energies),
                nfit=int(opts.bath_fit_nfreq),
                max_nfev=int(opts.bath_fit_max_nfev),
                energy_window=float(opts.bath_energy_window),
                coupling_bound=float(opts.bath_coupling_bound),
                xtol=float(opts.bath_fit_xtol),
                initial=self.bath,
            )
            self.diag.bath_refits += 1

        self.linearizer = LinearizedBathFit(
            grid.omega,
            self.mu,
            self.bath,
            nfit=int(opts.bath_fit_nfreq),
            rcond=float(opts.bath_svd_rcond),
            tikhonov=float(opts.bath_tikhonov),
        )
        self.base_sigma, self.base_bath_error = self._sigma_for_bath(self.bath)

    def _target_delta(self, g0_inv: np.ndarray) -> np.ndarray:
        return (
            (1j * self.grid.omega[:, None, None] + self.mu) * self.eye[None, :, :]
            - self.h_cluster[None, :, :]
            - np.asarray(g0_inv, dtype=complex)
        )

    def _sigma_for_bath(self, bath: BathParameters) -> tuple[np.ndarray, float]:
        himp = build_impurity_one_body(self.h_cluster, bath)
        impurity = FiniteBathImpurityED(
            himp,
            self.interactions,
            correlated_orbitals=tuple(range(self.h_cluster.shape[0])),
        )
        impurity.diagonalize()
        self.diag.ed_solves += 1
        Gimp, _ = impurity.green_iomega(
            1j * self.grid.omega,
            self.mu,
            self.grid.T,
            orbitals=tuple(range(self.h_cluster.shape[0])),
            discard_weight_tol=float(self.opts.discard_weight_tol),
        )
        delta_fit = bath_hybridization(
            self.grid.omega,
            self.mu,
            bath.energies,
            bath.couplings,
        )
        g0_fit_inv = (
            (1j * self.grid.omega[:, None, None] + self.mu) * self.eye[None, :, :]
            - self.h_cluster[None, :, :]
            - delta_fit
        )
        sigma = g0_fit_inv - np.linalg.inv(Gimp)
        target = self._target_delta(self.base_g0_inv)
        den = max(_fro(target), 1.0e-300)
        fit_error = _fro(delta_fit - target) / den
        return np.asarray(sigma), float(fit_error)

    def _scaled_step(self, direction: np.ndarray, reference: np.ndarray) -> float:
        dv = _maxabs(direction)
        if dv <= 1.0e-14:
            return 0.0
        scale = max(1.0, _maxabs(reference))
        step = float(self.opts.fd_rel_step) * scale / dv
        return float(np.clip(step, self.opts.fd_min_step, self.opts.fd_max_step))

    def _direction_linearized_bath(self, dg0_inv: np.ndarray) -> np.ndarray:
        dtheta = self.linearizer.parameter_direction(-np.asarray(dg0_inv, dtype=complex))
        theta0 = self.linearizer.theta()
        step = self._scaled_step(dtheta, theta0)
        if step == 0.0:
            return np.zeros_like(self.base_sigma)
        self.diag.last_step = float(step)
        self.diag.last_direction_norm = _fro(dg0_inv)
        scheme = str(self.opts.difference_scheme).lower()

        bp = self.linearizer.bath_from_theta(theta0 + step * dtheta)
        sp, ep = self._sigma_for_bath(bp)
        self.diag.last_bath_error_plus = float(ep)
        if scheme == "forward":
            return (sp - self.base_sigma) / step

        bm = self.linearizer.bath_from_theta(theta0 - step * dtheta)
        sm, em = self._sigma_for_bath(bm)
        self.diag.last_bath_error_minus = float(em)
        return (sp - sm) / (2.0 * step)

    def _solve_refit_sigma(self, g0_inv: np.ndarray) -> tuple[np.ndarray, float]:
        target_delta = self._target_delta(g0_inv)
        bath = fit_finite_bath_complex(
            target_delta,
            self.grid.omega,
            self.mu,
            nbath=len(self.bath.energies),
            nfit=int(self.opts.bath_fit_nfreq),
            max_nfev=int(self.opts.bath_fit_max_nfev),
            energy_window=float(self.opts.bath_energy_window),
            coupling_bound=float(self.opts.bath_coupling_bound),
            xtol=float(self.opts.bath_fit_xtol),
            initial=self.bath,
        )
        self.diag.bath_refits += 1
        sigma, _ = self._sigma_for_bath(bath)
        return sigma, float(bath.fit_error)

    def _direction_refit(self, dg0_inv: np.ndarray) -> np.ndarray:
        step = self._scaled_step(dg0_inv, self.base_g0_inv)
        if step == 0.0:
            return np.zeros_like(self.base_sigma)
        self.diag.last_step = float(step)
        self.diag.last_direction_norm = _fro(dg0_inv)
        gp = self.base_g0_inv + step * dg0_inv
        sp, ep = self._solve_refit_sigma(gp)
        self.diag.last_bath_error_plus = float(ep)
        if str(self.opts.difference_scheme).lower() == "forward":
            return (sp - self.base_sigma) / step
        gm = self.base_g0_inv - step * dg0_inv
        sm, em = self._solve_refit_sigma(gm)
        self.diag.last_bath_error_minus = float(em)
        return (sp - sm) / (2.0 * step)

    def direction(self, dg0_inv: np.ndarray) -> np.ndarray:
        self.diag.calls += 1
        if _maxabs(dg0_inv) <= 1.0e-14:
            return np.zeros_like(self.base_sigma)
        if str(self.opts.bath_derivative_mode).lower() == "linearized":
            return self._direction_linearized_bath(dg0_inv)
        return self._direction_refit(dg0_inv)


@dataclass
class ClusterJFQResult:
    q_index: tuple[int, int]
    Gamma: np.ndarray
    Sigma_imp_tangent: np.ndarray
    dmu: complex
    converged: bool
    info: int
    matvecs: int
    residual_max: float
    elapsed: float


class ClusterEDGWQJacobian:
    """Real-linear matrix-free Jacobian for one external momentum q."""

    def __init__(
        self,
        G: np.ndarray,
        W: np.ndarray,
        Vq: np.ndarray,
        G_cluster: np.ndarray,
        Sigma_imp: np.ndarray,
        bath: BathParameters,
        h_cluster: np.ndarray,
        interactions,
        mu: float,
        grid: MatsubaraGrid,
        q_index: tuple[int, int],
        opts: ClusterJFOptions,
        impurity_jf: ImpuritySelfEnergyJF | None = None,
    ):
        self.G = np.asarray(G, dtype=complex)
        self.W = np.asarray(W, dtype=complex)
        self.Vq = np.asarray(Vq, dtype=complex)
        self.Gc = np.asarray(G_cluster, dtype=complex)
        self.Sigma_imp = np.asarray(Sigma_imp, dtype=complex)
        self.grid = grid
        self.mu = float(mu)
        self.q = normalize_q_index(q_index, grid)
        self.opts = opts
        self.norb = int(self.G.shape[-1])
        self.eye = np.eye(self.norb, dtype=complex)
        self.invGc = np.linalg.inv(self.Gc)
        self.imp_scale = sqrt(float(max(grid.nk, 1)))
        self.mu_scale = sqrt(float(max(self.G.size / max(self.norb * self.norb, 1), 1.0)))
        self.matvecs = 0

        rho_c0 = np.zeros((self.norb, self.norb), dtype=complex)
        _, _, Wc = cluster_gw_self_energy(
            self.Gc,
            rho_c0,
            np.asarray(self.Vq[0, 0], dtype=complex),
            grid,
        )
        self.cgrid = MatsubaraGrid(
            nk1=1,
            nk2=1,
            nw=grid.nw,
            nOmega=grid.nOmega,
            T=grid.T,
        )
        self.Gc5 = self.Gc[:, None, None]
        self.Wc5 = np.asarray(Wc, dtype=complex)[:, None, None]
        self.Vc2 = np.asarray(self.Vq[0, 0], dtype=complex)[None, None]

        base_g0_inv = self.invGc + self.Sigma_imp
        if impurity_jf is None:
            self.impurity_jf = ImpuritySelfEnergyJF(
                grid,
                mu=self.mu,
                h_cluster=h_cluster,
                interactions=interactions,
                bath=bath,
                base_g0_inv=base_g0_inv,
                opts=opts,
            )
        else:
            self.impurity_jf = impurity_jf

        self.gw_vertex_opts = SupercellVertexOptions(
            max_iter=1,
            tol=1.0,
            solver="linear",
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=True,
            verbose=False,
            momentum_backend="fft",
        )

    @classmethod
    def from_embedding(
        cls,
        *,
        G: np.ndarray,
        W: np.ndarray,
        Vq: np.ndarray,
        G_cluster: np.ndarray,
        Sigma_imp: np.ndarray,
        bath: BathParameters,
        h_cluster: np.ndarray,
        interactions,
        mu: float,
        grid: MatsubaraGrid,
        q_index: tuple[int, int],
        opts: ClusterJFOptions,
        impurity_jf: ImpuritySelfEnergyJF | None = None,
    ) -> "ClusterEDGWQJacobian":
        return cls(
            G=G,
            W=W,
            Vq=Vq,
            G_cluster=G_cluster,
            Sigma_imp=Sigma_imp,
            bath=bath,
            h_cluster=h_cluster,
            interactions=interactions,
            mu=float(mu),
            grid=grid,
            q_index=q_index,
            opts=opts,
            impurity_jf=impurity_jf,
        )

    @property
    def gamma_shape(self) -> tuple[int, ...]:
        return self.G.shape

    @property
    def imp_shape(self) -> tuple[int, ...]:
        return self.Sigma_imp.shape

    @property
    def complex_size(self) -> int:
        return int(np.prod(self.gamma_shape) + np.prod(self.imp_shape) + 1)

    @property
    def real_size(self) -> int:
        return 2 * self.complex_size

    def _pack_complex(self, gamma: np.ndarray, simp: np.ndarray, dmu: complex) -> np.ndarray:
        return np.concatenate([
            np.asarray(gamma, dtype=complex).ravel(),
            self.imp_scale * np.asarray(simp, dtype=complex).ravel(),
            np.asarray([self.mu_scale * complex(dmu)], dtype=complex),
        ])

    def _unpack_complex(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, complex]:
        flat = np.asarray(z, dtype=complex).reshape(-1)
        ng = int(np.prod(self.gamma_shape))
        ni = int(np.prod(self.imp_shape))
        gamma = flat[:ng].reshape(self.gamma_shape)
        simp = (flat[ng:ng + ni] / self.imp_scale).reshape(self.imp_shape)
        dmu = complex(flat[ng + ni] / self.mu_scale)
        return gamma, simp, dmu

    def pack_real(self, gamma: np.ndarray, simp: np.ndarray, dmu: complex) -> np.ndarray:
        z = self._pack_complex(gamma, simp, dmu)
        return np.concatenate([z.real, z.imag]).astype(float, copy=False)

    def unpack_real(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, complex]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if len(arr) != self.real_size:
            raise ValueError("JF vector size mismatch")
        n = self.complex_size
        return self._unpack_complex(arr[:n] + 1j * arr[n:])

    def _x(self, gamma: np.ndarray) -> np.ndarray:
        Gp = roll_spatial(self.G, self.q[0], self.q[1])
        return np.einsum(
            "...ab,...bc,...cd->...ad", Gp, gamma, self.G, optimize=True
        )

    def _local_response(self, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._x(gamma)
        dGc = np.mean(X, axis=(1, 2))
        return X, dGc

    def _lattice_gw_tangent(self, gamma: np.ndarray) -> np.ndarray:
        parts = vertex_corrections_finite_q(
            self.G,
            self.W,
            self.Vq,
            gamma,
            self.q,
            self.grid,
            self.gw_vertex_opts,
        )
        return sum(parts)

    def _cluster_gw_tangent(self, dGc: np.ndarray) -> np.ndarray:
        local_gamma = np.einsum(
            "nab,nbc,ncd->nad", self.invGc, dGc, self.invGc, optimize=True
        )
        parts = vertex_corrections_q0(
            self.Gc5,
            self.Wc5,
            self.Vc2,
            local_gamma[:, None, None],
            self.cgrid,
            self.gw_vertex_opts,
        )
        return sum(parts)[:, 0, 0]

    def _density_tangent(self, X: np.ndarray) -> complex:
        diag = np.diagonal(X, axis1=-2, axis2=-1)
        return complex((self.grid.T / self.grid.nk) * np.sum(diag))

    def apply_complex(self, gamma: np.ndarray, simp: np.ndarray, dmu: complex):
        X, dGc = self._local_response(gamma)
        lgw = self._lattice_gw_tangent(gamma)
        lcgw = self._cluster_gw_tangent(dGc)
        dg0 = -np.einsum(
            "nab,nbc,ncd->nad", self.invGc, dGc, self.invGc, optimize=True
        ) + simp
        dimp = self.impurity_jf.direction(dg0)

        rgamma = gamma - lgw - dimp[:, None, None] + lcgw[:, None, None]
        q0 = (self.q[0] == 0 and self.q[1] == 0)
        if q0:
            rgamma = rgamma + complex(dmu) * self.eye[None, None, None, :, :]
            rmu = self._density_tangent(X)
        else:
            rmu = complex(dmu)
        rimp = simp - dimp
        return rgamma, rimp, rmu

    def matvec(self, x: np.ndarray) -> np.ndarray:
        self.matvecs += 1
        gamma, simp, dmu = self.unpack_real(x)
        rg, ri, rm = self.apply_complex(gamma, simp, dmu)
        return self.pack_real(rg, ri, rm)

    def linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.real_size, self.real_size),
            matvec=self.matvec,
            dtype=np.float64,
        )

    def preconditioner(self) -> LinearOperator | None:
        order = int(self.opts.preconditioner_order)
        if order <= 0:
            return None

        def mv(x):
            gamma, simp, dmu = self.unpack_real(x)
            rhs = np.asarray(gamma, dtype=complex)
            y = rhs.copy()
            cur = rhs.copy()
            for _ in range(order):
                cur = self._lattice_gw_tangent(cur)
                y += cur
            return self.pack_real(y, simp, dmu)

        return LinearOperator(
            (self.real_size, self.real_size),
            matvec=mv,
            dtype=np.float64,
        )

    def solve(
        self,
        K: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        recycle: list | None = None,
    ) -> tuple[ClusterJFQResult, np.ndarray, list]:
        K = np.asarray(K, dtype=complex)
        if K.shape != (self.norb, self.norb):
            raise ValueError("K shape mismatch")
        Kfield = np.broadcast_to(K, self.G.shape).copy()
        b = self.pack_real(Kfield, np.zeros_like(self.Sigma_imp), 0.0)
        if x0 is None or np.asarray(x0).shape != b.shape:
            x0 = b.copy()
        else:
            x0 = np.asarray(x0, dtype=float)
        CU = [] if recycle is None else recycle
        A = self.linear_operator()
        M = self.preconditioner()
        matvec0 = int(self.matvecs)
        t0 = perf_counter()
        x, info = gcrotmk(
            A,
            b,
            x0=x0,
            rtol=float(self.opts.rtol),
            atol=float(self.opts.atol),
            maxiter=int(self.opts.maxiter),
            M=M,
            m=int(self.opts.gcrot_m),
            k=int(self.opts.gcrot_k),
            CU=CU,
            discard_C=False,
            truncate="oldest",
        )
        elapsed = perf_counter() - t0
        gamma, simp, dmu = self.unpack_real(x)
        residual = b - A.matvec(x)
        err = float(np.max(np.abs(residual), initial=0.0))
        result = ClusterJFQResult(
            q_index=self.q,
            Gamma=np.asarray(gamma),
            Sigma_imp_tangent=np.asarray(simp),
            dmu=complex(dmu),
            converged=bool(info == 0 and np.isfinite(err)),
            info=int(info),
            matvecs=int(self.matvecs - matvec0),
            residual_max=err,
            elapsed=float(elapsed),
        )
        return result, np.asarray(x), CU


@dataclass
class PseudospinQScanPoint:
    q_index: tuple[int, int]
    chi: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    results: list[ClusterJFQResult] = field(default_factory=list)


def scan_pseudospin_q(
    *,
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    G_cluster: np.ndarray,
    Sigma_imp: np.ndarray,
    bath: BathParameters,
    h_cluster: np.ndarray,
    interactions,
    mu: float,
    grid: MatsubaraGrid,
    vertices: np.ndarray,
    q_indices: list[tuple[int, int]],
    opts: ClusterJFOptions,
) -> list[PseudospinQScanPoint]:
    """Solve a complete channel matrix at each q and diagonalize its Hermitian part."""
    K = np.asarray(vertices, dtype=complex)
    if K.ndim != 3 or K.shape[1:] != (G.shape[-1], G.shape[-1]):
        raise ValueError("vertices must have shape (nch,norb,norb)")
    previous_x: list[np.ndarray | None] = [None] * len(K)
    out: list[PseudospinQScanPoint] = []
    shared_impurity_jf: ImpuritySelfEnergyJF | None = None

    for q in q_indices:
        jac = ClusterEDGWQJacobian.from_embedding(
            G=G,
            W=W,
            Vq=Vq,
            G_cluster=G_cluster,
            Sigma_imp=Sigma_imp,
            bath=bath,
            h_cluster=h_cluster,
            interactions=interactions,
            mu=mu,
            grid=grid,
            q_index=q,
            opts=opts,
            impurity_jf=shared_impurity_jf,
        )
        if shared_impurity_jf is None:
            shared_impurity_jf = jac.impurity_jf
        if opts.verbose:
            bd = jac.impurity_jf.linearizer.diagnostics
            print(
                f"[cluster-JF] q={jac.q}: bath-rank={bd.rank}/{bd.n_parameters}, "
                f"bath-cond={bd.condition_number:.3e}",
                flush=True,
            )
        recycle: list = []
        gammas: list[np.ndarray] = []
        results: list[ClusterJFQResult] = []
        for a, Ka in enumerate(K):
            result, xsol, recycle = jac.solve(
                Ka,
                x0=previous_x[a],
                recycle=recycle,
            )
            previous_x[a] = xsol
            gammas.append(result.Gamma)
            results.append(result)
            if opts.verbose:
                print(
                    f"[cluster-JF] q={jac.q}, channel={a}: info={result.info}, "
                    f"residual={result.residual_max:.3e}, matvecs={result.matvecs}, "
                    f"dt={result.elapsed:.1f}s",
                    flush=True,
                )

        chi = susceptibility_matrix_finite_q(G, K, gammas, jac.q, grid)
        ch = 0.5 * (chi + chi.conj().T)
        vals, vecs = np.linalg.eigh(ch)
        order = np.argsort(vals.real)[::-1]
        out.append(
            PseudospinQScanPoint(
                q_index=jac.q,
                chi=np.asarray(chi),
                eigenvalues=np.asarray(vals[order].real),
                eigenvectors=np.asarray(vecs[:, order]),
                results=results,
            )
        )
    return out


__all__ = [
    "ClusterJFOptions",
    "BathLinearizationDiagnostics",
    "LinearizedBathFit",
    "ImpurityJFDiagnostics",
    "ImpuritySelfEnergyJF",
    "ClusterJFQResult",
    "ClusterEDGWQJacobian",
    "PseudospinQScanPoint",
    "scan_pseudospin_q",
]
