"""Pulay-accelerated 6-site cluster-ED + lattice-GW self-energy embedding.

This module keeps the physics of :mod:`rubycgw.cluster_ed_gw` but accelerates
its coupled fixed point.  The lattice embedded self-energy and the impurity
self-energy are mixed together, rather than damping the impurity map first and
then damping the lattice map a second time.

The dynamic state is expressed in nonredundant weak/local pieces,

    Sigma_weak(k,iw) = Sigma_emb(k,iw) - Sigma_imp(iw),
    X_dyn = (Sigma_weak(k,iw), sqrt(Nk) Sigma_imp(iw)).

Since Sigma_emb = Sigma_GW^lat - Sigma_GW^cluster + Sigma_imp, mixing Sigma_emb
and Sigma_imp directly duplicates the local impurity component in the Pulay
metric and becomes badly conditioned near soft density modes.  The weak/local
split removes that exact algebraic redundancy.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    build_intracell_h0,
    cluster_gw_self_energy,
    cluster_interaction_matrix,
    fit_finite_bath,
    split_static_hybridization,
    ruby_cluster_interactions,
)
from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _mixed_self_energies,
)
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .pseudospin import primitive_cell_pseudospin_channels
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast, solve_matrix_gw_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_matrix,
    one_body_density_matrix_tail,
)


@dataclass(frozen=True)
class ClusterEDGWFastOptions:
    max_iter: int = 30
    tol: float = 2.0e-5
    mixing: float = 0.40
    mixing_method: str = "pulay"  # "linear", "pulay", or "broyden"
    pulay_history: int = 6
    pulay_start: int = 3
    pulay_regularization: float = 1.0e-7
    pulay_step_cap: float = 3.0
    broyden_history: int = 8
    broyden_regularization: float = 1.0e-8
    broyden_step_cap: float = 3.0
    broyden_reset_growth: float = 1.5
    # Optional pre-damping of the raw impurity map.  The default 1 means that
    # all damping/acceleration is handled by the coupled outer mixer.
    impurity_mixing: float = 1.0
    nbath: int = 6
    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 300
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_fit_xtol: float = 1.0e-9
    bath_fit_metric: str = "delta"  # "delta" or "g0"
    discard_weight_tol: float = 1.0e-11
    verbose: bool = True


@dataclass
class ClusterEDGWFastResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_GW_lattice: np.ndarray
    Sigma_GW_cluster: np.ndarray
    Sigma_ED_cluster: np.ndarray
    G_cluster: np.ndarray
    G_impurity: np.ndarray
    mu: float
    density: np.ndarray
    bath: BathParameters
    converged: bool
    iterations: int
    final_error: float
    impurity_mismatch: float
    bath_fit_error: float
    impurity_static_shift: np.ndarray
    background: GWResult
    mixing_method: str
    residual_history: np.ndarray
    impurity_residual_history: np.ndarray
    impurity_mismatch_history: np.ndarray
    bath_fit_history: np.ndarray
    mu_history: np.ndarray
    bath_nfev_history: np.ndarray
    elapsed_history: np.ndarray
    pulay_fallbacks: int


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))



def _channel_projection(mat: np.ndarray, vertex: np.ndarray) -> complex:
    m = np.asarray(mat, dtype=complex)
    v = np.asarray(vertex, dtype=complex)
    den = np.vdot(v, v)
    if abs(den) < 1e-300:
        return 0.0j
    return np.vdot(v, m) / den

def _check_embed_mixing_method(method: str) -> str:
    key = str(method).lower()
    if key not in {"linear", "pulay", "broyden"}:
        raise ValueError("embedding mixing_method must be 'linear', 'pulay', or 'broyden'")
    return key


def _pack_broyden_state(
    sigma_h: np.ndarray,
    dyn: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pack complex Hartree/dynamic variables into a balanced real vector."""
    h = np.asarray(sigma_h, dtype=complex)
    d = np.asarray(dyn, dtype=complex).reshape(-1)
    hscale = np.sqrt(float(max(d.size, 1)) / float(max(h.size, 1)))
    z = np.concatenate([hscale * h.ravel(), d])
    return np.concatenate([z.real, z.imag]), float(hscale)


def _unpack_broyden_state(
    packed: np.ndarray,
    sigma_h_shape: tuple[int, ...],
    dyn_shape: tuple[int, ...],
    hscale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`_pack_broyden_state`."""
    x = np.asarray(packed, dtype=float).reshape(-1)
    nh = int(np.prod(sigma_h_shape))
    nd = int(np.prod(dyn_shape))
    nz = nh + nd
    if x.size != 2 * nz:
        raise ValueError("Broyden packed-state size mismatch")
    z = x[:nz] + 1j * x[nz:]
    h = (z[:nh] / float(hscale)).reshape(sigma_h_shape)
    d = z[nh:].reshape(dyn_shape)
    return h, d


class _LimitedMemoryBroyden:
    """Limited-memory multisecant inverse Broyden mixer.

    We solve q(x)=x-F(x)=0.  The initial inverse Jacobian is alpha*I, so with
    no secant history the update is the ordinary linear-mixing step

        x_next = x - alpha*q = x + alpha*(F(x)-x).

    The retained secant pairs satisfy H y_i ~= s_i with
    s_i=x_i-x_{i-1}, y_i=q_i-q_{i-1}.  The least-change multisecant inverse is

        H = alpha I + (S-alpha Y) (Y^T Y + reg I)^-1 Y^T.

    All vectors are explicitly real so the nonlinear complex self-energy map is
    treated as a real-linear problem rather than assuming holomorphicity.
    """

    def __init__(
        self,
        *,
        alpha: float,
        history: int,
        regularization: float,
    ):
        self.alpha = float(alpha)
        self.history = int(history)
        self.regularization = float(regularization)
        self.s_hist: list[np.ndarray] = []
        self.y_hist: list[np.ndarray] = []
        self.prev_x: np.ndarray | None = None
        self.prev_q: np.ndarray | None = None

    def clear(self) -> None:
        self.s_hist.clear()
        self.y_hist.clear()
        self.prev_x = None
        self.prev_q = None

    @property
    def rank(self) -> int:
        return len(self.s_hist)

    def _append_secant(self, s: np.ndarray, y: np.ndarray) -> None:
        ynorm = float(np.linalg.norm(y))
        snorm = float(np.linalg.norm(s))
        if (
            not np.isfinite(ynorm)
            or not np.isfinite(snorm)
            or ynorm <= 1.0e-14 * max(1.0, snorm)
        ):
            return
        self.s_hist.append(np.asarray(s / ynorm, dtype=float).copy())
        self.y_hist.append(np.asarray(y / ynorm, dtype=float).copy())
        keep = max(int(self.history), 1)
        if len(self.s_hist) > keep:
            del self.s_hist[:-keep]
            del self.y_hist[:-keep]

    def inverse_action(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if not self.s_hist:
            return self.alpha * q
        S = np.column_stack(self.s_hist)
        Y = np.column_stack(self.y_hist)
        gram = Y.T @ Y
        scale = max(
            float(np.max(np.abs(gram), initial=0.0)),
            np.finfo(float).tiny,
        )
        gram_scaled = gram / scale
        rhs = (Y.T @ q) / scale
        gram_scaled = gram_scaled + self.regularization * np.eye(gram.shape[0])
        try:
            coeff = np.linalg.solve(gram_scaled, rhs)
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(gram_scaled, rhs, rcond=None)[0]
        return self.alpha * q + (S - self.alpha * Y) @ coeff

    def propose(self, x: np.ndarray, q: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        q = np.asarray(q, dtype=float)
        if self.prev_x is not None and self.prev_q is not None:
            self._append_secant(x - self.prev_x, q - self.prev_q)
        step = self.inverse_action(q)
        self.prev_x = x.copy()
        self.prev_q = q.copy()
        return x - step


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _pack_dynamic(
    sigma_emb: np.ndarray,
    sigma_imp: np.ndarray,
    nk: int,
) -> np.ndarray:
    """Pack nonredundant weak-lattice and impurity dynamic self-energies."""
    emb = np.asarray(sigma_emb, dtype=complex)
    imp = np.asarray(sigma_imp, dtype=complex)
    if emb.ndim != 5 or imp.ndim != 3:
        raise ValueError("dynamic self-energy shapes must be lattice 5D and impurity 3D")
    weak = emb - imp[:, None, None, :, :]
    scale = np.sqrt(float(max(int(nk), 1)))
    return np.concatenate([
        weak.ravel(),
        scale * imp.ravel(),
    ])


def _unpack_dynamic(
    packed: np.ndarray,
    sigma_emb_shape: tuple[int, ...],
    sigma_imp_shape: tuple[int, ...],
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    nemb = int(np.prod(sigma_emb_shape))
    scale = np.sqrt(float(max(int(nk), 1)))
    flat = np.asarray(packed, dtype=complex).reshape(-1)
    weak = flat[:nemb].reshape(sigma_emb_shape)
    imp = (flat[nemb:] / scale).reshape(sigma_imp_shape)
    emb = weak + imp[:, None, None, :, :]
    return emb, imp


def solve_cluster_ed_gw_fast(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWFastOptions = ClusterEDGWFastOptions(),
    background: GWResult | None = None,
) -> ClusterEDGWFastResult:
    """Solve the coupled lattice-GW / finite-bath ED fixed point with Pulay."""
    backend = _check_backend(gw_opts.momentum_backend)
    method = _check_embed_mixing_method(embed_opts.mixing_method)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB):
        raise ValueError("cluster ED+GW expects the primitive 6-site lattice basis")
    if Vq.shape != h0.shape:
        raise ValueError("Vq/h0 shape mismatch")
    if not (0.0 < float(embed_opts.mixing) <= 1.0):
        raise ValueError("embedding mixing must lie in (0,1]")
    if not (0.0 < float(embed_opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity mixing must lie in (0,1]")
    if int(embed_opts.pulay_history) < 2:
        raise ValueError("pulay_history must be at least 2")
    if int(embed_opts.broyden_history) < 1:
        raise ValueError("broyden_history must be at least 1")
    if float(embed_opts.broyden_regularization) < 0.0:
        raise ValueError("broyden_regularization must be nonnegative")
    if float(embed_opts.broyden_step_cap) <= 0.0:
        raise ValueError("broyden_step_cap must be positive")
    if float(embed_opts.broyden_reset_growth) <= 1.0:
        raise ValueError("broyden_reset_growth must be > 1")

    if background is None:
        if embed_opts.verbose:
            print("[cluster-ED+GW] solving initial lattice GW background ...", flush=True)
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    elif embed_opts.verbose:
        print(
            f"[cluster-ED+GW] using cached lattice GW background: "
            f"residual={background.final_error:.3e}",
            flush=True,
        )
    if not background.converged:
        raise RuntimeError(
            f"initial lattice GW background is not converged: {background.final_error:.3e}"
        )

    h_cluster_strict = build_intracell_h0(params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    alias_norm = _maxabs(h_cluster - h_cluster_strict)
    if embed_opts.verbose and alias_norm > 1.0e-12:
        print(
            f"[cluster-ED+GW] finite-torus local-block correction: "
            f"max|h_local-h_R0|={alias_norm:.3e}",
            flush=True,
        )

    interactions = ruby_cluster_interactions(params)
    V_cluster = cluster_interaction_matrix(interactions, NSUB)

    sigma_h = np.asarray(background.Sigma_H, dtype=complex).copy()
    sigma_emb = np.asarray(background.Sigma_GW, dtype=complex).copy()
    mu = float(background.mu)
    G = np.asarray(background.G, dtype=complex).copy()
    bath: BathParameters | None = None

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    Gc = np.mean(G, axis=(1, 2))
    rho_c = np.mean(rho_k, axis=(0, 1))
    sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)
    sigma_imp = np.array(sigma_cgw, copy=True)

    mix_opts = GWOptions(
        mixing=float(embed_opts.mixing),
        mixing_method=("pulay" if method == "broyden" else method),
        pulay_history=int(embed_opts.pulay_history),
        pulay_start=int(embed_opts.pulay_start),
        pulay_regularization=float(embed_opts.pulay_regularization),
    )
    mix_history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    broyden = (
        _LimitedMemoryBroyden(
            alpha=float(embed_opts.mixing),
            history=int(embed_opts.broyden_history),
            regularization=float(embed_opts.broyden_regularization),
        )
        if method == "broyden"
        else None
    )
    fallbacks = 0

    residual_hist: list[float] = []
    imp_residual_hist: list[float] = []
    mismatch_hist: list[float] = []
    bath_hist: list[float] = []
    mu_hist: list[float] = []
    nfev_hist: list[int] = []
    elapsed_hist: list[float] = []

    W = np.asarray(background.W)
    P = np.asarray(background.P)
    sigma_gw_lattice = np.asarray(background.Sigma_GW)
    Gimp = np.asarray(Gc)
    density = np.asarray(background.density)
    converged = False
    err = float("inf")
    mismatch = float("inf")
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: build lattice GW map "
                f"(mix={method})",
                flush=True,
            )

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )

        Gc = np.mean(G, axis=(1, 2))
        sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)

        g0_inv = np.linalg.inv(Gc) + sigma_imp
        eye = np.eye(NSUB, dtype=complex)
        delta_target_raw = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - g0_inv
        )
        static_shift, delta_target = split_static_hybridization(
            delta_target_raw, grid.omega
        )
        h_impurity = h_cluster + static_shift

        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: fit {embed_opts.nbath} bath orbitals ...",
                flush=True,
            )
        bath = fit_finite_bath(
            delta_target,
            grid.omega,
            mu,
            nbath=int(embed_opts.nbath),
            nfit=int(embed_opts.bath_fit_nfreq),
            max_nfev=int(embed_opts.bath_fit_max_nfev),
            energy_window=float(embed_opts.bath_energy_window),
            coupling_bound=float(embed_opts.bath_coupling_bound),
            xtol=float(embed_opts.bath_fit_xtol),
            initial=bath,
            metric=str(embed_opts.bath_fit_metric),
            one_body=h_impurity,
        )
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: bath({embed_opts.bath_fit_metric}) "
                f"relerr={bath.fit_error:.3e}, static={_maxabs(static_shift):.3e}, "
                f"nfev={bath.nfev}; diagonalize impurity ...",
                flush=True,
            )

        himp = build_impurity_one_body(h_impurity, bath)
        impurity = FiniteBathImpurityED(
            himp,
            interactions,
            correlated_orbitals=tuple(range(NSUB)),
        )
        impurity.diagonalize()
        Gimp, selection = impurity.green_iomega(
            1j * grid.omega,
            mu,
            grid.T,
            orbitals=tuple(range(NSUB)),
            discard_weight_tol=float(embed_opts.discard_weight_tol),
        )
        delta_fit = bath_hybridization(
            grid.omega, mu, bath.energies, bath.couplings
        )
        g0_fit_inv = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_impurity[None, :, :]
            - delta_fit
        )
        sigma_ed_raw = g0_fit_inv - np.linalg.inv(Gimp)

        beta_imp = float(embed_opts.impurity_mixing)
        sigma_imp_out = sigma_imp + beta_imp * (sigma_ed_raw - sigma_imp)
        correction = sigma_imp_out - sigma_cgw
        sigma_emb_out = sigma_gw_lattice + correction[:, None, None, :, :]

        res_h = _maxabs(sigma_h_out - sigma_h)
        res_emb = _maxabs(sigma_emb_out - sigma_emb)
        imp_residual = sigma_imp_out - sigma_imp
        res_imp = _maxabs(imp_residual)
        err = max(res_h, res_emb, res_imp)

        iw_low = int(np.argmin(np.abs(grid.omega)))
        iw_high = int(np.argmax(np.abs(grid.omega)))
        imp_low = _maxabs(imp_residual[iw_low])
        imp_high = _maxabs(imp_residual[iw_high])
        hi_common = abs(np.trace(imp_residual[iw_high]) / float(NSUB))
        low_common = abs(np.trace(imp_residual[iw_low]) / float(NSUB))
        low_mat = np.asarray(imp_residual[iw_low], dtype=complex)
        ps = primitive_cell_pseudospin_channels()
        low_proj = {
            name: abs(_channel_projection(low_mat, ps[name]))
            for name in ("x_even", "x_odd", "y_even", "y_odd", "z_even", "z_odd")
        }
        max_idx = np.unravel_index(
            int(np.argmax(np.abs(imp_residual))), imp_residual.shape
        )
        max_iw, max_a, max_b = (int(max_idx[0]), int(max_idx[1]), int(max_idx[2]))
        mismatch = _relative_error(Gimp, Gc)
        elapsed = perf_counter() - t0

        residual_hist.append(float(err))
        imp_residual_hist.append(float(res_imp))
        mismatch_hist.append(float(mismatch))
        bath_hist.append(float(bath.fit_error))
        mu_hist.append(float(mu))
        nfev_hist.append(int(bath.nfev))
        elapsed_hist.append(float(elapsed))

        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), "
                f"Gimp/Gc={mismatch:.3e}, bath={bath.fit_error:.3e}, "
                f"Ntot_imp={selection.average_particles:.6f}, "
                f"mu={mu:+.9f}, dt={elapsed:.1f}s\n"
                f"    imp-res: low={imp_low:.3e}, high={imp_high:.3e}, "
                f"low-common={low_common:.3e}, high-common={hi_common:.3e}, "
                f"max@iw[{max_iw}]={grid.omega[max_iw]:+.3e},ab=({max_a},{max_b})\n"
                f"    low-iw channels: xE={low_proj['x_even']:.3e}, "
                f"xO={low_proj['x_odd']:.3e}, yE={low_proj['y_even']:.3e}, "
                f"yO={low_proj['y_odd']:.3e}, zE={low_proj['z_even']:.3e}, "
                f"zO={low_proj['z_odd']:.3e}",
                flush=True,
            )

        if err < float(embed_opts.tol):
            sigma_h = np.asarray(sigma_h_out)
            sigma_emb = np.asarray(sigma_emb_out)
            sigma_imp = np.asarray(sigma_imp_out)
            converged = True
        else:
            dyn = _pack_dynamic(sigma_emb, sigma_imp, grid.nk)
            dyn_out = _pack_dynamic(sigma_emb_out, sigma_imp_out, grid.nk)

            if method == "broyden":
                if broyden is None:
                    raise RuntimeError("internal Broyden mixer was not initialized")
                if (
                    len(residual_hist) >= 2
                    and residual_hist[-1]
                    > float(embed_opts.broyden_reset_growth) * residual_hist[-2]
                ):
                    broyden.clear()
                    if embed_opts.verbose:
                        print(
                            f"[cluster-ED+GW] outer {it:02d}: Broyden history reset "
                            f"(residual growth={residual_hist[-1]/max(residual_hist[-2],1e-300):.2f})",
                            flush=True,
                        )
                x, hscale = _pack_broyden_state(sigma_h, dyn)
                xout, hscale_out = _pack_broyden_state(sigma_h_out, dyn_out)
                if not np.isclose(hscale, hscale_out, rtol=0.0, atol=0.0):
                    raise RuntimeError("Broyden state scaling changed unexpectedly")
                q = x - xout
                xnext = broyden.propose(x, q)
                sigma_h_next, dyn_next = _unpack_broyden_state(
                    xnext, sigma_h.shape, dyn.shape, hscale
                )
                cap = float(embed_opts.broyden_step_cap)
            else:
                sigma_h_next, dyn_next = _mixed_self_energies(
                    sigma_h,
                    dyn,
                    sigma_h_out,
                    dyn_out,
                    mix_opts,
                    it,
                    mix_history,
                )
                cap = float(embed_opts.pulay_step_cap)

            raw_step = max(_maxabs(sigma_h_out - sigma_h), _maxabs(dyn_out - dyn))
            mixed_step = max(
                _maxabs(sigma_h_next - sigma_h),
                _maxabs(dyn_next - dyn),
            )
            unsafe = (
                not np.all(np.isfinite(sigma_h_next))
                or not np.all(np.isfinite(dyn_next))
                or (raw_step > 1e-14 and mixed_step > cap * raw_step)
            )
            if unsafe:
                fallbacks += 1
                mix_history.clear()
                if broyden is not None:
                    # Keep the current point as the reference so the next
                    # accepted linear step immediately supplies a fresh secant.
                    xcur, _ = _pack_broyden_state(sigma_h, dyn)
                    xoutcur, _ = _pack_broyden_state(sigma_h_out, dyn_out)
                    broyden.clear()
                    broyden.prev_x = np.asarray(xcur, dtype=float).copy()
                    broyden.prev_q = np.asarray(xcur - xoutcur, dtype=float).copy()
                a = float(embed_opts.mixing)
                sigma_h_next = sigma_h + a * (sigma_h_out - sigma_h)
                dyn_next = dyn + a * (dyn_out - dyn)
                if embed_opts.verbose:
                    label = "Broyden" if method == "broyden" else "Pulay"
                    print(
                        f"[cluster-ED+GW] outer {it:02d}: {label} safeguard -> "
                        f"linear fallback (mixed/raw={mixed_step/max(raw_step,1e-300):.2f})",
                        flush=True,
                    )

            sigma_h = np.asarray(sigma_h_next)
            sigma_emb, sigma_imp = _unpack_dynamic(
                dyn_next,
                sigma_emb.shape,
                sigma_imp.shape,
                grid.nk,
            )

        # Always update G so a converged return corresponds to the accepted
        # self-energy, and so the last non-converged iteration is not discarded.
        if gw_opts.target_filling is None:
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
        else:
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_emb,
                grid,
                float(gw_opts.target_filling),
                mu,
                float(gw_opts.mu_tol),
                int(gw_opts.mu_max_iter),
            )
        if converged:
            break

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.real(np.diag(np.mean(rho_k, axis=(0, 1))))
    Gc = np.mean(G, axis=(1, 2))

    if bath is None:
        raise RuntimeError("embedding loop did not execute")

    return ClusterEDGWFastResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_GW_lattice=np.asarray(sigma_gw_lattice),
        Sigma_GW_cluster=np.asarray(sigma_cgw),
        Sigma_ED_cluster=np.asarray(sigma_imp),
        G_cluster=np.asarray(Gc),
        G_impurity=np.asarray(Gimp),
        mu=float(mu),
        density=np.asarray(density),
        bath=bath,
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch=float(mismatch),
        bath_fit_error=float(bath.fit_error),
        impurity_static_shift=np.asarray(static_shift),
        background=background,
        mixing_method=method,
        residual_history=np.asarray(residual_hist, dtype=float),
        impurity_residual_history=np.asarray(imp_residual_hist, dtype=float),
        impurity_mismatch_history=np.asarray(mismatch_hist, dtype=float),
        bath_fit_history=np.asarray(bath_hist, dtype=float),
        mu_history=np.asarray(mu_hist, dtype=float),
        bath_nfev_history=np.asarray(nfev_hist, dtype=int),
        elapsed_history=np.asarray(elapsed_hist, dtype=float),
        pulay_fallbacks=int(fallbacks),
    )


__all__ = [
    "ClusterEDGWFastOptions",
    "ClusterEDGWFastResult",
    "solve_cluster_ed_gw_fast",
]
