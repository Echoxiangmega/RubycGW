"""Generic post-GW screened-interaction feedback for RubycGW.

Post-GW reconnects the screened interaction to the *physical reducible*
covariant density response.  With the signs used in this repository,

    P = + G G,
    W_GW = V + V P W_GW,
    chi_nn = delta n / delta phi = - G Gamma G,

so the exact screened-interaction identity is

    W_post(Q) = V(q) - V(q) chi_nn,cov(Q) V(q).

The response supplied here must therefore be the full orbital density matrix
chi_nn(Q), not the projected x/z pseudospin susceptibilities and not an
irreducible polarization P.

The post Green function is a one-shot construction: Sigma_post is evaluated
with the *background* G and W_post, then Dyson's equation is solved once.  The
background Hartree potential is kept fixed, matching the Hartree propagator of
the truncated theory.  For the split RubycGW self-energy this means

    Sigma_post = Sigma_H[bg]
               + Sigma_F[G_bg,V]
               + Sigma_c[G_bg,W_post-V]
               + Sigma_extra[G_bg],

where Sigma_extra is zero for GW and is the bare-SOX skeleton for GW+SOX.
Thus the same post engine applies to either background; only the covariant
response provider and optional extra skeleton change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariant_density import (
    CovariantDensityResult,
    compute_covariant_density_susceptibility,
)
from .grids import MatsubaraGrid
from .gw import GWOptions, _check_backend
from .sox_covariant import SOXOptions
from .supercell_cgw import SupercellVertexOptions
from .supercell_gw import dyson_from_sigma_matrix
from .supercell_gw_fast import (
    _build_tail_cache,
    _solve_mu_matrix_fast,
    density_from_G_cached,
)
from .supercell_gw_split import compute_sigma_gw_split_components


@dataclass
class PostGWResult:
    G: np.ndarray
    W_post: np.ndarray
    chi_cov: np.ndarray
    chi_raw: np.ndarray
    Sigma_H: np.ndarray
    Sigma_GW_post: np.ndarray
    Sigma_F: np.ndarray
    Sigma_c_post: np.ndarray
    Sigma_extra: np.ndarray
    Sigma_corr_post: np.ndarray
    mu: float
    density: np.ndarray
    background_mu: float
    background_density: np.ndarray
    include_sox: bool
    density_response: CovariantDensityResult
    used_background_w_mask: np.ndarray

    @property
    def Sigma_total(self) -> np.ndarray:
        return self.Sigma_H[None, None, None, :, :] + self.Sigma_corr_post


def build_post_screened_interaction(
    Vq: np.ndarray,
    chi_cov: np.ndarray,
    *,
    background_W: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return W_post=V-V chi_cov V and a fallback mask.

    NaN response entries are allowed only when ``background_W`` is supplied;
    this is used by the explicit diagnostic ``m_max`` window, where uncomputed
    high-frequency transfers retain the original background W.  A full
    post-GW calculation has no fallback entries.
    """
    Vq = np.asarray(Vq, dtype=complex)
    chi = np.asarray(chi_cov, dtype=complex)
    if chi.ndim != 5:
        raise ValueError("chi_cov must have shape (nb,nk1,nk2,norb,norb)")
    if Vq.shape != chi.shape[1:]:
        raise ValueError("Vq/chi_cov shape mismatch")
    if background_W is not None:
        bg = np.asarray(background_W, dtype=complex)
        if bg.shape != chi.shape:
            raise ValueError("background_W shape mismatch")
    else:
        bg = None

    out = np.empty_like(chi)
    fallback = np.zeros(chi.shape[:3], dtype=bool)
    for im in range(chi.shape[0]):
        for iq1 in range(chi.shape[1]):
            for iq2 in range(chi.shape[2]):
                c = chi[im, iq1, iq2]
                if not np.all(np.isfinite(c)):
                    if bg is None:
                        raise ValueError(
                            "chi_cov contains uncomputed/non-finite transfers; "
                            "supply background_W only for an explicit windowed diagnostic"
                        )
                    out[im, iq1, iq2] = bg[im, iq1, iq2]
                    fallback[im, iq1, iq2] = True
                    continue
                v = Vq[iq1, iq2]
                out[im, iq1, iq2] = v - v @ c @ v
    return out, fallback


def one_shot_post_dyson(
    G_background: np.ndarray,
    W_post: np.ndarray,
    Vq: np.ndarray,
    h0: np.ndarray,
    sigma_h_background: np.ndarray,
    mu_background: float,
    grid: MatsubaraGrid,
    *,
    target_filling: float | None,
    backend: str = "fft",
    mu_tol: float = 1.0e-10,
    mu_max_iter: int = 100,
    sigma_extra: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate Sigma_post on G_background and solve the fixed-Sigma Dyson step."""
    backend = _check_backend(backend)
    Gbg = np.asarray(G_background, dtype=complex)
    Wpost = np.asarray(W_post, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    sigma_h = np.asarray(sigma_h_background, dtype=complex)
    sigma_gw, sigma_f, sigma_c, _ = compute_sigma_gw_split_components(
        Gbg,
        Wpost,
        Vq,
        grid,
        h0,
        float(mu_background),
        sigma_h,
        backend=backend,
    )
    if sigma_extra is None:
        extra = np.zeros_like(sigma_gw)
    else:
        extra = np.asarray(sigma_extra, dtype=complex)
        if extra.shape != sigma_gw.shape:
            raise ValueError("sigma_extra shape mismatch")
    sigma_corr = sigma_gw + extra

    if target_filling is None:
        mu = float(mu_background)
        Gpost = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_corr)
        cache = _build_tail_cache(h0, sigma_h)
    else:
        mu, Gpost, cache, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_corr,
            grid,
            float(target_filling),
            float(mu_background),
            float(mu_tol),
            int(mu_max_iter),
        )
    density = density_from_G_cached(Gpost, grid, float(mu), cache)
    return {
        "G": np.asarray(Gpost),
        "Sigma_GW": np.asarray(sigma_gw),
        "Sigma_F": np.asarray(sigma_f),
        "Sigma_c": np.asarray(sigma_c),
        "Sigma_extra": np.asarray(extra),
        "Sigma_corr": np.asarray(sigma_corr),
        "mu": float(mu),
        "density": np.asarray(density),
    }


def run_post_gw(
    background,
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions,
    vertex_opts: SupercellVertexOptions = SupercellVertexOptions(),
    include_sox: bool = False,
    sox_opts: SOXOptions = SOXOptions(),
    m_max: int | None = None,
    allow_unconverged: bool = False,
) -> PostGWResult:
    """Common post-GW driver for either a GW or a GW+SOX background."""
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    Gbg = np.asarray(background.G, dtype=complex)
    Wbg = np.asarray(background.W, dtype=complex)
    sigma_h = np.asarray(background.Sigma_H, dtype=complex)

    # Resolve the static Fock piece used by both the observable tail and the
    # optional SOX transfer kernel.  GWSOXResult already stores it; ordinary
    # GWResult does not, so reconstruct the same production split once.
    if hasattr(background, "Sigma_F"):
        sigma_f_bg = np.asarray(background.Sigma_F, dtype=complex)
    else:
        _, sigma_f_bg, _, _ = compute_sigma_gw_split_components(
            Gbg,
            Wbg,
            Vq,
            grid,
            h0,
            float(background.mu),
            sigma_h,
            backend=gw_opts.momentum_backend,
        )

    if include_sox:
        if not hasattr(background, "Sigma_SOX"):
            raise ValueError(
                "include_sox=True requires a GW+SOX background with Sigma_SOX"
            )
        sigma_extra = np.asarray(background.Sigma_SOX, dtype=complex)
    else:
        sigma_extra = np.zeros_like(Gbg)

    dens = compute_covariant_density_susceptibility(
        Gbg,
        Wbg,
        Vq,
        h0,
        float(background.mu),
        sigma_h,
        sigma_f_bg,
        grid,
        vertex_opts=vertex_opts,
        include_sox=bool(include_sox),
        sox_opts=sox_opts,
        m_max=m_max,
        allow_unconverged=allow_unconverged,
    )
    Wpost, fallback = build_post_screened_interaction(
        Vq,
        dens.chi_completed,
        background_W=Wbg if m_max is not None else None,
    )
    step = one_shot_post_dyson(
        Gbg,
        Wpost,
        Vq,
        h0,
        sigma_h,
        float(background.mu),
        grid,
        target_filling=gw_opts.target_filling,
        backend=gw_opts.momentum_backend,
        mu_tol=gw_opts.mu_tol,
        mu_max_iter=gw_opts.mu_max_iter,
        sigma_extra=sigma_extra,
    )
    return PostGWResult(
        G=np.asarray(step["G"]),
        W_post=np.asarray(Wpost),
        chi_cov=np.asarray(dens.chi_completed),
        chi_raw=np.asarray(dens.chi_raw),
        Sigma_H=np.asarray(sigma_h),
        Sigma_GW_post=np.asarray(step["Sigma_GW"]),
        Sigma_F=np.asarray(step["Sigma_F"]),
        Sigma_c_post=np.asarray(step["Sigma_c"]),
        Sigma_extra=np.asarray(step["Sigma_extra"]),
        Sigma_corr_post=np.asarray(step["Sigma_corr"]),
        mu=float(step["mu"]),
        density=np.asarray(step["density"]),
        background_mu=float(background.mu),
        background_density=np.asarray(background.density),
        include_sox=bool(include_sox),
        density_response=dens,
        used_background_w_mask=np.asarray(fallback),
    )


__all__ = [
    "PostGWResult",
    "build_post_screened_interaction",
    "one_shot_post_dyson",
    "run_post_gw",
]
