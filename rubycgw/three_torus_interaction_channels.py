"""Matched finite-torus exact density, A1+E pseudospin and B/J Fierz variants.

These are different *interaction decoupling schemes* applied to the same
physical spinless Ruby Hamiltonian, not merely three observable form factors.

- density: physical site n_i screening, exact density HS representation.
- pseudospin: EXACT A1+E triangle identity, E-doublet (x,y,z) with
  g_E=-V/3 plus residual density coupling g_(n_A1,n_E)=V.
  The operator-square identity also contains +(V/2)n_E. The numerical
  static map is calibrated to the exact physical original-V Hartree-Fock
  expression to avoid counting that normal-ordering shift twice.
- bj: every intra-triangle bond has real B and imaginary J vertices,
  with both bosonic couplings g_B=g_J=-V/2 (symmetric exact Fierz choice).

All compare the SAME physical external observables O_ch(Q).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from time import perf_counter

import numpy as np

from .fierz_channel_gw import (
    ChannelDefinition,
    ChannelGWResult,
    build_channel_definition,
    channel_static_parts,
    channel_static_self_energy,
    compute_channel_correlation_self_energy,
    solve_channel_gw_same_torus,
)
from .fierz_channel_gamma import (
    compute_covariant_channel_susceptibility,
    build_channel_gamma_screening,
    solve_channel_vertex_transfer,
)
from .gw import GWOptions
from .grids import MatsubaraGrid
from .supercell_cgw import SupercellVertexOptions
from .supercell_gw_fast import _solve_mu_matrix_fast
from .supercell_gw_split import one_body_density_matrix_tail
from .production_cgw_sox import (
    static_gg_tail_completed, static_response_tail_completed,
)
from .response_tail import build_tail_reference


def triangle_v_pairs(ncell: int) -> tuple[tuple[int, int], ...]:
    """Undirected physical intra-triangle V bonds, six per primitive cell."""
    edges = ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5))
    return tuple((6*c+i, 6*c+j) for c in range(int(ncell)) for i,j in edges)


def build_torus_interaction_channel(geometry, V: float, mode: str) -> ChannelDefinition:
    pairs=triangle_v_pairs(geometry.ncell)
    return build_channel_definition(pairs,geometry.nsites,float(V),mode)


def channel_energy_estimate(state: ChannelGWResult | object, h0, grid: MatsubaraGrid):
    """Finite-temperature Galitskii-Migdal estimator, not ground-state E0."""
    G=np.asarray(state.G,dtype=complex)
    static=np.asarray(state.Sigma_static,dtype=complex)
    sigc=np.asarray(state.Sigma_c,dtype=complex)
    rho=one_body_density_matrix_tail(G,grid,h0,float(state.mu),static)
    kinetic=float(np.einsum("xyab,xyba->",h0,rho,optimize=True).real/grid.nk)
    first=.5*float(np.einsum("ab,xyba->",static,rho,optimize=True).real/grid.nk)
    dynamic=.5*float((grid.T/grid.nk)*np.einsum(
        "nxyab,nxyba->",sigc,G,optimize=True).real)
    return float(kinetic+first+dynamic)


def solve_interaction_channel_response(
    background: ChannelGWResult, definition: ChannelDefinition,
    geometry, grid: MatsubaraGrid, Q, channels,
    opts: SupercellVertexOptions, *, strict=True,
    initial_vertices: dict | None = None,
    output_vertices: dict | None = None,
):
    """Static bare bubble and conserving (within chosen skeleton) vertex.

    Uses *the same external sources* for all interaction-channel schemes.
    A complex-Q source is represented as cos/sin Hermitian components.
    """
    from .three_torus_benchmark import hermitian_source_components
    h0=np.asarray(geometry.h0,dtype=complex)
    hstatic=h0+background.Sigma_static[None,None]
    gg={};cov={};status={};timings={}
    for ch in channels:
        t0=perf_counter()
        val_gg=0.0
        val_cov=0.0
        ok=True
        for component_index,K in enumerate(
            hermitian_source_components(geometry,ch,Q)
        ):
            key=(str(ch),int(component_index))
            kk=np.asarray(K)[None]
            val_gg+=float(static_gg_tail_completed(
                background.G,kk,grid,hstatic,background.mu
            )["completed"][0,0].real)
            guess=(
                initial_vertices.get(key) if initial_vertices is not None
                else None
            )
            result=solve_channel_vertex_transfer(
                background,definition,K,h0,grid,0,opts=opts,
                initial_gamma=guess,
            )
            ok=ok and bool(result.converged)
            if result.converged and output_vertices is not None:
                # Only a certified converged linear response may become
                # the starting vertex at the next interaction strength.
                output_vertices[key]=np.asarray(result.Gamma).copy()
            if strict and not result.converged:
                raise RuntimeError(
                    f"{definition.mode} covariant {ch} did not converge: "
                    f"{result.final_error:.3e}"
                )
            val_cov+=float(static_response_tail_completed(
                background.G,kk,[result.Gamma],grid,hstatic,background.mu
            )["completed"][0,0].real)
        gg[ch]=val_gg
        cov[ch]=val_cov
        status[ch]=ok
        timings[ch]=perf_counter()-t0
    return gg,cov,status,timings


def one_shot_post_interaction_channel(
    background: ChannelGWResult, definition: ChannelDefinition,
    h0: np.ndarray, grid: MatsubaraGrid, target_particles: float,
    vertex_opts: SupercellVertexOptions, *,
    m_max: int | None = None,
    allow_unconverged: bool = False,
    geometry=None,
    optimize_translation: bool = True,
    recycle_krylov: bool = True,
    mu_tol: float = 1.e-10,
    mu_max_iter: int = 100,
):
    """Channel-consistent covariant W feedback and one-shot Dyson step.

    Shared GW background is not recomputed. The entire dynamic covariant
    INTERNAL n/B/J/E channel matrix is needed for a full post calculation.
    m_max restricts this to a window and keeps old W elsewhere, with
    fallback fraction explicitly recorded. This is not identical to the
    fixed-density-only post construction for BJ/E modes.
    """
    tstart = perf_counter()
    response, cache = compute_covariant_channel_susceptibility(
        background, definition, h0, grid, vertex_opts=vertex_opts,
        m_max=m_max, allow_unconverged=allow_unconverged,
        geometry=geometry, optimize_translation=optimize_translation,
        recycle_krylov=recycle_krylov,
    )
    tcov = perf_counter()
    Wpost,Ppost,fallback,identity_error=build_channel_gamma_screening(
        definition,response.chi_completed,
        background_W=background.W if m_max is not None else None
    )
    tw = perf_counter()
    sigma_c_post=compute_channel_correlation_self_energy(
        background.G,Wpost,definition,grid
    )
    tsigma = perf_counter()
    mu,G,_,_= _solve_mu_matrix_fast(
        h0,background.Sigma_static,sigma_c_post,grid,
        float(target_particles),float(background.mu),float(mu_tol),
        int(mu_max_iter)
    )
    tdyson = perf_counter()
    return SimpleNamespace(
        G=np.asarray(G),W=np.asarray(Wpost),Sigma_static=background.Sigma_static,
        Sigma_c=np.asarray(sigma_c_post),mu=float(mu),
        fallback_fraction=float(np.mean(fallback)),
        screening_identity_error=float(identity_error),
        response=response,background=background,
        translation_reduced=bool(response.translation_reduced),
        source_vertices_solved=int(response.source_vertices_solved),
        recycled_fallbacks=int(response.recycled_fallbacks),
        transfer_wall_seconds=np.asarray(response.transfer_wall_seconds),
        phase_seconds=dict(
            covariant=float(tcov-tstart),screening=float(tw-tcov),
            sigma=float(tsigma-tw),dyson=float(tdyson-tsigma)
        ),
    )
