"""Matched Fierz GW skeleton for six-orbital cluster ED embedding.

For each mode, the full lattice GW and the six-site GW double-counting
subtraction MUST use the same bosonic vertices and couplings. The ED impurity
Hamiltonian is the original density-density V interaction independent of Fierz
choice. For the historical density mode, retain the original split between
Hartree and Fock+correlation exactly.

For pseudospin and BJ the full physical Hartree-Fock static self-energy is
kept as Sigma_H in the common Dyson solver, and Sigma_emb contains the
correlation self-energy and ED-minus-GW local correction. This is only a
storage convention: Sigma_H + Sigma_emb is the physical embedding self-energy.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .cluster_ed_gw import cluster_gw_self_energy
from .fierz_channel_gw import (
    ChannelDefinition,build_channel_definition,channel_static_self_energy,
    compute_channel_polarization,compute_channel_screened_interaction,
    compute_channel_correlation_self_energy,
)
from .supercell_gw import (
    hartree_self_energy_matrix,compute_polarization_matrix,
    compute_screened_interaction_matrix,
)
from .supercell_gw_split import compute_sigma_gw_split_matrix
from .grids import MatsubaraGrid


@dataclass(frozen=True)
class EmbeddingChannelDefinitions:
    mode: str
    V: float
    lattice: ChannelDefinition | None
    cluster: ChannelDefinition | None


def build_embedding_channels(interactions,ncell,mode,Vq=None):
    """Build one *matched* Fierz representation on lattice and six-site ED.

    The supported interaction is the original six intra-triangle bonds of
    equal strength V per six-site cell. Other nonlocal density interactions
    are rejected rather than being silently omitted from the transformed
    lattice GW channel.
    """
    mode=str(mode).strip().lower()
    if mode not in ("density","pseudospin","bj"):
        raise ValueError("cluster GW channel must be density, pseudospin or bj")
    ncell=int(ncell)
    if ncell<1:
        raise ValueError("cluster needs at least one primitive cell")
    pairs=[]
    strengths=[]
    for i,j,v in interactions:
        i,j=int(i),int(j)
        if not (0<=i<6 and 0<=j<6 and i!=j):
            raise ValueError("invalid six-site cluster interaction index")
        if not np.isfinite(v):
            raise ValueError("nonfinite cluster bond strength")
        pairs.append((i,j))
        strengths.append(float(v))
    if len(pairs)!=6 or len(set(tuple(sorted(p)) for p in pairs))!=6:
        raise ValueError("Fierz cluster supports six distinct intra-triangle V bonds")
    if not np.allclose(strengths,strengths[0],rtol=1e-12,atol=1e-12):
        raise ValueError("Fierz cluster currently requires a single intra-triangle V")
    V=float(strengths[0])
    nsite=6*ncell
    full_pairs=tuple((6*c+i,6*c+j) for c in range(ncell) for i,j in pairs)
    if Vq is not None:
        actual=np.asarray(Vq,dtype=complex)
        if actual.shape!=(1,1,nsite,nsite):
            raise ValueError("folded interaction matrix shape mismatch")
        expected=np.zeros((nsite,nsite),complex)
        for i,j in full_pairs:
            expected[i,j]+=V
            expected[j,i]+=V
        error=np.max(np.abs(actual[0,0]-expected),initial=0.)
        if error>1e-9:
            raise ValueError(
                f"Fierz-channel lattice has unrepresented interaction bonds "
                f"(max error {error:.3e}); use density instead"
            )
    if mode=="density":
        return EmbeddingChannelDefinitions(mode,V,None,None)
    return EmbeddingChannelDefinitions(
        mode,V,
        build_channel_definition(full_pairs,nsite,V,mode),
        build_channel_definition(tuple(pairs),6,V,mode),
    )


def lattice_gw_map(G,rho,h0,Vq,grid,M,*,static_reference=None):
    """Return (static, dynamic/rest, P, W) at one embedding iteration.

    density: static=Hartree, rest=Fock+dynamic; identical to legacy.
    bj/pseudospin: static=full HF, rest=dynamic correlation only.
    """
    if M.mode=="density":
        density=np.real(np.diag(np.asarray(rho)[0,0]))
        sh=hartree_self_energy_matrix(density,np.asarray(Vq)[0,0])
        P=compute_polarization_matrix(G,grid,backend="fft")
        W=compute_screened_interaction_matrix(P,Vq)
        rest=compute_sigma_gw_split_matrix(
            G,W,Vq,grid,h0,0.,
            sh if static_reference is None else static_reference,
            backend="fft",
        )
        return sh,rest,P,W
    rho0=np.asarray(rho,dtype=complex)[0,0]
    static,_,_=channel_static_self_energy(rho0,M.lattice)
    P=compute_channel_polarization(G,M.lattice,grid)
    W=compute_channel_screened_interaction(P,M.lattice)
    rest=compute_channel_correlation_self_energy(G,W,M.lattice,grid)
    return static,rest,P,W


def cluster_gw_map(Gc,rhoc,interactions,grid,M,*,Vc=None):
    """Return the local GW total including exact static HF for subtraction."""
    if M.mode=="density":
        if Vc is None:
            from .cluster_ed_gw import cluster_interaction_matrix
            Vc=cluster_interaction_matrix(interactions,6)
        return cluster_gw_self_energy(Gc,rhoc,Vc,grid)
    local_grid=MatsubaraGrid(
        nk1=1,nk2=1,nw=grid.nw,nOmega=grid.nOmega,T=grid.T
    )
    static,_,_=channel_static_self_energy(rhoc,M.cluster)
    Gfield=np.asarray(Gc,complex)[:,None,None]
    P=compute_channel_polarization(Gfield,M.cluster,local_grid)
    W=compute_channel_screened_interaction(P,M.cluster)
    sigma_c=compute_channel_correlation_self_energy(
        Gfield,W,M.cluster,local_grid
    )[:,0,0]
    return static[None]+sigma_c,P,W


__all__=[
    "EmbeddingChannelDefinitions","build_embedding_channels",
    "lattice_gw_map","cluster_gw_map",
]
