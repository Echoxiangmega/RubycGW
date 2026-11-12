"""Finite-temperature grand-canonical ED susceptibility on selected N sectors.

The exact finite torus has integer eigenvalues of N, but its mean
<N>_mu may be fractional. The response of a number-conserving O(Q) is a
grand-canonical mixture of canonical time correlators:

    C_GC(tau) = sum_N p_N [C_N(tau) + <O>_N* <O>_N]
                       - <O>_GC* <O>_GC.

The canonical connected results and the *full partition-function trace*
estimates are reused. Treat a truncated charge-sector window and a
stochastic trace as an approximation; report boundary weight and sector
probabilities. Do not reinterpret selected N ground energies as E0 at
a noninteger mean filling.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import logsumexp

from .momentum_ed import MomentumTorusED
from .ed18 import ED18Solver
from .ed18_thermal import thermal_response as hex_thermal_response
from .ed_cluster import RubyEDClusterSolver
from .ed_cluster_thermal import thermal_response as rect_thermal_response
from .three_torus_benchmark import TorusGeometry


def selected_charge_sectors(nsites: int, target_particles: float, padding: int):
    """All integer N adjacent to target plus optional extra thermal sectors."""
    nsites = int(nsites)
    target = float(target_particles)
    pad = int(padding)
    if not np.isfinite(target) or not 0 < target < nsites:
        raise ValueError("grand-canonical ED target mean N must be in (0, nsites)")
    if pad < 0:
        raise ValueError("charge-sector padding must be nonnegative")
    lo = max(0, int(np.floor(target + 1.e-11)) - pad)
    hi = min(nsites, int(np.ceil(target - 1.e-11)) + pad)
    # Vacuum and full-band sectors each have a unique Fock state. Their
    # canonical connected response vanishes identically, and their weight
    # can be included exactly, even at very dilute/nearly full filling.
    if lo >= hi:
        # A grand-canonical mean strictly between its only two supported
        # sectors is necessary; users can increase padding explicitly.
        raise ValueError(
            "grand-canonical trace requires >=2 charge sectors: "
            "increase --ed-charge-sector-padding"
        )
    if not lo < target < hi:
        raise ValueError(
            f"target <N>={target} not bracketed by selected N={lo}..{hi}; "
            "increase --ed-charge-sector-padding"
        )
    return list(range(lo,hi+1))


def _grand_weights(log_partition, target: float, T: float):
    nums = np.asarray(sorted(log_partition),dtype=float)
    logz = np.asarray([float(log_partition[int(n)]) for n in nums],float)
    beta = 1.0/float(T)
    def eval_mean(mu):
        lp=logz + beta*float(mu)*nums
        p=np.exp(lp-logsumexp(lp))
        return float(p@nums), p
    center = float((logz[0]-logz[-1])/(beta*(nums[-1]-nums[0])))
    span = max(1.0,10*float(T),float(np.ptp(logz))/max(beta,1))
    lo,hi=center-span,center+span
    for i in range(50):
        nlo,_=eval_mean(lo)
        nhi,_=eval_mean(hi)
        if nlo < target < nhi:
            break
        span *= 2.0
        lo,hi=center-span,center+span
    else:
        raise RuntimeError("grand-canonical ED chemical potential cannot be bracketed")
    mu=float(brentq(lambda x:eval_mean(x)[0]-target,lo,hi,xtol=1e-12))
    navg,weights=eval_mean(mu)
    return mu,navg,dict(zip(map(int,nums),map(float,weights)))


def _fock_boundary_response(geom,params,V,T,Q,channels,n,tau_points):
    """Exact singleton thermal sector N=0 or N=Nsites, zero connected χ."""
    if n not in (0,geom.nsites):
        raise ValueError("not a single-state boundary sector")
    from .pseudospin import primitive_pseudospin_vertex
    Q=geom.validate_momentum(Q)
    ncell=int(geom.ncell)
    mean=np.zeros(len(channels),dtype=complex)
    if n:
        phase=np.exp(-2j*np.pi*np.asarray(geom.cells,dtype=float)@np.asarray(Q))/np.sqrt(ncell)
        for i,ch in enumerate(channels):
            mean[i]=np.sum(phase)*np.trace(primitive_pseudospin_vertex(ch))
        E=float(np.trace(geom.h0[0,0]).real+6*ncell*float(V))
    else:
        E=0.
    tau=np.linspace(0,1/float(T),int(tau_points))
    nc=len(channels)
    response=SimpleNamespace(
        shifted_partition_estimate=1.0,
        means=mean,
        correlation_tau_matrix=np.zeros((int(tau_points),nc,nc),complex),
        structure_matrix=np.zeros((nc,nc),complex),
        tau_grid=tau,
    )
    return E,response


def grand_canonical_ed_response(
    geom: TorusGeometry, params, *, V: float,T: float,target_particles: float,
    Q,channels,n_trace_vectors:int=6,tau_points:int=17,
    seed:int=20261111,padding:int=1,ed_eigs:int=2,
    use_momentum:bool=True,trace_mode:str="stochastic",
):
    """Truncated-sector GC ED charge-conserving static response, mean N fixed."""
    Ns=selected_charge_sectors(geom.nsites,target_particles,padding)
    responses={}
    ground_energies={}
    logZN={}
    dimension={}
    for n in Ns:
        if n in (0,geom.nsites):
            energy_boundary,response=_fock_boundary_response(
                geom,params,V,T,Q,channels,n,int(tau_points)
            )
            E=float(energy_boundary)
            z=1.0
            ground_energies[n]=E
            logZN[n]=-E/float(T)
            dimension[n]=1
            responses[n]=response
            continue
        if use_momentum:
            sol=MomentumTorusED(geom,params,n_particles=n)
            spec=sol.ground_spectrum(V,n_eigs=max(2,int(ed_eigs)))
            response=sol.thermal_response(
                V,T,Q,channels,
                n_trace_vectors=n_trace_vectors,tau_points=tau_points,
                seed=int(seed)+7919*n,energy_shift=float(spec.energies[0]),
                trace_mode=trace_mode,
            )
        else:
            if geom.name=="hex18":
                sol=ED18Solver(params,n_particles=n)
                spec=sol.solve(V,n_eigs=max(2,int(ed_eigs)))
                response=hex_thermal_response(
                    sol,V,T,Q,channels=channels,
                    n_trace_vectors=n_trace_vectors,tau_points=tau_points,
                    seed=int(seed)+7919*n,
                    energy_shift=float(spec.energies[0])
                )
            else:
                L1,L2=(3,1) if geom.name=="3x1" else (2,2)
                sol=RubyEDClusterSolver(L1,L2,params,n_particles=n)
                spec=sol.solve(V,n_eigs=max(2,int(ed_eigs)))
                response=rect_thermal_response(
                    sol,V,T,Q,channels=channels,
                    n_trace_vectors=n_trace_vectors,tau_points=tau_points,
                    seed=int(seed)+7919*n,
                    energy_shift=float(spec.energies[0])
                )
        z=float(response.shifted_partition_estimate)
        if z<=0 or not np.isfinite(z):
            raise RuntimeError(f"invalid ED thermal trace in charge sector N={n}")
        responses[n]=response
        E=float(spec.energies[0])
        ground_energies[n]=E
        logZN[n]=float(np.log(z)-E/float(T))
        dimension[n]=int(sol.dimension)

    mu,navg,probs=_grand_weights(logZN,float(target_particles),float(T))
    nc=len(channels)
    mean=np.zeros(nc,dtype=complex)
    corr_raw=np.zeros_like(np.asarray(next(iter(responses.values())).correlation_tau_matrix),dtype=complex)
    structure_raw=np.zeros((nc,nc),dtype=complex)
    for n in Ns:
        r=responses[n]
        weight=probs[n]
        means=np.asarray(r.means,dtype=complex)
        outer=np.outer(means.conj(),means)
        mean+=weight*means
        corr_raw+=weight*(np.asarray(r.correlation_tau_matrix,dtype=complex)+outer[None])
        structure_raw+=weight*(np.asarray(r.structure_matrix,dtype=complex)+outer)
    outer_global=np.outer(mean.conj(),mean)
    corr=corr_raw-outer_global[None]
    struct=.5*(structure_raw-outer_global+
                (structure_raw-outer_global).conj().T)
    tau_grid=np.asarray(next(iter(responses.values())).tau_grid,float)
    chi=simpson(corr,x=tau_grid,axis=0)
    chi=.5*(chi+chi.conj().T)
    return SimpleNamespace(
        ensemble="grand",temperature=float(T),V=float(V),
        target_particles=float(target_particles),average_particles=float(navg),
        mu=float(mu),sectors=np.asarray(Ns,int),
        sector_probabilities=np.asarray([probs[n] for n in Ns],float),
        edge_probability=float(probs[Ns[0]]+probs[Ns[-1]]),
        ground_energies=ground_energies,
        dimensions=dimension,log_partition=logZN,
        means=mean,tau_grid=tau_grid,
        correlation_tau_matrix=corr,structure_matrix=struct,
        susceptibility_matrix=chi,n_trace_vectors=n_trace_vectors,
        tau_points=tau_points,
    )
