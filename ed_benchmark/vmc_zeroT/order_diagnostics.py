"""Finite-cluster order diagnostics for Ruby ED and enumerated VMC states.

Keep the historical CO and LC normalizations intact. All new mode strengths
are normalized per triangle, not per primitive cell. Intertriangle matrices
are signed (not positive semidefinite). No automatic phase label is assigned.
"""
from __future__ import annotations
import numpy as np


def _peak(matrix, reps, momenta, basis):
    """Largest eigenvalue over q in an orthonormal orbital subspace."""
    nc=len(reps)
    best=None
    for q in momenta:
        phase=np.exp(2j*np.pi*(np.asarray(reps)@np.asarray(q)))/np.sqrt(nc)
        F=(phase[:,None,None]*basis[None,:,:]).reshape(6*nc,-1)
        C=F.conj().T@matrix@F
        vals,U=np.linalg.eigh((C+C.conj().T)/2)
        val=float(vals[-1])
        if best is None or val>best["value"]+1e-12:
            best=dict(value=val,q=tuple(map(float,q)),mode=F@U[:,-1],
                      multiplicity=int(np.sum(vals>=vals[-1]-1e-10*max(1.,abs(val)))))
    return best


def _numerical_radius(A):
    """max_{||v||=1}|v^dagger A v|; exact for Hermitian, numerical otherwise."""
    from scipy.optimize import minimize_scalar
    if np.linalg.norm(A-A.conj().T)<1e-12:
        return float(np.max(np.abs(np.linalg.eigvalsh((A+A.conj().T)/2))))
    def f(phi):
        B=np.exp(-1j*phi)*A
        return float(np.linalg.eigvalsh((B+B.conj().T)/2)[-1])
    grid=np.linspace(0.,2*np.pi,129)
    vals=np.array([f(x) for x in grid[:-1]])
    if np.ptp(vals)<1e-12:
        return max(0.,float(vals.max()))
    step=grid[1]
    ans=float(vals.max())
    for j in range(len(vals)):
        if vals[j]>=vals[j-1] and vals[j]>=vals[(j+1)%len(vals)]:
            r=minimize_scalar(lambda phi:-f(phi),
                              bounds=(grid[j]-step,grid[j]+step),method="bounded",
                              options={"xatol":1e-12})
            ans=max(ans,-float(r.fun))
    return max(0.,ans)


def _ground_fields(label,A,total,*,hermitian=False):
    """Equal-weight complete ground-manifold projected contribution."""
    d=A.shape[0]
    inside=float(np.vdot(A,A).real/d)
    total=float(total)
    outside=total-inside
    tol=1e-9*max(1.,abs(total))
    if outside < -tol:
        raise RuntimeError(f"{label}: projected weight exceeds full second moment")
    outside=max(0.,outside)
    radius=_numerical_radius(A)
    return {
        f"{label}_ground_manifold_contribution":inside,
        f"{label}_excited_contribution":outside,
        f"{label}_ground_manifold_weight":min(1.,inside/total) if total>1e-14 else None,
        f"{label}_max_ground_polarization":radius,
    }


def order_diagnostics(g, occ, psi, weights, raw_delta_n, J_second, tri,
                      current_projected=None, *, ground_manifold=False):
    """Historical leading CO, triangle-internal E, and triangle-total N modes.

    For CO a normalized mode u defines O=sum_i (F u)_i (n_i-nbar)/sqrt(Ntri).
    For LC O is the historical average signed loop current. The CO
    polarization is max |<O>| (numerical radius of P0 O P0); it can be a
    complex finite-q order parameter. Ground weights refer to the *same*
    full-correlation leading mode, not the reoptimized intertriangle mode.
    """
    nc=len(g.reps); ntri=len(tri); ns=g.Ns
    if ns!=6*nc or ntri!=2*nc:
        raise ValueError("Expected two elementary triangles per Ruby cell")
    block_id=np.full(ns,-1,dtype=int)
    for i,loop in enumerate(tri):
        if len(loop)!=3 or np.any(block_id[np.asarray(loop)]>=0):
            raise ValueError("Expected a disjoint triangle partition")
        block_id[np.asarray(loop)]=i
    if np.any(block_id<0):
        raise ValueError("Incomplete triangle partition")
    inter=block_id[:,None]!=block_id[None,:]
    # Basis convention matches CHARGE_STRUCTURE_FACTOR.md.
    E=np.zeros((6,4)); N=np.zeros((6,2))
    for a in range(2):
        E[3*a:3*a+3,2*a]=np.array([2,-1,-1])/np.sqrt(6.)
        E[3*a:3*a+3,2*a+1]=np.array([0,-1,1])/np.sqrt(2.)
        N[3*a:3*a+3,a]=1/np.sqrt(3.)
    # Validate that the cell-local A/B convention matches actual loops.
    for k in range(0,ns,3):
        if len(set(block_id[k:k+3]))!=1:
            raise ValueError("Ruby A/B orbital grouping differs from triangle loops")
    raw=np.asarray(raw_delta_n,complex)
    nbar=float(np.dot(occ.sum(axis=1),np.abs(psi[:,0])**2)/ns)
    dn=occ-nbar
    out={"order_diagnostics_Ntriangle":ntri,
         "order_diagnostics_Ncell":nc}
    for label,basis in (("CO",np.eye(6)),("CO_E",E),("CO_N",N)):
        peak=_peak(raw,g.reps,g.permitted_momenta(),basis)
        ip=_peak(raw*inter,g.reps,g.permitted_momenta(),basis)
        u=peak["mode"]
        total=float(peak["value"]/ntri)
        same_mode_inter=float(np.vdot(u,(raw*inter)@u).real/ntri)
        out.update({
            f"{label}_q1_diagnostic":peak["q"][0],
            f"{label}_q2_diagnostic":peak["q"][1],
            f"{label}_leading_mode_multiplicity":peak["multiplicity"],
            f"S_{label}_over_Ntriangle":total,
            f"S_{label}_local":total-same_mode_inter,
            f"S_{label}_intertriangle":same_mode_inter,
            f"S_{label}_intertriangle_qmax":float(ip["value"]/ntri),
            f"{label}_intertriangle_q1":ip["q"][0],
            f"{label}_intertriangle_q2":ip["q"][1],
        })
        if ground_manifold:
            a=dn@u/np.sqrt(ntri)
            A=psi.conj().T@(a[:,None]*psi)
            out.update(_ground_fields(label,A,total))
    # Six-component CO may include fluctuations of total triangle charge.
    u=_peak(raw,g.reps,g.permitted_momenta(),np.eye(6))["mode"].reshape(nc,6)
    out["CO_leading_E_fraction"]=float(np.linalg.norm(u@E)**2)
    out["CO_leading_N_fraction"]=float(np.linalg.norm(u@N)**2)
    J=np.asarray(J_second,complex)
    local=float(np.trace(J).real/ntri**2)
    out["S_LC_local"]=local
    ab=np.array([1. if loop[0]%6<3 else -1. for loop in tri])
    for label,sign in (("LC_same",np.ones(ntri)),("LC_opposite",ab)):
        v=sign/ntri
        total=float(np.vdot(v,J@v).real)
        out[f"S_{label}_intertriangle"]=total-local
        if ground_manifold:
            A=np.einsum("i,ijk->jk",v,current_projected)
            out.update(_ground_fields(label,A,total))
    # Local number constraint, independent of the chirality doublet assumption.
    probs=np.abs(psi)**2@weights
    number=np.column_stack([occ[:,loop].sum(axis=1) for loop in tri])
    out["triangle_number_mean"]=float(np.mean(probs@number))
    out["triangle_number_deviation_from_one"]=float(np.mean(probs@((number-1.)**2)))
    out["triangle_single_occupancy_probability"]=float(np.mean(probs@(number==1)))
    out["order_ground_manifold_complete"]=bool(ground_manifold)
    return out
