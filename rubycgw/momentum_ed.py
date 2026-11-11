"""Genuine many-body translation-momentum blocks for finite Ruby PBC ED.

Supports rectangular 3x1, 2x2 and index-three skew hex18 tori, all with
spinless fermions. Fock-basis orbit projectors include fermionic permutation
signs and translation stabilizers. H(K) is built directly as a small sparse
matrix, *not* by applying the full Hilbert-space Hamiltonian at every Krylov
step. The symmetry is exact for the primitive-translation-invariant model.

Finite-temperature canonical typicality separately samples each K sector,
with operators O(Q) transferring K -> K-Q, and uses the exact block trace
normalization. A grand-canonical Green-sector algorithm is a distinct problem:
the existing thermal_lanczos_green still uses the particle-sector solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply

from .model import RubyParameters
from .ed_cluster import RubyEDClusterSolver
from .ed18 import ED18Solver
from .pseudospin import primitive_pseudospin_vertex
from .supercell import primitive_cell_to_supercell
from .three_torus_benchmark import TorusGeometry


def _translated_state(state: int, site_perm: np.ndarray) -> tuple[int, int]:
    """Translate a Fock bitstring, preserving the fermionic reordering sign."""
    moved = []
    v = int(state)
    while v:
        bit = v & -v
        i = bit.bit_length()-1
        moved.append(int(site_perm[i]))
        v ^= bit
    inversions = 0
    for j in range(len(moved)):
        inversions += sum(moved[j] > x for x in moved[j+1:])
    target = sum(1 << j for j in moved)
    return target, (-1 if inversions%2 else 1)


def _fermion_hop(state: int, i: int, j: int):
    """Apply c_i^dagger c_j on a Fock state."""
    if i == j:
        return (state, 1) if (state & (1 << i)) else None
    if not (state & (1 << j)) or (state & (1 << i)):
        return None
    a, b = sorted((i,j))
    between = ((1 << b)-1) ^ ((1 << (a+1))-1)
    sign = -1 if ((state & between).bit_count()%2) else 1
    return state ^ (1 << i) ^ (1 << j),sign


def _orbital_permutation(geom: TorusGeometry, shift) -> np.ndarray:
    Rmap = {tuple(map(int,R)):c for c,R in enumerate(geom.cells)}
    Lx,Ly = (3,1) if geom.name=="3x1" else (2,2)
    out=np.empty(geom.nsites,dtype=np.int32)
    for c,R in enumerate(geom.cells):
        pos=np.asarray(R,dtype=int)+np.asarray(shift,dtype=int)
        if geom.name=="hex18":
            tgt,_=primitive_cell_to_supercell(pos)
        else:
            tgt=Rmap[(int(pos[0])%Lx,int(pos[1])%Ly)]
        for a in range(6):
            out[6*c+a]=6*tgt+a
    return out


@dataclass
class MomentumBlock:
    k: tuple[float,float]
    projector: sparse.csr_matrix
    full_indices: np.ndarray
    full_amplitudes: np.ndarray
    full_to_col: np.ndarray
    full_to_amp: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.projector.shape[1])


class MomentumTorusED:
    """Fixed-N momentum spaces for the SAME finite ED and GW geometry."""

    def __init__(self, geom: TorusGeometry, params: RubyParameters, n_particles: int):
        self.geom=geom
        self.params=params
        self.n_particles=int(n_particles)
        if geom.name=="hex18":
            self.solver=ED18Solver(params,n_particles=self.n_particles)
        else:
            Lx,Ly=(3,1) if geom.name=="3x1" else (2,2)
            self.solver=RubyEDClusterSolver(
                Lx,Ly,params,n_particles=self.n_particles
            )
        self.basis=np.asarray(self.solver.basis,dtype=np.uint64)
        self.full_dimension=int(self.solver.dimension)
        self.dimension=self.full_dimension
        self.k_values=tuple(tuple(map(float,x)) for x in geom.allowed_momenta())
        self.perms=[_orbital_permutation(geom,g) for g in geom.cells]
        self.blocks=self._build_blocks()
        self._hcache={}

    def index_of_state(self,state: int) -> int:
        solver=self.solver
        if hasattr(solver,"_dense_lookup") and solver._dense_lookup is not None:
            return int(solver._dense_lookup[int(state)])
        if hasattr(solver,"index"):
            return int(solver.index[int(state)])
        return int(solver._state_indices(np.array([state],dtype=np.uint64))[0])

    def k_index(self,k) -> int:
        kv=np.asarray(k,dtype=float).reshape(2)
        for i,v in enumerate(self.k_values):
            if np.max(np.abs(np.remainder(kv-np.asarray(v)+.5,1.0)-.5))<1e-8:
                return i
        raise ValueError(f"K={k} is not allowed on {self.geom.name}")

    def _build_blocks(self) -> tuple[MomentumBlock,...]:
        D=self.full_dimension
        seen=np.zeros(D,dtype=bool)
        rows=[[] for _ in self.k_values]
        cols=[[] for _ in self.k_values]
        vals=[[] for _ in self.k_values]
        dims=[0 for _ in self.k_values]
        qvals=np.asarray(self.k_values,dtype=float)
        cell_coords=np.asarray(self.geom.cells,dtype=float)
        group_phases=np.exp(2j*np.pi*(qvals@cell_coords.T))
        for fullid,raw in enumerate(self.basis):
            if seen[fullid]:
                continue
            state=int(raw)
            images=[]
            for perm in self.perms:
                dest,sign=_translated_state(state,perm)
                destid=self.index_of_state(dest)
                images.append((destid,sign))
                seen[destid]=True
            orbit=np.array(sorted(set(x[0] for x in images)),dtype=np.int64)
            for ik in range(len(self.k_values)):
                coeff={int(j):0j for j in orbit}
                for g,(idj,sign) in enumerate(images):
                    coeff[idj]+=sign*group_phases[ik,g]
                norm=float(np.sqrt(sum(abs(z)**2 for z in coeff.values())))
                if norm < 1e-9:
                    continue
                for idj,z in coeff.items():
                    if abs(z)>1e-12:
                        rows[ik].append(idj)
                        cols[ik].append(dims[ik])
                        vals[ik].append(z/norm)
                dims[ik]+=1
        if not np.all(seen):
            raise RuntimeError("translation orbit decomposition left Fock states out")
        blocks=[]
        for ik,k in enumerate(self.k_values):
            P=sparse.coo_matrix(
                (np.asarray(vals[ik],dtype=complex),
                 (np.asarray(rows[ik],dtype=np.int64),np.asarray(cols[ik],dtype=np.int64))),
                shape=(D,dims[ik]),dtype=complex
            ).tocsr()
            P.sort_indices()
            full_to_col=np.full(D,-1,dtype=np.int32)
            full_to_amp=np.zeros(D,dtype=complex)
            for row,col,v in zip(rows[ik],cols[ik],vals[ik]):
                full_to_col[row]=col
                full_to_amp[row]=v
            blocks.append(MomentumBlock(
                k=k,projector=P,
                full_indices=np.asarray(rows[ik],dtype=np.int64),
                full_amplitudes=np.asarray(vals[ik],dtype=complex),
                full_to_col=full_to_col,full_to_amp=full_to_amp
            ))
        if sum(b.dimension for b in blocks)!=D:
            raise RuntimeError("momentum sector dimensions do not add up to full sector")
        return tuple(blocks)

    def project_onebody(self, M: np.ndarray, k_from: int, k_to: int) -> sparse.csr_matrix:
        """Second-quantize a one-body matrix directly between momentum sectors."""
        src=self.blocks[int(k_from)]
        dst=self.blocks[int(k_to)]
        mat=np.asarray(M,dtype=complex)
        if mat.shape!=(self.geom.nsites,self.geom.nsites):
            raise ValueError("one-body matrix dimensions mismatch")
        pairs=[(int(i),int(j),complex(mat[i,j]))
               for i,j in np.argwhere(np.abs(mat)>1e-14)]
        csc=src.projector.tocsc()
        r=[];c=[];v=[]
        for col in range(src.dimension):
            lo,hi=csc.indptr[col],csc.indptr[col+1]
            for ip in range(lo,hi):
                fullid=int(csc.indices[ip])
                amp=csc.data[ip]
                raw=int(self.basis[fullid])
                for i,j,h in pairs:
                    res=_fermion_hop(raw,i,j)
                    if res is None:
                        continue
                    target,sign=res
                    tid=self.index_of_state(target)
                    dcol=int(dst.full_to_col[tid])
                    if dcol>=0:
                        r.append(dcol);c.append(col)
                        v.append(np.conj(dst.full_to_amp[tid])*amp*sign*h)
        return sparse.coo_matrix(
            (v,(r,c)),shape=(dst.dimension,src.dimension),dtype=complex
        ).tocsr()

    def hamiltonian(self, V: float, ik: int) -> sparse.csr_matrix:
        key=(float(V),int(ik))
        if key in self._hcache:
            return self._hcache[key]
        B=self.blocks[int(ik)]
        M=self.project_onebody(self.solver.h0,ik,ik)
        D=self.full_dimension
        if hasattr(self.solver,"interaction_count"):
            diag=V*np.asarray(self.solver.interaction_count,dtype=float)
        else:
            diag=np.zeros(D)
        weights=np.abs(B.full_to_amp)**2
        diag_by_col=np.bincount(
            np.maximum(B.full_to_col,0),
            weights=diag*weights,
            minlength=B.dimension
        )[:B.dimension]
        M=M+sparse.diags(diag_by_col,format="csr")
        M=(M+M.getH())*.5
        self._hcache[key]=M.tocsr()
        return self._hcache[key]

    def ground_spectrum(self,V:float,*,n_eigs:int=5,tol=1e-10,maxiter=6000):
        records=[]
        for ik,block in enumerate(self.blocks):
            d=block.dimension
            if d==0:
                continue
            H=self.hamiltonian(V,ik)
            if d<=max(6,int(n_eigs)+1):
                vals,vecs=np.linalg.eigh(H.toarray())
            else:
                k=min(max(int(n_eigs),2),d-2)
                vals,vecs=eigsh(H,k=k,which="SA",tol=tol,maxiter=maxiter)
            order=np.argsort(np.asarray(vals).real)
            for pos in order[:min(len(order),int(n_eigs))]:
                records.append((float(np.asarray(vals)[pos].real),ik,vecs[:,pos]))
        records.sort(key=lambda x:x[0])
        energies=np.asarray([x[0] for x in records[:max(2,n_eigs)]])
        if len(energies)<2:
            raise RuntimeError("insufficient many-body eigenvalues")
        multiplicity=int(np.count_nonzero(abs(energies-energies[0])<=1e-8))
        gap=(float(energies[multiplicity]-energies[0])
             if multiplicity<len(energies) else float("nan"))
        return SimpleNamespace(
            energies=energies,ground_multiplicity=multiplicity,
            gap_above_manifold=gap,
            momentum_labels=[self.k_values[x[1]] for x in records[:len(energies)]],
            sector_dimensions=[b.dimension for b in self.blocks]
        )

    def thermal_response(
        self,V:float,T:float,Q,channels,*,n_trace_vectors=6,
        tau_points=17,seed=20261111,energy_shift=None,
        trace_mode="stochastic",
    ):
        """Canonical finite-T C(Q,tau) and chi using K-sector stochastic traces."""
        if T<=0 or tau_points<3 or tau_points%2!=1:
            raise ValueError("T must be positive, tau_points odd and >=3")
        Q=self.geom.validate_momentum(Q)
        channels=tuple(str(x) for x in channels)
        nc=len(channels)
        if energy_shift is None:
            energy_shift=float(self.ground_spectrum(V,n_eigs=2).energies[0])
        from scipy.integrate import simpson
        tau=np.linspace(0,1.0/T,int(tau_points))
        beta=1.0/T
        coord=np.asarray(self.geom.cells)
        phase=np.exp(-2j*np.pi*(coord@Q))/np.sqrt(self.geom.ncell)
        sources=[]
        for ch in channels:
            vertex=primitive_pseudospin_vertex(ch)
            M=np.zeros((self.geom.nsites,self.geom.nsites),dtype=complex)
            for c in range(self.geom.ncell):
                M[6*c:6*(c+1),6*c:6*(c+1)]=phase[c]*vertex
            sources.append(M)
        rng=np.random.default_rng(int(seed))
        zsum=0.0
        mean=np.zeros(nc,dtype=complex)
        structure=np.zeros((nc,nc),dtype=complex)
        corr=np.zeros((len(tau),nc,nc),dtype=complex)
        n_exp=0
        kvalues=np.asarray(self.k_values)
        for ik,block in enumerate(self.blocks):
            if block.dimension==0:
                continue
            kto=self.k_index(kvalues[ik]-Q)
            btarget=self.blocks[kto]
            Hs=self.hamiltonian(V,ik)
            Ht=self.hamiltonian(V,kto)
            As=(-Hs + sparse.eye(Hs.shape[0],format="csr")*float(energy_shift)).tocsr()
            At=(-Ht + sparse.eye(Ht.shape[0],format="csr")*float(energy_shift)).tocsr()
            transfer=[self.project_onebody(M,ik,kto) for M in sources]
            # E[r r^dag]=I for unit-modulus random phases. In exact
            # test mode, iterate the full orthonormal block basis instead.
            if trace_mode == "exact":
                if block.dimension > 128:
                    raise ValueError("exact momentum-sector trace is restricted to dimensions <=128")
                trace_vecs=np.eye(block.dimension,dtype=complex)
                trace_weight=1.0
            elif trace_mode == "stochastic":
                trace_vecs=np.exp(
                    1j*rng.uniform(0,2*np.pi,(block.dimension,int(n_trace_vectors)))
                )
                trace_weight=1.0/float(n_trace_vectors)
            else:
                raise ValueError("trace_mode must be stochastic or exact")
            for ir in range(trace_vecs.shape[1]):
                r=np.asarray(trace_vecs[:,ir],dtype=complex)
                wr=trace_weight
                rt=expm_multiply(
                    As,r,start=0,stop=beta,num=tau_points,
                    endpoint=True,traceA=complex(As.diagonal().sum())
                )
                n_exp+=1
                mid=rt[tau_points//2]
                zsum+=wr*float(np.vdot(mid,mid).real)
                Bmid=np.column_stack([Op@mid for Op in transfer])
                structure+=wr*(Bmid.conj().T@Bmid)
                if ik==kto:
                    mean+=wr*np.asarray([np.vdot(mid,Op@mid) for Op in transfer])
                B0=np.column_stack([Op@r for Op in transfer])
                Bt=expm_multiply(
                    At,B0,start=0,stop=beta,num=tau_points,
                    endpoint=True,traceA=complex(At.diagonal().sum())
                )
                n_exp+=1
                for j in range(tau_points):
                    left=np.column_stack([
                        Op@rt[tau_points-1-j] for Op in transfer
                    ])
                    corr[j]+=wr*(left.conj().T@Bt[j])
        if zsum<=0 or not np.isfinite(zsum):
            raise RuntimeError("invalid sector thermal partition normalization")
        mean/=zsum
        offset=np.outer(mean.conj(),mean)
        structure=structure/zsum-offset
        corr=corr/zsum-offset[None]
        chi=simpson(corr,x=tau,axis=0)
        chi=.5*(chi+chi.conj().T)
        structure=.5*(structure+structure.conj().T)
        return SimpleNamespace(
            temperature=T,beta=beta,V=V,q=np.asarray(Q),channels=channels,
            tau_grid=tau,correlation_tau_matrix=corr,
            susceptibility_matrix=chi,structure_matrix=structure,
            means=mean,energy_shift=float(energy_shift),
            shifted_partition_estimate=float(zsum),
            n_trace_vectors=int(n_trace_vectors),
            tau_points=int(tau_points),
            sector_dimensions=[b.dimension for b in self.blocks],
            full_dimension=self.full_dimension,
            block_exponential_calls=n_exp,
            momentum_labels=self.k_values,
        )
