"""Ruby ED geometry, PBC, elementary loops, and verified space-group maps.

Conventions: H0=sum t_ij c^dag_i c_j+h.c. (positive t); U(1) spinless;
interactions V only on red intra-triangle bonds. All orbitals are zero-based.
"""
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np

SQ3 = math.sqrt(3)
A = np.array([[-(1+SQ3)/2, -(1+SQ3)/2],
              [-(3+SQ3)/2,  (3+SQ3)/2]], dtype=float)
ORBITALS = np.array([[0,SQ3/2],[-.5,0],[.5,0],
                     [0,-1-SQ3/2],[-.5,-1],[.5,-1]], dtype=float)
# source orbital, destination orbital, primitive dx,dy, hopping label
BONDS = [(0,1,0,0,'t'),(0,2,0,0,'t'),(2,1,0,0,'t'),
         (3,4,0,0,'t'),(3,5,0,0,'t'),(4,5,0,0,'t'),
         (1,4,0,0,'t1'),(5,0,0,-1,'t1'),(2,3,-1,0,'t1'),
         (3,1,0,-1,'t2'),(2,5,0,0,'t2'),(0,4,-1,0,'t2')]
SPECS = {
    '2x2': ([[2,0],[0,2]], [(0,0),(1,0),(0,1),(1,1)]),
    '3x1': ([[3,0],[0,1]], [(0,0),(1,0),(2,0)]),
    'ring3': ([[1,1],[-1,2]], [(0,0),(1,0),(1,1)]),
}
# A physical hexagon center. Its 60-degree rotations exchange t1 and t2.
POINT_CENTER = np.array([0., -2.-SQ3/2])

@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    kind: str
    delta: tuple
    winding: tuple


def _int_solution(M, d):
    v = np.linalg.solve(M, d)
    if np.max(np.abs(v-np.rint(v))) < 1e-8:
        return tuple(np.rint(v).astype(int))
    return None


class RubyPBC:
    def __init__(self, name, t=1., t1=.5, t2=.5):
        if name not in SPECS:
            raise ValueError(f'Unknown cluster: {name}')
        matrix,reps = SPECS[name]
        self.name=name
        self.M=np.array(matrix,dtype=int)
        self.reps=list(reps)
        self.num_cells=len(reps)
        self.Ns=6*len(reps)
        self.t={'t':float(t), 't1':float(t1), 't2':float(t2)}
        self.B=A@self.M
        self.positions=np.array([A@np.array(R)+ORBITALS[o]
                                 for R in reps for o in range(6)])
        self.edges=[]
        for cell,R in enumerate(reps):
            for a,b,dx,dy,kind in BONDS:
                target=(R[0]+dx,R[1]+dy)
                cell2, winding = self.reduce(target)
                self.edges.append(Edge(6*cell+a, 6*cell2+b,kind,(dx,dy),winding))
        self.check()

    def reduce(self, cell):
        for i,r in enumerate(self.reps):
            w=_int_solution(self.M,np.array(cell,dtype=int)-r)
            if w is not None: return i,w
        raise ValueError(f'No representative for {cell}')

    def translations(self):
        return [self.translate((1,0)),self.translate((0,1))]

    def translate(self, dr):
        p=np.empty(self.Ns,dtype=np.int32)
        for ci,R in enumerate(self.reps):
            cj,_=self.reduce((R[0]+dr[0], R[1]+dr[1]))
            p[6*ci:6*ci+6]=np.arange(6*cj,6*cj+6)
        return p

    def check(self):
        deg=np.zeros(self.Ns,dtype=int)
        for e in self.edges: deg[e.i]+=1;deg[e.j]+=1
        assert np.all(deg==4)
        assert len(self.edges)==2*self.Ns
        assert len({tuple(sorted((e.i,e.j))) for e in self.edges})==len(self.edges)
        from collections import Counter
        c=Counter(e.kind for e in self.edges)
        assert c=={'t':6*self.num_cells,'t1':3*self.num_cells,'t2':3*self.num_cells}

    def site_map(self, matrix, center=POINT_CENTER):
        """Geometric site permutation, irrespective of t1/t2 amplitudes.

        A candidate is valid only if it maps the finite torus into itself.
        """
        p=np.empty(self.Ns,dtype=np.int32)
        for i,r in enumerate(self.positions):
            q=center+matrix@(r-center)
            candidates=[]
            for j,s in enumerate(self.positions):
                d=np.linalg.solve(self.B,q-s)
                if np.max(np.abs(d-np.rint(d)))<1e-8:candidates.append(j)
            if len(candidates)!=1:return None
            p[i]=candidates[0]
        if len(set(map(int,p)))!=self.Ns:return None
        return p

    def check_signed_map(self,p,signs=None):
        """Test c_i -> signs[i]*c_{p[i]} against ALL independent edges.

        This check tests the actual hopping numbers and the interaction
        (intra-triangle) bonds, not a guessed geometric point-group label.
        """
        if p is None:return False
        if signs is None:signs=np.ones(self.Ns,dtype=np.int8)
        if len(signs)!=self.Ns:return False
        edges={}
        for e in self.edges:
            key=tuple(sorted((e.i,e.j)))
            edges[key]=(self.t[e.kind],e.kind=='t')
        for e in self.edges:
            key=tuple(sorted((int(p[e.i]),int(p[e.j]))))
            value=edges.get(key)
            if value is None:return False
            target_t,target_v=value
            if target_v != (e.kind=='t'):return False
            if not np.isclose(self.t[e.kind]*signs[e.i]*signs[e.j],target_t,
                              atol=1.e-11,rtol=1.e-11):return False
        return True

    def symmetry(self,matrix,center=POINT_CENTER):
        """Plain geometric symmetry suitable for QuSpin site permutations."""
        p=self.site_map(matrix,center)
        return p if self.check_signed_map(p) else None

    def signed_symmetry(self,matrix,center=POINT_CENTER):
        """Gauge-compensated rotation/reflection for t1=-t2.

        A and B triangle orbitals receive relative sign (-1). This is
        a NUMBER-CONSERVING phase gauge, NOT QuSpin negative-site mapping
        (which represents particle-hole conjugation instead).
        Returns (permutation, source-site signs) when the test succeeds.
        """
        p=self.site_map(matrix,center)
        if p is None:return None
        signs=np.array([1 if (i%6)<3 else -1 for i in range(self.Ns)],
                       dtype=np.int8)
        return (p,signs) if self.check_signed_map(p,signs) else None

    @staticmethod
    def _rotation(degrees):
        a=math.radians(degrees)
        return np.array([[math.cos(a),-math.sin(a)],
                         [math.sin(a),math.cos(a)]])

    @staticmethod
    def _mirror(degrees):
        a=math.radians(degrees)
        return np.array([[math.cos(2*a),math.sin(2*a)],
                         [math.sin(2*a),-math.cos(2*a)]])

    def point_maps(self):
        """All validated *plain* point group maps of the finite torus.

        Generic t1/t2: 2x2 and ring3 have C3 and 3 mirrors, while
        3x1 retains one mirror but breaks C3 by its boundary conditions.
        When t1=t2, C6 and the additional mirrors are ordinary symmetries.
        """
        out={f'C{n}':self.symmetry(self._rotation(360/n)) for n in (6,3,2)}
        out['mirrors']={str(a):p for a in range(0,180,30)
                        if (p:=self.symmetry(self._mirror(a))) is not None}
        # Keep legacy key, but choose a verified mirror, not an assumed axis.
        out['mirror']=next(iter(out['mirrors'].values()),None)
        return out

    def gauge_point_maps(self):
        """Validated gauge-compensated symmetries, NOT ordinary QuSpin blocks.

        In particular, (G*C6)^6=(-1)^N on the fixed-N Fock space,
        because C6 exchanges A and B triangles and G changes the B sign.
        The point map and fermion phases are returned separately.
        """
        out={f'C{n}':self.signed_symmetry(self._rotation(360/n)) for n in (6,3,2)}
        out['mirrors']={str(a):op for a in range(0,180,30)
                        if (op:=self.signed_symmetry(self._mirror(a))) is not None}
        return out

    def _directed_edge(self,i,j):
        for e in self.edges:
            if e.i==i and e.j==j: return e,1
            if e.i==j and e.j==i: return e,-1
        return None,0

    def cycles(self, lengths=(3,4,6)):
        """Contractible elementary geometrical loops, CCW by convention.
        Winding is tracked in *primitive lattice* units; non-contractible
        PBC loops are excluded. Non-elementary cycles with enclosed sites
        or incorrect polygon area are discarded.
        """
        adj={i:set() for i in range(self.Ns)}
        for e in self.edges: adj[e.i].add(e.j);adj[e.j].add(e.i)
        def normalize(seq):
            seq=list(seq);rots=[tuple(seq[i:]+seq[:i]) for i in range(len(seq))]
            rev=seq[::-1];rots+= [tuple(rev[i:]+rev[:i]) for i in range(len(seq))]
            return min(rots)
        candidates=set()
        for k in lengths:
            for start in range(self.Ns):
                def visit(path):
                    if len(path)==k:
                        if start in adj[path[-1]]: candidates.add(normalize(path))
                        return
                    for x in adj[path[-1]]:
                        if x not in path and x>=start:visit(path+[x])
                visit([start])
        loops=[]
        expected={3:SQ3/4,4:1.,6:3*SQ3/2}
        for cyc in sorted(candidates):
            k=len(cyc)
            co=[self.positions[cyc[0]]]
            shift=np.array([0,0],dtype=int)
            for i,j in zip(cyc,cyc[1:]+cyc[:1]):
                e,sgn=self._directed_edge(i,j)
                if e is None:raise AssertionError('Broken graph')
                # canonical primitive-cell coordinate difference for hopping
                Ri=np.array(self.reps[i//6]);Rj=np.array(self.reps[j//6])
                ds=(np.array(e.delta) if sgn>0 else -np.array(e.delta)) + (Ri-Rj if sgn>0 else Ri-Rj)
                shift+=ds
                co.append(self.positions[j]+A@shift)
            if not np.all(shift==0) or np.linalg.norm(co[-1]-co[0])>1e-7: continue
            xy=np.array(co[:-1]);area=.5*np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
            if abs(abs(area)-expected.get(k,1.e100))>1e-6:continue
            if area<0:cyc=tuple([cyc[0]]+list(cyc[:0:-1]))
            loops.append(cyc)
        return loops

    def permitted_momenta(self):
        """k1,k2 in units of reciprocal primitive lattice vectors."""
        out={}
        d=abs(round(np.linalg.det(self.M)))
        for i in range(d):
            for j in range(d):
                q=np.linalg.solve(self.M.T,np.array([i,j],dtype=float))%1
                out[tuple(np.round(q,10))]=tuple(float(x) for x in q)
        assert len(out)==d
        return [out[key] for key in sorted(out)]

    def metadata(self):
        return {'name':self.name,'Ns':self.Ns,'M':self.M.tolist(),
                'reps':self.reps,'t':self.t,'primitive_vectors':A.tolist(),
                'orbitals':ORBITALS.tolist(), 'momenta':self.permitted_momenta(),
                'edges':[vars(e) for e in self.edges]}


def twisted_c6_gauge(g):
    """Gauge d_i=z_i c_i in which the t1=-t2 twisted C6 is plain C6.

    For U c_i U^-1 = s_i c_{p(i)} and a six-cycle with product(s)=-1,
    choose lambda=exp(i*pi/6) and z_{p(i)}=lambda*s_i*z_i. Then
    H'_{ij}=z_i*H_{ij}*conj(z_j) is invariant under site permutation p.
    Return (site_permutation, phases z), or None when unavailable.
    """
    signed=g.gauge_point_maps()['C6']
    if signed is None:return None
    p,signs=signed
    lam=np.exp(1j*np.pi/6)
    z=np.zeros(g.Ns,dtype=np.complex128)
    for a in range(g.Ns):
        if abs(z[a])>0:continue
        z[a]=1.
        i=a
        while True:
            j=int(p[i]);next_z=lam*signs[i]*z[i]
            if abs(z[j])>0:
                if abs(z[j]-next_z)>1.e-9:
                    raise RuntimeError('Inconsistent C6 orbit phase gauge')
                break
            z[j]=next_z
            i=j
    if not np.allclose(np.abs(z),1.,atol=1.e-9):raise AssertionError('nonunitary gauge')
    # Verify actual transformed H' rather than assuming signed symmetry alone.
    hopp={tuple(sorted((e.i,e.j))):g.t[e.kind] for e in g.edges}
    for e in g.edges:
        i,j=e.i,e.j
        ref=hopp[tuple(sorted((int(p[i]),int(p[j]))))]
        amp=z[i]*g.t[e.kind]*np.conj(z[j])
        target=z[int(p[i])]*ref*np.conj(z[int(p[j])])
        if abs(amp-target)>1.e-9:raise AssertionError('twisted gauge does not restore ordinary C6')
    return p,z