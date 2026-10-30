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

    def symmetry(self, matrix, center=POINT_CENTER):
        """Return a site permutation only when the geometric map preserves H.
        The check compares HOPPING AMPLITUDES (not bond labels) and V edges.
        """
        p=np.empty(self.Ns,dtype=np.int32)
        for i,r in enumerate(self.positions):
            q=center+matrix@(r-center)
            candidates=[]
            for j,s in enumerate(self.positions):
                d=np.linalg.solve(self.B,q-s)
                if np.max(np.abs(d-np.rint(d)))<1e-8: candidates.append(j)
            if len(candidates)!=1:return None
            p[i]=candidates[0]
        if len(set(map(int,p)))!=self.Ns:return None
        def signature(edge):
            return (min(edge.i,edge.j),max(edge.i,edge.j),
                    round(self.t[edge.kind],11),edge.kind=='t')
        before=sorted(map(signature,self.edges))
        after=sorted((min(p[e.i],p[e.j]),max(p[e.i],p[e.j]),
                      round(self.t[e.kind],11),e.kind=='t') for e in self.edges)
        return p if before==after else None

    def point_maps(self):
        """Use only generators verified against the exact PBC bond list."""
        angle=math.pi/3
        rot=lambda a:np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        axis=math.pi/6 if self.name=='3x1' else 0.
        u=np.array([math.cos(axis),math.sin(axis)])
        mirror=2*np.outer(u,u)-np.eye(2)
        r6=self.symmetry(rot(angle))
        r2=self.symmetry(rot(math.pi))
        m=self.symmetry(mirror)
        return {'C6':r6,'C2':r2,'mirror':m}

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