"""Independent fixed-N sparse ED cross-check (not QuSpin)."""
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from geometry import RubyPBC

def compute(name='ring3',Nf=6,V=1.):
 g=RubyPBC(name)
 states=np.array([sum(1<<(g.Ns-1-i) for i in occ) for occ in combinations(range(g.Ns),Nf)],dtype=np.uint64)
 states=np.sort(states)[::-1]
 loc={int(s):j for j,s in enumerate(states)}
 rows=[];cols=[];vals=[]
 for col,s0 in enumerate(states):
  s=int(s0);en=0.
  for e in g.edges:
   ii=1<<(g.Ns-1-e.i);jj=1<<(g.Ns-1-e.j)
   ni=bool(s&ii);nj=bool(s&jj)
   if e.kind=='t' and ni and nj:en+=V
   if ni==nj:continue
   if nj:
    dest=s^ii^jj
    # fermion sign due to required swap of occupations between i and j
    lo=min(e.i,e.j);hi=max(e.i,e.j)
    n_between=sum(bool(s&(1<<(g.Ns-1-k))) for k in range(lo+1,hi))
    sign=(-1)**n_between
    rows.append(loc[dest]);cols.append(col);vals.append(g.t[e.kind]*sign)
   else:
    dest=s^ii^jj
    lo=min(e.i,e.j);hi=max(e.i,e.j)
    n_between=sum(bool(s&(1<<(g.Ns-1-k))) for k in range(lo+1,hi))
    sign=(-1)**n_between
    rows.append(loc[dest]);cols.append(col);vals.append(g.t[e.kind]*sign)
  rows.append(col);cols.append(col);vals.append(en)
 H=coo_matrix((vals,(rows,cols)),shape=(len(states),len(states))).tocsr()
 assert (H-H.T).nnz==0
 evals=eigsh(H,k=4,which='SA',return_eigenvectors=False,tol=1.e-9)
 return np.sort(evals)
if __name__=='__main__':
 for x in ['3x1','ring3']:
  for v in [0.,1.]:
   es=compute(x,6,v)
   print(x,'V',v,'lowest 4',es)