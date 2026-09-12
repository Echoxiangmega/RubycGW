import json
import warnings
import numpy as np

import analyze_ed18_pbc_low_energy_modes as diag
from rubycgw.ed18 import ED18Solver
from rubycgw.model import RubyParameters


def test_ed18_pbc_low_energy_modes_diagnostic():
    solver = ED18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0), primitive_filling=2.0)
    ops, local_j = diag._operator_dictionary(solver)
    payload=[]
    for V in (1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5):
        spec=solver.solve(V,n_eigs=16,tol=1e-9,maxiter=8000,v0=None)
        E=np.asarray(spec.energies,float); vecs=np.asarray(spec.eigenvectors,complex)
        amp,w=diag._fingerprint(vecs,ops)
        labels,gw=diag._grouped_weights(w)
        mult=diag._first_excited_multiplet(E,1e-7)
        agg=np.sum(gw[mult],axis=0); ig=int(np.argmax(agg))
        wz=np.asarray(w[:,2,0,0],float)
        ic=1+int(np.argmax(wz[1:])); gaps=E-E[0]
        good=(np.arange(len(E))>0)&(gaps>1e-12)
        S=float(np.sum(wz[good])); chi=float(2*np.sum(wz[good]/gaps[good])); deff=float(2*S/chi)
        _,rows=diag._projected_chiral_states(E,vecs,local_j,4)
        rmin,rmax=rows
        states=[]
        for n in range(1,8):
            ign=int(np.argmax(gw[n]))
            states.append({"n":n,"gap":float(gaps[n]),"best":str(labels[ign]),"W":float(gw[n,ign]),"w_zG":float(wz[n])})
        payload.append({
            "V":V,"E0":float(E[0]),"gaps":[float(x) for x in gaps[:8]],
            "first_label":str(labels[ig]),"first_mult":[int(x) for x in mult.tolist()],"first_W":float(agg[ig]),
            "current_n":int(ic),"current_gap":float(gaps[ic]),"current_weight":float(wz[ic]),
            "current_S":S,"current_chi":chi,"current_deff":deff,
            "Jmax_M4":float(rmax[0]),"Jmin_M4":float(rmin[0]),
            "Emax_M4_gap":float(rmax[1]-E[0]),"sigmaE_M4":float(rmax[2]),
            "localJ_M4":[float(x) for x in rmax[3]],"prob_M4":[float(x) for x in rmax[4]],
            "states":states,
        })
    warnings.warn("ED18_PBC_LOW_MODE_RESULT "+json.dumps(payload,sort_keys=True))
    assert all(np.isfinite(x["current_gap"]) for x in payload)
