"""Momentum-resolved exact ED density structure factors for Ruby PBC clusters.

The equal-ground-manifold ensemble is used. A structure-factor eigenvector is
an orbital fluctuation pattern, NOT a nonzero one-point density order parameter.
Fourier convention: F[R,a;b](q) = delta(a,b) exp(+2pi*i*q.R)/sqrt(Nc).
S(q) = F(q)^dagger C F(q), C_ij=<n_i n_j>-<n_i><n_j>.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from geometry import RubyPBC
from plot_current_pattern import selected_site_positions


def fourier_orbital_matrix(g,q):
    q=np.asarray(q,float)
    F=np.zeros((g.Ns,6),complex)
    for cell,R in enumerate(g.reps):
        p=np.exp(2j*np.pi*float(np.dot(q,np.asarray(R,float))))/np.sqrt(g.num_cells)
        F[6*cell:6*cell+6,:]=p*np.eye(6)
    return F


def choose_eigenvector_phase(z):
    z=np.asarray(z,complex).copy()
    k=int(np.argmax(np.abs(z)))
    if abs(z[k])>1e-14:z*=np.exp(-1j*np.angle(z[k]))
    return z


def triangle_charge_content(orbital_vector):
    """Irrep-inspired decomposition of a six-orbital charge fluctuation.

    This is a diagnostic of the microscopic *density* pattern, not a projected
    spin matrix. E_A/E_B are the two triangle-internal density contrasts;
    total_A/total_B describe fluctuations in triangle total occupations.
    The six real basis vectors form an orthonormal basis in C^6.
    """
    a=np.asarray(orbital_vector,complex)
    if a.shape!=(6,):raise ValueError('Six orbital weights required')
    rt=np.sqrt(3.)
    basis=np.array([[1,1,1,0,0,0], [0,0,0,1,1,1],
         [2,-1,-1,0,0,0], [0,-1,1,0,0,0],
         [0,0,0,2,-1,-1], [0,0,0,0,1,-1]],float)
    basis[0:2]/=rt
    basis[2]/=np.sqrt(6);basis[3]/=np.sqrt(2)
    basis[4]/=np.sqrt(6);basis[5]/=np.sqrt(2)
    weight=abs(basis@a)**2
    weight/=max(1e-15,float(np.vdot(a,a).real))
    labels=('total_A','total_B','E_Ax','E_Ay','E_Bx','E_By')
    out={label:float(v) for label,v in zip(labels,weight)}
    out['E_internal_total']=float(sum(weight[2:]))
    out['triangle_total']=float(sum(weight[:2]))
    assert abs(sum(weight)-1.)<1e-8
    return out


def structure_factor_all(g,C,tol=1e-8):
    """All allowed q, all six eigenmodes. q and -q partners retained.

    Data arrays: nq x 6 x 6 matrices and eigenvectors; eigenvalues ascending.
    No finite q is discarded even if its charge mode has small weight.
    """
    C=np.asarray(C,complex)
    if C.shape!=(g.Ns,g.Ns):raise ValueError('Expected full Ns x Ns covariance')
    if np.linalg.norm(C-C.conj().T)>tol*max(1.,np.linalg.norm(C)):
        raise AssertionError('Site covariance not Hermitian')
    qs=np.asarray(g.permitted_momenta(),float)
    matrices=[];values=[];vectors=[];real_patterns=[];complex_patterns=[]
    for q in qs:
        F=fourier_orbital_matrix(g,q)
        mat=F.conj().T@C@F
        mat=.5*(mat+mat.conj().T)
        w,U=np.linalg.eigh(mat)
        if np.min(w)<-tol:
            raise AssertionError(f'Negative density covariance eigenvalue at q={q}: {w}')
        for j in range(6):U[:,j]=choose_eigenvector_phase(U[:,j])
        v=F@U[:,-1]
        template=np.sqrt(2.)*np.real(v) if not np.allclose(q,0.,atol=1e-9) else np.real(v)
        norm=np.linalg.norm(template)
        if norm<1e-11:template=np.imag(v);norm=np.linalg.norm(template)
        if norm>0:template/=norm
        matrices.append(mat);values.append(w);vectors.append(U)
        real_patterns.append(template)
        complex_patterns.append(v)
    return {'momenta':qs,'S_q':np.array(matrices),
            'eigenvalues':np.array(values),'eigenvectors':np.array(vectors),
            'leading_real_template':np.array(real_patterns),
            'leading_complex_mode':np.array(complex_patterns)}


def ground_ensemble_charge(states,psis,ns):
    """Basis-independent covariance of the maximally mixed ground multiplet."""
    states=np.asarray(states,dtype=np.uint64)
    psis=np.asarray(psis,complex)
    if psis.ndim!=2 or psis.shape[0]!=len(states):raise ValueError('Bad ground multiplet shape')
    d=psis.shape[1]
    if not np.allclose(psis.conj().T@psis,np.eye(d),atol=1e-8):
        raise ValueError('Ground basis must be orthonormal')
    occ=np.column_stack([((states>>np.uint64(ns-1-i))&np.uint64(1)).astype(float)
                          for i in range(ns)])
    p=np.sum(abs(psis)**2,axis=1)/d
    density=p@occ
    covariance=occ.T@(p[:,None]*occ)-np.outer(density,density)
    return density,covariance


def mode_reals(z):
    """Unit real-space density sources spanning Re/Im(z), with Gram-Schmidt."""
    z=np.asarray(z,complex)
    out=[]
    for x in (np.real(z),np.imag(z)):
        v=x.copy()
        for prev in out:v-=np.dot(v,prev)*prev
        n=np.linalg.norm(v)
        if n>1e-10:out.append(v/n)
    return out


def project_polarized_density(projected_site_matrices,complex_mode,ensemble):
    """Extremize a leading Fourier density source *within* ground multiplet.

    Returns pure-state site densities if available; never fabricates static CO
    for nondegenerate ground states. Global mode phase arbitrary; scan real
    quadratures consistently and report maximum absolute ensemble deviation.
    """
    if projected_site_matrices is None:
        return np.asarray(ensemble,float),{'available':False,'reason':'No saved/solved ground multiplet'}
    M=np.asarray(projected_site_matrices,complex)
    d=M.shape[1]
    if M.shape[0]!=len(ensemble) or M.shape[2]!=d:raise ValueError('Invalid projected density matrices')
    if d==1:
        return np.asarray(ensemble,float),{'available':True,'ground_dim':1,'contrast':0.,'reason':'Unique ground state cannot be polarized by mixing'}
    best=None
    for component,weights in enumerate(mode_reals(complex_mode)):
        O=np.einsum('i,iab->ab',weights,M)
        O=.5*(O+O.conj().T)
        w,U=np.linalg.eigh(O)
        ens=float(np.dot(weights,ensemble))
        for label,j in (('minimum',0),('maximum',-1)):
            contrast=abs(w[j]-ens)
            if best is None or contrast>best[0]:
                vec=U[:,j]
                density=np.einsum('a,iab,b->i',vec.conj(),M,vec).real
                best=(float(contrast),density,dict(component='real' if component==0 else 'imag',
                    extremum=label,source_eigenvalue=float(w[j]),source_shift=float(w[j]-ens),
                    coefficients=[[float(z.real),float(z.imag)] for z in vec]))
    if best is None:raise AssertionError('No real density source')
    meta=dict(best[2],available=True,ground_dim=d,contrast=best[0])
    return best[1],meta


def _q_label(q):
    return f'({q[0]:.4g}, {q[1]:.4g})'


def _draw_site_pattern(g,values,out,title,vmin=None,vmax=None,mode=False):
    xy,ends,boundary,wrap,winds,cells=selected_site_positions(g)
    arr=np.asarray(values,float)
    fig,ax=plt.subplots(figsize=(10,8))
    from plot_current_pattern import CLASS_COLOR
    for e,endpoint,cross in zip(g.edges,ends,wrap):
        ax.plot([xy[e.i,0],endpoint[0]],[xy[e.i,1],endpoint[1]],
                color=CLASS_COLOR[e.kind] if not mode else '#9CAABD',
                ls='--' if cross else '-',lw=1.25,alpha=.58,zorder=1)
    ax.add_patch(plt.Polygon(boundary,closed=True,fill=False,ec='#203B56',lw=2.,ls='--'))
    if mode:
        norm=TwoSlopeNorm(vcenter=0.,vmin=-max(1e-10,float(np.max(abs(arr)))),
                          vmax=max(1e-10,float(np.max(abs(arr)))))
        scatter=ax.scatter(xy[:,0],xy[:,1],c=arr,cmap='RdBu_r',norm=norm,
                           s=230,edgecolors='#172E47',linewidths=.8,zorder=4)
    else:
        scatter=ax.scatter(xy[:,0],xy[:,1],c=arr,cmap='viridis',vmin=vmin,vmax=vmax,
                           s=230,edgecolors='#172E47',linewidths=.8,zorder=4)
    for i,p in enumerate(xy):
        ax.annotate(f'{i%6}',(p[0],p[1]),color='white' if mode else '#102337',
                    ha='center',va='center',fontsize=8.2,fontweight='bold',zorder=5)
    ax.set_title(title,fontsize=13)
    ax.axis('equal');ax.axis('off');fig.colorbar(scatter,ax=ax,shrink=.62)
    fig.savefig(str(out)+'.png',dpi=190,bbox_inches='tight');fig.savefig(str(out)+'.svg',bbox_inches='tight')
    plt.close(fig)


def plot_q_matrices(g,result,dest):
    qs=result['momenta'];S=result['S_q'];ev=result['eigenvalues'];
    for k,q in enumerate(qs):
        qdir=dest/f'q_{k:02d}'
        qdir.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(10,4.1))
        ax=axes[0]
        im=ax.imshow(S[k].real,cmap='RdBu_r',vmin=-np.max(abs(S[k].real))-1e-12,
                     vmax=np.max(abs(S[k].real))+1e-12)
        ax.set_title('Re S_ab(q)');ax.set_xlabel('orbital b');ax.set_ylabel('orbital a');fig.colorbar(im,ax=ax,shrink=.78)
        ax=axes[1]
        im=ax.imshow(S[k].imag,cmap='RdBu_r',vmin=-np.max(abs(S[k].imag))-1e-12,
                     vmax=np.max(abs(S[k].imag))+1e-12)
        ax.set_title('Im S_ab(q)');ax.set_xlabel('orbital b');ax.set_ylabel('orbital a');fig.colorbar(im,ax=ax,shrink=.78)
        fig.suptitle('Charge S(q), q='+_q_label(q));fig.tight_layout()
        fig.savefig(qdir/'S_matrix.png',dpi=175);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4.2))
        ax.plot(np.arange(1,7),ev[k,::-1],'o-')
        ax.set_xlabel('Eigenmode rank (descending)');ax.set_ylabel('S(q) eigenvalue')
        ax.set_title('Charge eigenmodes at q='+_q_label(q));ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(qdir/'eigenvalues.png',dpi=180);plt.close(fig)
        for mode in (False,):
            _draw_site_pattern(g,result['leading_real_template'][k],qdir/'leading_template',
                  'Leading charge fluctuation template at q='+_q_label(q),mode=True)


def export_structure_one(g,covariance,ensemble_density,V,dest,projected_density=None):
    """Save exact connected S(q), eigenmodes, and optional polarized density."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    S=structure_factor_all(g,covariance)
    qidx=int(np.argmax(S['eigenvalues'][:,-1]))
    q=S['momenta'][qidx]
    n_selected,sel=project_polarized_density(projected_density,
                            S['leading_complex_mode'][qidx],ensemble_density)
    np.savez_compressed(dest/'charge_structure.npz',momenta=S['momenta'],S_q=S['S_q'],
               eigenvalues=S['eigenvalues'],eigenvectors=S['eigenvectors'],
               leading_complex_mode=S['leading_complex_mode'],
               leading_real_template=S['leading_real_template'],
               n_ensemble=np.asarray(ensemble_density,float),n_selected=n_selected,
               covariance=np.asarray(covariance,float),
               top_q_index=qidx,
               definition='S(q)=F(q)^dagger[<n_i n_j>-<n_i><n_j>]F(q), F=exp(+i2pi q.R)/sqrt(Nc)')
    with (dest/'charge_structure_modes.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['q_index','q1','q2','rank_desc','eigenvalue'])
        for k,p in enumerate(S['momenta']):
            for rank,lam in enumerate(S['eigenvalues'][k,::-1]):w.writerow([k,p[0],p[1],rank+1,lam])
    meta={'cluster':g.name,'V':float(V),'Nf':float(np.sum(ensemble_density)),
          'ground_dim':None if projected_density is None else int(projected_density.shape[1]),
          'leading_q_index':qidx,'leading_q':q.tolist(),
          'leading_eigenvalue':float(S['eigenvalues'][qidx,-1]),
          'polarization':sel,'note':'S(q) probes fluctuations. Real-space eigenmode is a template, '
                 'NOT the expectation density of an ED eigenstate. Selected density comes ONLY from '
                 'an actual state inside the degenerate ground manifold.'}
    (dest/'charge_structure_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    with (dest/'charge_orbital_character.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh)
        columns=['q_index','q1','q2','Smax','total_A','total_B','E_Ax','E_Ay',
                 'E_Bx','E_By','E_internal_total','triangle_total']
        w.writerow(columns)
        for k,qval in enumerate(S['momenta']):
            char=triangle_charge_content(S['eigenvectors'][k,:,-1])
            w.writerow([k,*qval,S['eigenvalues'][k,-1],
                        *[char[key] for key in columns[4:]]])
    meta['leading_character']=triangle_charge_content(S['eigenvectors'][qidx,:,-1])
    (dest/'charge_structure_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    plot_q_matrices(g,S,dest)
    lo=float(min(np.min(ensemble_density),np.min(n_selected)))
    hi=float(max(np.max(ensemble_density),np.max(n_selected)))
    _draw_site_pattern(g,ensemble_density,dest/'ensemble_density',
                       'Ground-manifold average site density',vmin=lo,vmax=hi)
    description=('Pure ground-state density selected by q='+_q_label(q)
                 if sel.get('available') and sel.get('ground_dim',1)>1
                 else 'Ground-state density (no distinct polarized state available)')
    _draw_site_pattern(g,n_selected,dest/'selected_density',
                       description,vmin=lo,vmax=hi)
    return meta


def plot_scan_summary(records,dest):
    if not records:return
    records=sorted(records,key=lambda x:x['V'])
    qs=np.asarray(records[0]['momenta'],float)
    V=np.array([r['V'] for r in records]);values=np.array([r['eigmax_each_q'] for r in records]);
    fig,ax=plt.subplots(figsize=(8.2,5))
    for k,q in enumerate(qs):
        ax.plot(V,values[:,k],'-o',label='q='+_q_label(q))
    ax.set_title('Momentum-resolved charge structure factors')
    ax.set_xlabel('V/t');ax.set_ylabel('largest eigenvalue of S(q)')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(dest/'charge_Sq_vs_V.png',dpi=185);plt.close(fig)
    with (dest/'charge_Sq_vs_V.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['V']+[f'Smax_q_{k}' for k in range(len(qs))]+['q_leading_index','q1_leading','q2_leading'])
        for r in records:w.writerow([r['V'],*r['eigmax_each_q'],r['leading_q_index'],*r['leading_q']])