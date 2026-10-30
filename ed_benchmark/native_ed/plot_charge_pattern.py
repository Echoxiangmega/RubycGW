#!/usr/bin/env python3
"""Plot Ruby ED real-space site occupations, quantum variances and CO patterns.

Works with saved ground_states.npz; otherwise only selected V points are
re-solved. A nondegenerate finite-size state cannot be forced to have CO.
The maximally inhomogeneous density within a degenerate ground multiplet is
found by a multi-start nonlinear optimization and is labeled as such, NOT a
rigorous global optimization theorem or proof of thermodynamic order.

Windows PowerShell/CMD:
 python plot_charge_pattern.py --input results\\chi_ring3_n2 --V 1 2 3
"""
import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize

from geometry import RubyPBC
from scan_susceptibility import (pseudospin_e_charge_modes,
                                 leading_charge_mode,charge_real_sources)
from plot_current_pattern import selected_site_positions,load_scan_metadata,load_or_compute_ground
from core import FixedHamiltonian


def projected_site_densities(states,psis,ns):
    """Each M[i] = P0 n_i P0, with M Hermitian on the ground-state manifold."""
    states=np.asarray(states,dtype=np.uint64)
    psis=np.asarray(psis,dtype=complex)
    d=psis.shape[1]
    if psis.shape!=(len(states),d):raise ValueError('Wrong ground vector size')
    if not np.allclose(psis.conj().T@psis,np.eye(d),atol=1.e-7):
        raise ValueError('Ground multiplet must be orthonormal')
    occ=np.column_stack([((states >> np.uint64(ns-1-i)) & 1).astype(float)
                         for i in range(ns)])
    m=np.asarray([psis.conj().T@(occ[:,i,None]*psis) for i in range(ns)])
    m=.5*(m+np.swapaxes(m.conj(),1,2))
    return m,occ


def density_for_coeff(m,coeff):
    coeff=np.asarray(coeff,complex).reshape(-1)
    coeff=coeff/np.linalg.norm(coeff)
    return np.einsum('a,iab,b->i',coeff.conj(),m,coeff).real


def within_triangle_center(x):
    y=np.asarray(x,float).copy()
    if y.ndim!=1 or len(y)%3:raise ValueError('Sites must group into triangles')
    return (y.reshape(-1,3)-y.reshape(-1,3).mean(axis=1,keepdims=True)).reshape(-1)


def spatial_measure(density,kind='triangle'):
    n=np.asarray(density,float)
    y=within_triangle_center(n) if kind=='triangle' else n-n.mean()
    return float(np.dot(y,y)/len(n))


def optimize_quartic(m,kind='triangle',seed=20261030,restarts=12,initial=()):
    """Multi-start optimization of Var_sites(<n_i>) inside ground manifold.

    Unlike diagonalizing P O P for a FIXED O, this quartic objective is
    nonlinear in ground-state superposition coefficients. No global
    optimality claim is made. Return best stationary candidate and spread.
    """
    ns,d=m.shape[:2]
    identity=np.eye(d,dtype=complex)
    if d==1:return identity[:,0],spatial_measure(density_for_coeff(m,identity[:,0]),kind),0.
    def evaluate(x):
        z=x[:d]+1j*x[d:]
        norm=float(np.vdot(z,z).real)
        if norm<1.e-15:return 1e9,np.zeros_like(x)
        ni=np.einsum('a,iab,b->i',z.conj(),m,z).real/norm
        dev=within_triangle_center(ni) if kind=='triangle' else ni-ni.mean()
        value=float(np.dot(dev,dev)/ns)
        # Wirtinger derivative of density-variance Rayleigh quotients.
        v=np.einsum('i,iab,b->a',dev,m,z)
        term=float(np.dot(dev,ni))
        dz=(2/(ns*norm))*(v-term*z)
        grad=-2*np.r_[dz.real,dz.imag]
        return -value,grad
    rng=np.random.default_rng(seed)
    seeds=[identity[:,i] for i in range(d)]
    seeds.extend(initial)
    for _ in range(restarts):seeds.append(rng.normal(size=d)+1j*rng.normal(size=d))
    best=None;results=[]
    for z in seeds:
        z=np.asarray(z,complex).reshape(-1)
        if len(z)!=d or np.linalg.norm(z)<1.e-12:continue
        z=z/np.linalg.norm(z)
        x=np.r_[z.real,z.imag]
        res=minimize(evaluate,x,jac=True,method='L-BFGS-B',
                     options={'maxiter':600,'ftol':1e-13,'gtol':5e-11})
        value=-float(evaluate(res.x)[0]);results.append(value)
        if best is None or value>best[0]:
            z=res.x[:d]+1j*res.x[d:]
            best=(value,z/np.linalg.norm(z))
    return best[1],best[0],float(max(results)-min(results))


def projected_charge_candidates(g,m,occ,psis):
    """Diagonalize P0 O P0 for fixed HF charge and adaptive-leading sources."""
    d=psis.shape[1]
    pop=np.mean(np.abs(psis)**2,axis=1)
    nbar=pop@occ
    weights=pseudospin_e_charge_modes(g)
    leading=leading_charge_mode(occ,psis,g)
    for i,z in enumerate(charge_real_sources(leading['site_vector'])):
        weights['CO_leading_'+('real' if i==0 else 'imag')]=np.asarray(z,float)
    proposals=[]
    for name,w in weights.items():
        w=np.asarray(w,float)
        O=np.einsum('i,iab->ab',w,m)
        O=.5*(O+O.conj().T)
        zvals,zvecs=np.linalg.eigh(O)
        ensemble_value=float(np.dot(w,nbar))
        for j in (0,d-1):
            c=zvecs[:,j]
            density=density_for_coeff(m,c)
            proposals.append({'name':name,'sign':'minimum' if j==0 else 'maximum',
                 'vector':c,'O_value':float(zvals[j]),
                 'O_centered':float(zvals[j]-ensemble_value),
                 'spatial_triangle_variance':spatial_measure(density,'triangle'),
                 'spatial_global_variance':spatial_measure(density,'global')})
    return proposals,leading


def optimized_charge_state(g,states,psis,E0=None,V=None,kind='triangle',seed=20261030,
                           restarts=12,check_h=True,source='auto',polarity='positive'):
    """Compare TR-symmetric ground ensemble with a CO-polarized PURE ground state."""
    m,occ=projected_site_densities(states,psis,g.Ns)
    ensemble_density=np.trace(m,axis1=1,axis2=2).real/psis.shape[1]
    p=np.mean(abs(psis)**2,axis=1)
    nn=occ.T@(p[:,None]*occ)
    cov=nn-np.outer(ensemble_density,ensemble_density)
    proposals,leading=projected_charge_candidates(g,m,occ,psis)
    init=[v['vector'] for v in proposals]
    if source=='auto':
        coeff,quartic,spread=optimize_quartic(m,kind,seed,restarts,init)
        selection=f'numerical maximum of {kind} density spatial variance'
    else:
        candidates=[x for x in proposals if x['name']==source and
                    x['sign']==('maximum' if polarity=='positive' else 'minimum')]
        if not candidates:raise ValueError(f'Unknown or unavailable source {source}: {polarity}')
        coeff=candidates[0]['vector']
        quartic=spatial_measure(density_for_coeff(m,coeff),kind)
        spread=0.0
        selection=f'{polarity} eigenvector of P0 {source} P0'
    psi=psis@coeff;psi/=np.linalg.norm(psi)
    density=density_for_coeff(m,coeff)
    mean0=float(np.mean(ensemble_density));mean1=float(np.mean(density))
    n_expected=int(states[0]).bit_count()
    if abs(sum(density)-n_expected)>1e-8 or abs(sum(ensemble_density)-n_expected)>1e-8:
        raise AssertionError('Fixed-N density sum rule failed')
    onsite_ensemble=ensemble_density*(1-ensemble_density)
    onsite_opt=density*(1-density)
    assert np.max(np.abs(np.imag(np.einsum('a,iab,b->i',coeff.conj(),m,coeff))))<1e-9
    residual=None
    if check_h and E0 is not None and V is not None:
        H=FixedHamiltonian(g,n_expected,V,states=states).operator
        residual=float(np.linalg.norm(H@psi-E0*psi))
        if residual>2e-6: warnings.warn(f'Selected state not eigenstate at requested tolerance: {residual:g}')
    # A simple TR-related state has identical density (but potentially reversed J).
    psi_tr=psi.conj()
    # A gauge-independent, site-diagonal probability checks this directly.
    tr_density=(np.abs(psi_tr)**2)@occ
    assert np.allclose(density,tr_density,atol=1e-9)
    per_tri=[]
    for j in range(0,g.Ns,3):
        n=density[j:j+3]; n0=ensemble_density[j:j+3]
        A=(j%6==0)
        tx=2*n[0]-n[1]-n[2]
        ty=np.sqrt(3)*(n[2]-n[1])*(1 if A else -1)
        per_tri.append({'type':'A' if A else 'B','sites':[j,j+1,j+2],
           'density':n.tolist(),'ensemble_density':n0.tolist(),
           'n_total':float(n.sum()),'tau_x':float(tx),'tau_y':float(ty),
           'pseudospin_magnitude':float(np.hypot(tx,ty)),
           'triangle_site_variance':float(np.var(n))})
    return {'psi':psi,'psi_TR':psi_tr,'coeff':coeff,'ground_dim':psis.shape[1],
       'm':m,'n_ensemble':ensemble_density,'n_selected':density,
       'var_n_ensemble':onsite_ensemble,'var_n_selected':onsite_opt,
       'covariance_ensemble':cov,'triangles':per_tri,'proposals':proposals,
       'leading':leading,'optimization_kind':kind,'quartic_best':quartic,
       'quartic_restart_spread':spread,'E0':E0,'ground_residual':residual,
       'ensemble_global_spatial_var':spatial_measure(ensemble_density,'global'),
       'selected_global_spatial_var':spatial_measure(density,'global'),
       'ensemble_tri_spatial_var':spatial_measure(ensemble_density,'triangle'),
       'selected_tri_spatial_var':spatial_measure(density,'triangle'),
       'mean_onsite_var_ensemble':float(onsite_ensemble.mean()),
       'mean_onsite_var_selected':float(onsite_opt.mean()),
       'leading_charge_cov_eigenvalue':float(np.linalg.eigvalsh(cov)[-1]),
       'leading_charge_structure_factor':leading['structure_factor'],
       'lead_q':leading['q'],'selection':selection,'selection_source':source}


def plot_density_contrast(g,dat,path):
    """Diverging map makes small charge order visible without hiding its scale."""
    coords,ends,boundary,_,_,_=selected_site_positions(g)
    delta=dat['n_selected']-dat['n_ensemble']
    vmax=max(1e-8,float(np.max(abs(delta))))
    fig,ax=plt.subplots(figsize=(10.5,8))
    draw_lattice(ax,g,coords,ends,boundary)
    cmap=plt.get_cmap('RdBu_r');norm=Normalize(-vmax,vmax)
    for i,(p,z) in enumerate(zip(coords,delta)):
        ax.scatter(*p,c=[z],vmin=-vmax,vmax=vmax,cmap=cmap,s=170,
                   edgecolors='#253B53',lw=.9,zorder=3)
        ax.text(p[0],p[1]-.23,f'{z:+.3f}',fontsize=8,ha='center',zorder=4)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.68,
                 label='<n_i> selected - <n_i> symmetric ground ensemble')
    ax.set_title('Site charge polarization in the selected ground state')
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=200,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)


def plot_covariance(dat,path):
    C=np.asarray(dat['covariance_ensemble'])
    ns=C.shape[0]
    vmax=max(float(np.max(abs(C))),1e-10)
    fig,ax=plt.subplots(figsize=(7.3,6.4))
    im=ax.imshow(C,vmin=-vmax,vmax=vmax,cmap='RdBu_r',interpolation='nearest')
    ax.set_xlabel('site j');ax.set_ylabel('site i')
    ax.set_title('Connected charge covariance of full ground-state ensemble')
    fig.colorbar(im,ax=ax,shrink=.78,label='<n_i n_j> - <n_i><n_j>')
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=190,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)


def draw_lattice(ax,g,coords,ends,boundary):
    from plot_current_pattern import CLASS_COLOR
    ax.add_patch(patches.Polygon(boundary,closed=True,facecolor='#EFF4F9',
          edgecolor='#3E5269',linewidth=1.5,linestyle='--',alpha=.75))
    for e,dst in zip(g.edges,ends):
        start=coords[e.i]
        wrap=np.linalg.norm(dst-coords[e.j])>1e-7
        ax.plot([start[0],dst[0]],[start[1],dst[1]],
                color=CLASS_COLOR[e.kind],lw=1.4,alpha=.35,ls='--' if wrap else '-')
    allpts=np.vstack((coords,ends,boundary))
    lo=allpts.min(axis=0);hi=allpts.max(axis=0)
    ax.set_xlim(lo[0]-.6,hi[0]+.6);ax.set_ylim(lo[1]-.6,hi[1]+.6)
    ax.set_aspect('equal');ax.axis('off')


def plot_site_panels(g,dat,path,field='density'):
    coords,ends,boundary,_,_,_=selected_site_positions(g)
    if field=='density':
        A,B=dat['n_ensemble'],dat['n_selected'];lo,hi=0.,1.
        subtitle='Site density <n_i> (same scale in both panels)'
    else:
        A,B=dat['var_n_ensemble'],dat['var_n_selected'];lo,hi=0.,.25
        subtitle='Local quantum variance Var(n_i)=<n_i>(1-<n_i>)'
    fig,axes=plt.subplots(1,2,figsize=(16,7.1))
    cmap=plt.get_cmap('viridis');norm=Normalize(lo,hi)
    for ax,arr,title in zip(axes,[A,B],['Equal mixture over ground manifold',f'Selected pure ground state: {dat["selection_source"]}']):
        draw_lattice(ax,g,coords,ends,boundary)
        for i,(r,n) in enumerate(zip(coords,arr)):
            size=(65+185*(dat['n_ensemble'][i] if field=='density' and ax==axes[0] else
                            dat['n_selected'][i] if field=='density' else n/0.25))
            ax.scatter(*r,c=[n],vmin=lo,vmax=hi,cmap=cmap,s=size,edgecolors='#19364F',lw=.9,zorder=3)
            ax.text(r[0],r[1]-.19,f'{n:.2f}',ha='center',va='top',fontsize=8,color='#172A41',zorder=4)
        ax.set_title(title,fontsize=13)
    cax=fig.add_axes([.915,.23,.012,.53])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,label=subtitle)
    fig.suptitle(f'{g.name} | Nf={int(round(sum(dat["n_selected"])))}, V={dat["V"]:+g} | ground d={dat["ground_dim"]}',fontsize=14)
    fig.subplots_adjust(left=.02,right=.89,top=.85,bottom=.05,wspace=.06)
    fig.savefig(str(path)+'.png',dpi=195,bbox_inches='tight');fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)


def plot_triangle_contrast(g,dat,path):
    coords,ends,boundary,_,_,_=selected_site_positions(g)
    fig,ax=plt.subplots(figsize=(10,8))
    draw_lattice(ax,g,coords,ends,boundary)
    from matplotlib.colors import TwoSlopeNorm
    mag=max((abs(tri['tau_x']) for tri in dat['triangles']),default=0.)
    scale=max(mag,1e-8)
    cmap=plt.get_cmap('RdBu_r')
    for tri in dat['triangles']:
        v=np.array([coords[i] for i in tri['sites']]);center=v.mean(axis=0)
        ax.add_patch(patches.Polygon(v,closed=True,facecolor=cmap(.5+.5*tri['tau_x']/scale),
                    edgecolor='#435B72',linewidth=1.1,alpha=.7,zorder=2))
        ax.text(center[0],center[1],f'{tri["type"]}  ({tri["tau_x"]:+.2f},{tri["tau_y"]:+.2f})',
                fontsize=8.8,ha='center',va='center',color='#172A41',zorder=4)
    ax.set_title('Triangle E-type charge contrast (physical tau_x,tau_y)',fontsize=13)
    fig.savefig(str(path)+'.png',dpi=220,bbox_inches='tight');fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)


def export_one(g,states,psis,E0,V,folder,kind='triangle',check_h=True,restarts=12,
               source='auto',polarity='positive'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    dat=optimized_charge_state(g,states,psis,E0,V,kind,restarts=restarts,
                               check_h=check_h,source=source,polarity=polarity)
    dat['V']=float(V)
    plot_site_panels(g,dat,folder/'site_density','density')
    plot_density_contrast(g,dat,folder/'density_contrast')
    plot_covariance(dat,folder/'charge_covariance')
    plot_site_panels(g,dat,folder/'onsite_quantum_variance','variance')
    plot_triangle_contrast(g,dat,folder/'triangle_charge')
    coords,_,_,_,_,_=selected_site_positions(g)
    with (folder/'site_density.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['site','cell','orbital','x','y','n_ensemble','n_selected',
                    'var_n_ensemble','var_n_selected','n_difference'])
        for i in range(g.Ns):w.writerow([i,i//6,i%6,*coords[i],
            dat['n_ensemble'][i],dat['n_selected'][i],
            dat['var_n_ensemble'][i],dat['var_n_selected'][i],
            dat['n_selected'][i]-dat['n_ensemble'][i]])
    with (folder/'projected_charge_sources.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['charge_source','extremum','projected_O_eigenvalue',
                    'deviation_from_ensemble','spatial_triangle_variance',
                    'spatial_global_variance'])
        for pr in dat['proposals']:
            w.writerow([pr['name'],pr['sign'],pr['O_value'],pr['O_centered'],
                        pr['spatial_triangle_variance'],pr['spatial_global_variance']])
    with (folder/'triangle_charge.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['type','sites','n_total','tau_x','tau_y','tau_abs',
                    'triangle_site_variance','n0','n1','n2'])
        for t in dat['triangles']:w.writerow([t['type'],' '.join(map(str,t['sites'])),
                t['n_total'],t['tau_x'],t['tau_y'],t['pseudospin_magnitude'],
                t['triangle_site_variance'],*t['density']])
    np.savez_compressed(folder/'charge_pattern.npz',states=np.asarray(states),
       psi_selected=dat['psi'],ground_coefficients=dat['coeff'],
       n_ensemble=dat['n_ensemble'],n_selected=dat['n_selected'],
       quantum_var_ensemble=dat['var_n_ensemble'],quantum_var_selected=dat['var_n_selected'],
       charge_covariance_ensemble=dat['covariance_ensemble'],
       projected_site_density_matrices=dat['m'])
    meta={key:dat[key] for key in ('ground_dim','optimization_kind','quartic_best',
           'quartic_restart_spread','ground_residual','selection','selection_source',
           'ensemble_global_spatial_var',
           'selected_global_spatial_var','ensemble_tri_spatial_var','selected_tri_spatial_var',
           'mean_onsite_var_ensemble','mean_onsite_var_selected',
           'leading_charge_cov_eigenvalue','leading_charge_structure_factor','lead_q')}
    meta.update(cluster=g.name,Nf=int(states[0]).bit_count(),V=float(V),
                hopping=g.t,triangle_pseudospins=[{'sites':x['sites'],
                     'tau_x':x['tau_x'],'tau_y':x['tau_y']} for x in dat['triangles']],
                note='Onsite Var(n_i)=n_i*(1-n_i) does NOT diagnose CO by itself. '
                     'Spatial variance of <n_i> and ground-manifold polarization are separate. '
                     'Multistart nonlinear optimum is numerical, not a proof of global maximality.')
    (folder/'charge_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'V={V:g} d={dat["ground_dim"]}: ensemble triangle variance={dat["ensemble_tri_spatial_var"]:.6g} '
          f'best pure={dat["selected_tri_spatial_var"]:.6g}, '
          f'mean local quantum variance={dat["mean_onsite_var_selected"]:.6g} -> {folder}',flush=True)
    return meta


def plot_variance_scan(records,output):
    """Plot two genuinely different notions of charge variance versus V."""
    records=sorted(records,key=lambda row:row['V'])
    x=np.array([r['V'] for r in records],float)
    degeneracy=np.array([r['ground_dim']>1 for r in records],bool)
    def line(ax,values,label):
        y=np.asarray(values,float)
        color=ax._get_lines.get_next_color()
        for i in range(len(x)-1):
            style='--' if degeneracy[i] or degeneracy[i+1] else '-'
            if np.isfinite(y[i]) and np.isfinite(y[i+1]):
                ax.plot(x[i:i+2],y[i:i+2],color=color,lw=1.8,ls=style)
        ax.scatter(x[~degeneracy],y[~degeneracy],color=color,s=33,zorder=4)
        ax.scatter(x[degeneracy],y[degeneracy],facecolors='white',edgecolors=color,s=55,zorder=5,lw=1.5)
        ax.plot([],[],color=color,marker='o',label=label)
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for col,label in [('ensemble_tri_spatial_var','triangle spatial: symmetric ensemble'),
                      ('selected_tri_spatial_var','triangle spatial: optimized pure state'),
                      ('ensemble_global_spatial_var','global spatial: symmetric ensemble'),
                      ('selected_global_spatial_var','global spatial: optimized pure state')]:
        line(ax,[r[col] for r in records],label)
    ax.set_xlabel('V / t');ax.set_ylabel('Variance over site expectation densities')
    ax.set_title('Charge disproportionation (not local quantum variance)')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(str(output/'charge_variance_vs_V.png'),dpi=190)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for col,label in [('mean_onsite_var_ensemble','Mean Var(n_i): ground ensemble'),
                      ('mean_onsite_var_selected','Mean Var(n_i): optimized pure state')]:
        line(ax,[r[col] for r in records],label)
    ax.set_xlabel('V / t');ax.set_ylabel('Mean [<n_i>(1-<n_i>)]')
    ax.set_title('Local quantum occupation fluctuation (not proof of charge order)')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(str(output/'onsite_variance_vs_V.png'),dpi=190)
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='A susceptibility scan root, or V subfolder')
    p.add_argument('--V',type=float,nargs='+',default=None)
    p.add_argument('--output',default=None)
    p.add_argument('--kind',choices=['triangle','global'],default='triangle',
                   help='Maximize intra-triangle density disproportionation or global spatial variance')
    p.add_argument('--restarts',type=int,default=12,help='Number of random optimizer starting vectors')
    p.add_argument('--source',choices=['auto','CO_E_even_x','CO_E_even_y',
                  'CO_E_odd_x','CO_E_odd_y','CO_leading_real','CO_leading_imag'],
                   default='auto',help='auto: maximize density variance; otherwise exact projected source eigenstate')
    p.add_argument('--polarity',choices=['positive','negative'],default='positive',
                   help='For a fixed source, choose its largest or smallest P0 O P0 eigenvalue')
    p.add_argument('--force-resolve',action='store_true')
    p.add_argument('--no-check-H',action='store_true')
    p.add_argument('--low',type=int,default=None)
    p.add_argument('--max-basis',type=int,default=70000)
    args=p.parse_args(argv)
    if args.restarts<0:raise ValueError('--restarts must be >=0')
    root=Path(args.input)
    Vlist=[None] if (root/'responses.json').exists() else args.V
    if not Vlist:raise ValueError('Provide one or more --V points')
    dest=Path(args.output) if args.output else (root if (root/'scan_metadata.json').exists() else root.parent)/'charge_patterns'
    dest.mkdir(parents=True,exist_ok=True)
    # Output depends on the explicit pure-state selection prescription.
    marker=dest/'charge_selection.json'
    identity={'kind':args.kind,'source':args.source,'polarity':args.polarity,
              'restarts':args.restarts}
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Output directory was generated using a different charge selection; choose --output')
    marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    cache=None;records=[]
    for V in Vlist:
        folder,info,val,row=load_scan_metadata(root,V)
        info=dict(info,max_basis=args.max_basis)
        if args.low is not None:info['low']=args.low
        g,states,psis,e0,source,cache=load_or_compute_ground(folder,info,val,cache,
                                   force_resolve=args.force_resolve)
        if row and int(row.get('ground_dim',psis.shape[1]))!=psis.shape[1]:
            raise ValueError(f'ground_dim mismatch at V={val}; adjust --low/degeneracy-tol')
        record=export_one(g,states,psis,e0,val,dest/f'V_{val:+.9f}',kind=args.kind,
                          check_h=not args.no_check_H,restarts=args.restarts,
                          source=args.source,polarity=args.polarity)
        record['state_source']=source;records.append(record)
    keys=['V','Nf','ground_dim','ensemble_global_spatial_var','selected_global_spatial_var',
          'ensemble_tri_spatial_var','selected_tri_spatial_var','mean_onsite_var_ensemble',
          'mean_onsite_var_selected','ground_residual','state_source']
    with (dest/'overview.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for x in records:wr.writerow({key:x.get(key) for key in keys})
    plot_variance_scan(records,dest)
    return records

if __name__=='__main__':main()