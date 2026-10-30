#!/usr/bin/env python3
"""Plot *physical bond currents* from symmetry-selected Ruby ED ground states.

Use a fixed-N ground multiplet, diagonalize P0 O_LC P0, and draw every
independent PBC hopping current on the displayed Ruby lattice. For ring3,
choose periodic images so six entire triangles enclose a real hexagon.

No source field is added. Eigenvectors from a numerically near-degenerate
manifold are only stationary to the indicated energy tolerance.

Windows PowerShell/CMD (one line):
  python plot_current_pattern.py --input results\\chi_ring3_n2 --V 1 2 3 --channel opposite

If ground_states.npz was not saved, only those selected V points are solved
again; the input susceptibility scan is left unchanged.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from geometry import A, ORBITALS, POINT_CENTER, RubyPBC
from core import edge_triplets, make_states, FixedHamiltonian
from scan_susceptibility import VScanSolver, projected_current_manifold, lc_triangle_channels, unique_dir

CLASS_COLOR={'t':'#C84D59','t1':'#327CC4','t2':'#21927D'}
RING_DISPLAY={
    0: {'A':(1,-1), 'B':(0,0)},
    1: {'A':(1,0),  'B':(1,0)},
    2: {'A':(0,-1), 'B':(0,-1)},
}


def selected_site_positions(g):
    """Return physical-image coordinates, display cell and PBC fundamental boundary."""
    cells=[]
    for site in range(g.Ns):
        ci,orb=divmod(site,6)
        if g.name=='ring3':
            cell=RING_DISPLAY[ci]['A' if orb<3 else 'B']
            cj,_=g.reduce(cell)
            if cj!=ci:raise AssertionError('Wrong ring3 displayed image')
        else:
            cell=g.reps[ci]
        cells.append(np.asarray(cell,dtype=int))
    xy=np.array([A@cell+ORBITALS[i%6] for i,cell in enumerate(cells)])
    if g.name=='ring3':
        # Wigner-Seitz hexagon around a physical six-sided plaquette.
        from scipy.spatial import Voronoi
        center=POINT_CENTER
        sites=[center+g.B@np.array([u,v]) for u in range(-2,3)
               for v in range(-2,3) if (u,v)!=(0,0)]
        sites.append(center)
        vv=Voronoi(sites)
        region=vv.regions[vv.point_region[len(sites)-1]]
        if -1 in region or len(region)!=6:raise AssertionError('WS boundary is not hexagonal')
        boundary=vv.vertices[region]
        theta=np.arctan2(boundary[:,1]-center[1],boundary[:,0]-center[0])
        boundary=boundary[np.argsort(theta)]
    else:
        mid=xy.mean(axis=0)
        lo=mid-.5*(g.B[:,0]+g.B[:,1])
        boundary=np.array([lo,lo+g.B[:,0],lo+g.B[:,0]+g.B[:,1],lo+g.B[:,1]])
    # One short physical link for each hopping representative; the endpoint
    # can be a periodic image rather than the stored destination index.
    endpoints=[];crossed=[];winds=[]
    for e in g.edges:
        phys=A@(cells[e.i]+np.asarray(e.delta))+ORBITALS[e.j%6]
        end=xy[e.j]
        if abs(np.linalg.norm(phys-xy[e.i])-1)>1e-8:
            raise AssertionError('PBC link not nearest-neighbour in display')
        raw=np.linalg.solve(g.B,phys-end)
        wrapped=np.rint(raw).astype(int)
        if not np.allclose(raw,wrapped,atol=1e-8):
            raise AssertionError('Display winding not integer')
        crossed.append(bool(np.any(wrapped)))
        winds.append(wrapped)
        endpoints.append(phys)
    return xy,np.asarray(endpoints),boundary,np.asarray(crossed),np.asarray(winds),np.asarray(cells)


def directed_bond_currents(g,states,psi):
    """J_{i->j}= i t ( <c_i^dag c_j>-<c_j^dag c_i> ).

    One representative per undirected bond, and no double counting of its
    Hermitian conjugate. This uses vectorized fermionic hoppings and computes
    complex rho_ij without allocating the full Ns x Ns density matrix.
    """
    psi=np.asarray(psi,dtype=complex)
    states=np.asarray(states,dtype=np.uint64)
    if len(states)!=len(psi):raise ValueError('State/vector sizes differ')
    rho=np.empty(len(g.edges),complex)
    J=np.empty(len(g.edges),float)
    for n,e in enumerate(g.edges):
        row,col,sign=edge_triplets(g,states,n)
        # Only the c_i^dag c_j direction contributes to rho_ij.
        filled_j=(states[col] & np.uint64(1<<(g.Ns-1-e.j)))!=0
        z=np.vdot(psi[row[filled_j]],sign[filled_j]*psi[col[filled_j]])
        rho[n]=z
        J[n]=float((1j*g.t[e.kind]*(z-z.conjugate())).real)
    out=np.zeros(g.Ns,float)
    for e,j in zip(g.edges,J):out[e.i]+=j;out[e.j]-=j
    return rho,J,out


def plaquette_data(g,rho_edge,J,coords):
    loops=[]
    for path in g.cycles():
        cur=[];s=[];z=[]
        for i,j in zip(path,path[1:]+path[:1]):
            e,sign=g._directed_edge(i,j)
            ix=next(k for k,ee in enumerate(g.edges) if ee is e)
            cur.append(sign*J[ix]);s.append(g.t[e.kind])
            z.append(rho_edge[ix] if sign==1 else rho_edge[ix].conjugate())
        avg=float(np.mean(cur))
        product=np.prod(z)
        flux=float(np.angle(product)) if abs(product)>1e-12 else np.nan
        # Unwrap geometry from one first site, stepping by unit bond vectors.
        pp=[coords[path[0]]]
        for i,j in zip(path,path[1:]+path[:1]):
            e,sgn=g._directed_edge(i,j)
            dr=np.asarray(e.delta) if sgn==1 else -np.asarray(e.delta)
            step=A@dr+ORBITALS[j%6]-ORBITALS[i%6]
            pp.append(pp[-1]+step)
        if np.linalg.norm(pp[-1]-pp[0])>1e-8:raise AssertionError('Noncontractible loop')
        centroid=np.mean(pp[:-1],axis=0)
        loops.append({'sites':list(path),'length':len(path),'type':'A' if path[0]%6<3 and len(path)==3 else ('B' if len(path)==3 else f'{len(path)}-gon'),
                      'I_ccw':avg,'rho_flux':flux,'rho_product_abs':float(abs(product)),
                      'bare_flux':float(np.angle(np.prod(s))),
                      'centroid':centroid,'real_coordinates':np.asarray(pp[:-1])})
    return loops


def load_scan_metadata(input_root,V):
    inp=Path(input_root)
    if (inp/'responses.json').exists():
        rec=json.loads((inp/'responses.json').read_text(encoding='utf-8'))
        info=rec.get('parameters',{})
        return inp,info,float(rec['V']),rec['row']
    if not (inp/'scan_metadata.json').exists():
        raise FileNotFoundError(f'Expected scan_metadata.json or responses.json in {inp}')
    info=json.loads((inp/'scan_metadata.json').read_text(encoding='utf-8'))
    if V is None:raise ValueError('Specify --V for a scan directory')
    g=RubyPBC(info['cluster'],t=info.get('t',1.),t1=info['t1'],t2=info['t2'])
    folder=unique_dir(inp,g,int(info['nf']),float(V))
    row={}
    if (folder/'responses.json').exists():
        rec=json.loads((folder/'responses.json').read_text(encoding='utf-8'))
        if abs(float(rec['V'])-V)>1e-9:raise RuntimeError('V mismatch')
        row=rec.get('row',{})
    return folder,info,float(V),row


def load_or_compute_ground(folder,info,V,cache=None,force_resolve=False):
    g=RubyPBC(info['cluster'],t=info.get('t',1.),t1=info['t1'],t2=info['t2'])
    nf=int(info['nf']);ground=folder/'ground_states.npz'
    if ground.exists() and not force_resolve:
        with np.load(ground) as z:
            states=z['states'].copy()
            psis=z['psi'].copy()
            e0=float(z['E0'])
        if not np.array_equal(states,make_states(g.Ns,nf)):
            raise AssertionError('Inconsistent saved fixed-N basis ordering')
        if psis.shape[0]!=len(states):raise AssertionError('Saved ground vectors wrong dimension')
        return g,states,psis,e0,'loaded ground_states.npz',cache
    if cache is None:
        cache=VScanSolver(g,nf,symmetry=info.get('symmetry','point'),
                          low=int(info.get('low',5)),max_basis=int(info.get('max_basis',70_000)))
    H,e0,psis,records,incomplete,res=cache.solve(V,float(info.get('degeneracy_tol',1e-7)))
    if incomplete:raise RuntimeError('Ground multiplet incomplete; increase --low in original scan')
    if res>1e-6:raise RuntimeError(f'Ground eigenvector residual is too large: {res}')
    return g,cache.states,psis,e0,'re-solved this V',cache


def optimize_state(g,states,psis,channel='opposite',E0=None,V=None,check_h=True):
    chan='LC_'+channel
    projection=projected_current_manifold(g,psis,states)
    eig=projection[chan]['eigenvalues']
    vec=projection[chan]['eigenvectors']
    # Positive source h>0 picks maximal SIGNED current, not the largest |J|.
    choose=int(np.argmax(eig))
    coeff=vec[:,choose]
    psi=psis@coeff
    psi/=np.linalg.norm(psi)
    psi_tr=psi.conjugate()  # spinless T=complex conjugation, real original hopping gauge
    if not np.isclose(np.vdot(psi_tr,psi_tr).real,1.,atol=1e-8):raise AssertionError('Bad TR norm')
    rho,j,div=directed_bond_currents(g,states,psi)
    rho_tr,j_tr,div_tr=directed_bond_currents(g,states,psi_tr)
    if np.max(np.abs(j+j_tr))>1e-8:raise AssertionError('TR did not reverse bond currents')
    # Current matrix check: projected eigenvalue equals direct bond-circulation sum.
    loops=plaquette_data(g,rho,j,selected_site_positions(g)[0])
    ccw=np.array([x['I_ccw'] for x in loops if x['length']==3])
    signs=np.array([1 if x['type']=='A' else -1 for x in loops if x['length']==3])
    order=ccw.sum()/np.sqrt(len(ccw)) if channel=='same' else np.dot(signs,ccw)/np.sqrt(len(ccw))
    if abs(order-eig[choose])>1e-7:
        raise AssertionError(f'Current vertex mismatch: direct={order}, projected={eig[choose]}')
    residual=np.nan
    if check_h and E0 is not None and V is not None:
        H=FixedHamiltonian(g,int(states[0]).bit_count(),V,states=states,method='sparse').operator
        residual=float(np.linalg.norm(H@psi-E0*psi))
        if residual>2e-6:
            warnings.warn(f'Optimized current state is only approximately stationary: ||(H-E0)psi||={residual:.2e}')
    winding_current=np.sum(np.array([e.winding for e in g.edges],dtype=float)*j[:,None],axis=0)
    return {'psi':psi,'psi_TR':psi_tr,'rho_edges':rho,'bond_current':j,'divergence':div,
            'winding_current':winding_current,
            'loop_data':loops,'projection':projection,'selected_channel':chan,
            'J_selected':float(eig[choose]),'current_from_bonds':float(order),
            'eigen_coefficients':coeff,'residual':residual,
            'J_TR_error':float(np.max(np.abs(j+j_tr))),
            'max_divergence':float(np.max(np.abs(div))),
            'relative_divergence':float(np.max(np.abs(div))/(np.max(np.abs(j))+1e-15))}


def plot_pattern(g,dat,output,show_labels=True,threshold=0.025):
    coords,ends,boundary,crossed,winds,dcells=selected_site_positions(g)
    j=dat['bond_current']
    loops=dat['loop_data']
    mag=np.abs(j)
    mx=float(max(np.max(mag),1e-10))
    fig,ax=plt.subplots(figsize=(10.5,8.4))
    ax.add_patch(patches.Polygon(boundary,closed=True,fill=True,facecolor='#EAF0FA',alpha=.4,
                                  edgecolor='#1F344E',linewidth=2.2,linestyle=(0,(5,4)),zorder=1))
    # Plot all 12*Ncell independent bonds, even where current is too weak for arrow.
    ghost={}
    for idx,(e,dst,wrap) in enumerate(zip(g.edges,ends,crossed)):
        src=coords[e.i]
        ax.plot([src[0],dst[0]],[src[1],dst[1]],color=CLASS_COLOR[e.kind],
                lw=1.15,alpha=.40,ls='--' if wrap else '-',zorder=2)
        if wrap: ghost[(round(dst[0],9),round(dst[1],9))]=(dst,e.j)
    for (key,(p,site)) in ghost.items():
        ax.scatter(*p,s=28,facecolors='white',edgecolors='#8698AB',linewidths=1,zorder=3)
    # Actual physical-current direction is indicated by arrow direction;
    # arrow width / opacity encodes its magnitude, and color is hopping class.
    for idx,(e,dst,wrap) in enumerate(zip(g.edges,ends,crossed)):
        current=j[idx]
        if abs(current)<threshold*mx:continue
        start,end=coords[e.i],dst
        if current<0:start,end=end,start
        tail=.29*start+.71*end
        head=.12*start+.88*end
        arrow=patches.FancyArrowPatch(tail,head,arrowstyle='-|>',
                 mutation_scale=11+7*min(1.,abs(current)/mx),
                 linewidth=1.2+2.8*abs(current)/mx,
                 color=CLASS_COLOR[e.kind],
                 linestyle='--' if wrap else '-',zorder=5)
        ax.add_patch(arrow)
    for k,p in enumerate(coords):
        ax.scatter(*p,s=92,facecolors='#FFFFFF',edgecolors='#21374D',linewidths=1.5,zorder=6)
        if show_labels:ax.text(p[0],p[1],str(k),ha='center',va='center',fontsize=7.6,color='#20384F',zorder=7)
    all_pts=np.vstack((coords,ends,boundary))
    lo=np.min(all_pts,axis=0);hi=np.max(all_pts,axis=0)
    width=hi[0]-lo[0];height=hi[1]-lo[1]
    ax.set_xlim(lo[0]-.55,hi[0]+.65);ax.set_ylim(lo[1]-.60,hi[1]+.65)
    ax.set_aspect('equal');ax.axis('off')
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color=c,lw=3,label=f'{kind}: t={g.t[kind]:+g}') for kind,c in CLASS_COLOR.items()]
    ax.legend(handles=legend,loc='upper right',framealpha=.93,fontsize=9,title='Bond type')
    ax.set_title(f"{g.name} PBC: physical bond currents | maximize {dat['selected_channel']}",pad=12)
    subtitle=(f"J={dat['J_selected']:+.6f} (normalized operator); max |Jedge|={mx:.5f}; "
              f"max |div J|={dat['max_divergence']:.1e}")
    fig.text(.5,.058,subtitle,ha='center',fontsize=10)
    fig.text(.5,.034,'Arrows: actual particle-flow direction; width: magnitude. Dashed edge / hollow destination: PBC image.',ha='center',fontsize=8.5,color='#51677D')
    fig.tight_layout(rect=[0,.08,1,.95]);fig.savefig(str(output)+'.png',dpi=230,bbox_inches='tight');fig.savefig(str(output)+'.svg',bbox_inches='tight');plt.close(fig)


def plot_triangles(g,dat,output):
    coords,ends,boundary,_,_,_=selected_site_positions(g)
    triangles=[x for x in dat['loop_data'] if x['length']==3]
    mag=max((abs(p['I_ccw']) for p in triangles),default=0)
    scale=max(mag,1e-8)
    cmap=plt.get_cmap('RdBu_r')
    fig,ax=plt.subplots(figsize=(9.5,8))
    ax.add_patch(patches.Polygon(boundary,closed=True,facecolor='#F2F6FB',alpha=.6,edgecolor='#2C4763',linewidth=2,linestyle='--'))
    sm=ScalarMappable(norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),cmap=cmap)
    for p in triangles:
        xy=np.asarray(p['real_coordinates'])
        ax.add_patch(patches.Polygon(xy,closed=True,facecolor=sm.to_rgba(p['I_ccw']),edgecolor='#36546C',linewidth=1.8,zorder=2))
        center=np.mean(xy,axis=0)
        label_color='white' if abs(p['I_ccw'])>0.55*scale else '#1D283A'
        ax.text(center[0],center[1],f"{p['type']}\n{p['I_ccw']:+.3f}",ha='center',va='center',fontsize=9,fontweight='bold',color=label_color,zorder=3)
    for p in coords:
        ax.scatter(*p,s=22,facecolor='white',edgecolor='#52677B',lw=.6,zorder=4)
    pts=np.vstack((coords,boundary));lo=np.min(pts,axis=0);hi=np.max(pts,axis=0)
    ax.set_xlim(lo[0]-.5,hi[0]+.5);ax.set_ylim(lo[1]-.5,hi[1]+.5);ax.set_aspect('equal');ax.axis('off')
    fig.colorbar(sm,ax=ax,shrink=.72,pad=.03,label='CCW triangle current (positive = CCW)')
    ax.set_title(f"{g.name}: six triangle currents in optimized ground state\n"
                 f"{dat['selected_channel']} = {dat['J_selected']:+.6f}",fontsize=13)
    fig.tight_layout();fig.savefig(str(output)+'.png',dpi=230,bbox_inches='tight');fig.savefig(str(output)+'.svg',bbox_inches='tight');plt.close(fig)


def export_one(g,states,psis,E0,V,folder,channel='opposite',check_h=True,threshold=.025):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    dat=optimize_state(g,states,psis,channel,E0,V,check_h)
    coords,ends,boundary,crossed,winds,dcells=selected_site_positions(g)
    plot_pattern(g,dat,folder/'bond_current',threshold=threshold)
    plot_triangles(g,dat,folder/'triangle_current')
    with (folder/'bond_current.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['edge_idx','site_i','site_j','cell_i','cell_j','orbital_i','orbital_j','kind','hopping','J_i_to_j','rho_ij_real','rho_ij_imag','display_PBC','display_wrap_1','display_wrap_2','canonical_winding_1','canonical_winding_2'])
        for k,(e,j,r,w,wr) in enumerate(zip(g.edges,dat['bond_current'],dat['rho_edges'],crossed,winds)):
            writer.writerow([k,e.i,e.j,e.i//6,e.j//6,e.i%6,e.j%6,e.kind,g.t[e.kind],float(j),float(r.real),float(r.imag),int(w),*wr,*e.winding])
    with (folder/'plaquette_current.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['length','type','site_indices','I_CCW','rho_flux_phase','rho_product_abs','bare_hopping_flux'])
        for p in dat['loop_data']:
            writer.writerow([p['length'],p['type'],' '.join(map(str,p['sites'])),p['I_ccw'],p['rho_flux'],p['rho_product_abs'],p['bare_flux']])
    np.savez_compressed(folder/'current_pattern.npz',sites=np.arange(g.Ns),positions=coords,
                        edge_i=np.array([e.i for e in g.edges]),edge_j=np.array([e.j for e in g.edges]),
                        hopping_kind=np.array([e.kind for e in g.edges]),
                        edge_J=dat['bond_current'],rho_edge=dat['rho_edges'],
                        divergence=dat['divergence'],
                        winding_current=dat['winding_current'],
                        wrapped=crossed,display_winding=winds,
                        eigen_coefficients=dat['eigen_coefficients'],
                        psi_selected=dat['psi'],psi_time_reverse=dat['psi_TR'])
    summary={'cluster':g.name,'Nf':int(states[0]).bit_count(),'V':float(V),'hoppings':g.t,
             'ground_dim':int(psis.shape[1]),'channel':dat['selected_channel'],
             'Jmax_signed':dat['J_selected'],'J_from_bonds':dat['current_from_bonds'],
             'Jmax_abs':float(dat['projection'][dat['selected_channel']]['jmax_abs']),
             'Jmax_per_triangle':float(dat['projection'][dat['selected_channel']]['jmax_per_triangle']),
             'max_abs_bond_current':float(np.max(abs(dat['bond_current']))),
             'canonical_net_winding_current':dat['winding_current'].tolist(),
             'max_divergence':dat['max_divergence'],
             'relative_divergence':dat['relative_divergence'],
             'time_reversal_error':dat['J_TR_error'],
             'ground_eigen_residual':dat['residual'],
             'selected_ground_current_eigenvalues':dat['projection'][dat['selected_channel']]['eigenvalues'].tolist(),
             'triangle_currents':[{'type':p['type'],'sites':p['sites'],'I_ccw':p['I_ccw']} for p in dat['loop_data'] if p['length']==3],
             'note':'Real-space currents in an ED ground-multiplet superposition; finite cluster does not prove thermodynamic LC phase.'}
    (folder/'current_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f"V={V:g}: {dat['selected_channel']}={dat['J_selected']:+.6f}, "
          f"ground_dim={psis.shape[1]}, max_div={dat['max_divergence']:.2e} -> {folder}",flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Scan root with scan_metadata.json, or V subfolder with responses.json')
    p.add_argument('--V',nargs='+',type=float,default=None,help='V values to plot from a scan root')
    p.add_argument('--channel',choices=['opposite','same'],default='opposite')
    p.add_argument('--output',default=None,help='Output folder (default: SCAN/current_patterns)')
    p.add_argument('--force-resolve',action='store_true',help='Ignore saved ground_states.npz')
    p.add_argument('--no-check-H',action='store_true',help='Skip final H-psi residual to save memory')
    p.add_argument('--arrow-threshold',type=float,default=.025,help='Minimum |Jedge| / max|Jedge| shown as arrow')
    p.add_argument('--low',type=int,default=None,help='Override --low per symmetry block if re-solving')
    p.add_argument('--max-basis',type=int,default=70000)
    args=p.parse_args(argv)
    root=Path(args.input)
    Vlist=[None] if (root/'responses.json').exists() else args.V
    if not Vlist:raise ValueError('Specify one or more --V values for a scan directory')
    dest=Path(args.output) if args.output else (root if (root/'scan_metadata.json').exists() else root.parent)/'current_patterns'
    dest.mkdir(parents=True,exist_ok=True)
    cache=None;summary=[]
    for V in Vlist:
        folder,info,val,row=load_scan_metadata(root,V)
        if args.low is not None:info=dict(info,low=args.low)
        info=dict(info,max_basis=args.max_basis)
        g,states,psis,e0,source,cache=load_or_compute_ground(folder,info,val,cache,force_resolve=args.force_resolve)
        if row and int(row.get('ground_dim',psis.shape[1]))!=psis.shape[1]:
            raise ValueError(f'Wrong ground-dimensionality for V={val}: existing scan says {row["ground_dim"]}, reconstructed {psis.shape[1]}. Increase --low or tighten --degeneracy-tol.')
        sub=dest/f'V_{val:+.9f}'
        rec=export_one(g,states,psis,e0,val,sub,channel=args.channel,check_h=not args.no_check_H,threshold=args.arrow_threshold)
        rec['state_source']=source;summary.append(rec)
    with (dest/'overview.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['V','ground_dim','channel','Jmax_signed','Jmax_abs','Jmax_per_triangle','max_abs_bond_current','max_divergence','relative_divergence','ground_eigen_residual','state_source'])
        writer.writeheader()
        for rec in summary:writer.writerow({k:rec.get(k) for k in writer.fieldnames})
    return summary


if __name__=='__main__': main()