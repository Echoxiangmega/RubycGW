#!/usr/bin/env python3
"""Classify fixed-filling Ruby ED V scans by finite-torus ground-state symmetry.

Uses V_*/sectors.json and responses.json, already written by scan_susceptibility.py.
Do not interpret a symmetry-sector regime as a proven thermodynamic phase.
Optional --refine recomputes *energies only* at bisection points.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def canonical_label(label):
    return (tuple(label.get('K') or ()),
            tuple(label.get('rotation') or ()),
            label.get('mirror'), bool(label.get('signed_rotation', False)))


def label_str(label):
    k,r,m,sg=canonical_label(label)
    rot='none' if not r else f'{r[0]}:m{r[1]}' + ('*' if sg else '')
    return 'K='+','.join(map(str,k))+' '+rot+' M='+('?' if m is None else ('+' if m==0 else '-'))


def momentum_name(k,cluster):
    k=tuple(k)
    if not k or all(v==0 for v in k): return 'Gamma'
    if cluster=='ring3' and len(k)==1 and k[0] in (1,2):return 'K' if k[0]==1 else "K'"
    if cluster=='2x2':return 'M'+''.join(map(str,k))
    if cluster=='3x1':return 'q='+str(k[0])+'/3'
    return 'K='+','.join(map(str,k))


def multiplet_summary(labels,cluster):
    groups=Counter(canonical_label(x) for x in labels)
    names=sorted(set(momentum_name(k,cluster) for k,_,_,_ in groups))
    momentum='K/K\'' if names==['K',"K'"] else ','.join(names)
    rot=sorted(set((r[0]+':m'+str(r[1]) if r else '-') for _,r,_,_ in groups))
    return f'{momentum} | rot={"/".join(rot)} | d={len(labels)}'


def extract(records,deg_tol,cluster):
    rows=[]
    for rec in records:
        lab=rec.get('labels',{})
        for level,e in enumerate(rec.get('energies',rec.get('eigenvalues',[]))):
            rows.append({'E':float(e),'level':level,'labels':lab,'label':label_str(lab),
                         'key':canonical_label(lab)})
    rows.sort(key=lambda r:r['E'])
    if not rows:raise ValueError('No eigenvalues in sectors.json')
    e0=rows[0]['E']
    gs=[r for r in rows if r['E']-e0<=deg_tol]
    signature=tuple(sorted([r['key'] for r in gs],key=str))
    gap=next((r['E']-e0 for r in rows if r['E']-e0>deg_tol),float('nan'))
    return {'E0':e0,'gs':gs,'signature':signature,'gap':gap,
            'symmetry':multiplet_summary([r['labels'] for r in gs],cluster), 'rows':rows}


def read_scan(root,deg_tol=None):
    root=Path(root)
    meta_path=root/'scan_metadata.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    if not meta.get('cluster'):
        geometry=next(root.glob('V_*/geometry.json'),None)
        if geometry:
            cfg=json.loads(geometry.read_text(encoding='utf-8'))
            meta.setdefault('cluster',cfg.get('name','ring3'))
            meta.setdefault('t',cfg.get('t',{}).get('t',1.))
            meta.setdefault('t1',cfg.get('t',{}).get('t1',.5))
            meta.setdefault('t2',cfg.get('t',{}).get('t2',.5))
            meta.setdefault('nf',cfg.get('Nf'))
    tol=float(deg_tol if deg_tol is not None else meta.get('degeneracy_tol',1e-7))
    cluster=meta.get('cluster','ring3')
    points=[]
    for fn in root.glob('V_*/sectors.json'):
        folder=fn.parent
        resp=folder/'responses.json'
        data=json.loads(resp.read_text(encoding='utf-8')) if resp.exists() else {}
        try:V=float(data.get('V',folder.name[2:]))
        except ValueError:continue
        raw=json.loads(fn.read_text(encoding='utf-8'))
        res=extract(raw,tol,cluster)
        res.update(V=V,dir=folder,responses=data.get('row',{}),incomplete=bool(data.get('row',{}).get('ground_incomplete',False)))
        if res['incomplete']:
            raise RuntimeError(f'{folder}: ground manifold flagged incomplete; increase --low for ED first')
        if 'ground_dim' in res['responses'] and len(res['gs'])!=int(res['responses']['ground_dim']):
            raise RuntimeError(f'{folder}: sector ground count {len(res["gs"])} != recorded ground_dim {res["responses"]["ground_dim"]}')
        points.append(res)
    points.sort(key=lambda x:x['V'])
    if not points:raise ValueError(f'No V_*/sectors.json found in {root}')
    if len(set(p['V'] for p in points))!=len(points):raise ValueError('Duplicate V points')
    return meta,tol,points


def make_segments(points):
    ans=[];changes=[]
    start=0
    for j in range(1,len(points)):
        if points[j]['signature']!=points[j-1]['signature']:
            left,right=points[j-1],points[j]
            changes.append({'V_left':left['V'],'V_right':right['V'],
                            'from':left['symmetry'],'to':right['symmetry'],
                            'left_signature':left['signature'],'right_signature':right['signature']})
            ans.append(points[start:j]);start=j
    ans.append(points[start:])
    return ans,changes


def sector_minimum(point,signature):
    match_keys=set(signature)
    if not match_keys:return float('nan')
    vals=[r['E'] for r in point['rows'] if r['key'] in match_keys]
    return min(vals) if vals else float('nan')


def estimate_intersection(a):
    A,B=a['left_signature'],a['right_signature']
    la=sector_minimum(a['point_left'],A)-sector_minimum(a['point_left'],B)
    lb=sector_minimum(a['point_right'],A)-sector_minimum(a['point_right'],B)
    if np.isfinite(la) and np.isfinite(lb) and la*lb<0 and abs(lb-la)>1e-15:
        return a['V_left']+(a['V_right']-a['V_left'])*(-la)/(lb-la)
    return float('nan')


def annotate_points(points):
    rows=[]
    for p in points:
        r=p['responses']
        d={'V':p['V'],'E0':p['E0'],'ground_dim':len(p['gs']),
           'gap_above_ground':p['gap'],'ground_symmetry':p['symmetry'],
           'ground_labels_json':json.dumps([z['labels'] for z in p['gs']],ensure_ascii=False,sort_keys=True)}
        for k in ('CO_leading_q1','CO_leading_q2','CO_leading_equal_time_lambda',
                  'CO_E_matrix_even_fraction','CO_E_matrix_odd_fraction',
                  'CO_leading_E_fraction','LC_same_ground_Jmax_abs',
                  'LC_opposite_ground_Jmax_abs','LC_same_zero_weight',
                  'LC_opposite_zero_weight','CO_leading_status',
                  'CO_leading_chi','LC_same_status','LC_opposite_status',
                  'LC_same_chi_regular','LC_opposite_chi_regular'):
            d[k]=r.get(k, '')
        rows.append(d)
    return rows


def summarize_evidence(group,cluster):
    # DO NOT compare CO susceptibility numerically to LC susceptibility;
    # units/normalizations differ. Only print physically observable indicators.
    vals=[]
    q_names=[]
    for p in group:
        r=p['responses']
        q1=r.get('CO_leading_q1');q2=r.get('CO_leading_q2')
        if q1 is not None and q2 is not None:
            q_names.append(tuple(round(float(x),3) for x in (q1,q2)))
    if q_names:
        common,count=Counter(q_names).most_common(1)[0]
        vals.append(f'CO strongest equal-time q={common} in {count}/{len(group)} points')
    for tag in ('LC_same','LC_opposite'):
        js=[p['responses'].get(tag+'_ground_Jmax_abs') for p in group]
        js=[float(z) for z in js if z is not None and np.isfinite(float(z))]
        if js: vals.append(f'{tag} Jmax range [{min(js):.4g},{max(js):.4g}]')
    return '; '.join(vals) or 'No response metadata; symmetry labels only'


def write_csv(path,records):
    if not records:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def prepare_reports(root,meta,tol,points,segments,changes,refined=None):
    root=Path(root);out=root/'symmetry_intervals';out.mkdir(exist_ok=True)
    points_csv=annotate_points(points)
    write_csv(out/'symmetry_points.csv',points_csv)
    flat=[]
    for p in points:
        for r in p['rows']:
            flat.append({'V':p['V'],'E':r['E'],'E_minus_E0':r['E']-p['E0'],
                         'K':str(r['labels'].get('K')),'rotation':str(r['labels'].get('rotation')),
                         'mirror':str(r['labels'].get('mirror')),'signed_rotation':r['labels'].get('signed_rotation',False),
                         'level_in_sector':r['level'],'sector':r['label']})
    write_csv(out/'sector_levels.csv',flat)
    interval_rows=[]
    for i,seg in enumerate(segments,1):
        r0=seg[0];r1=seg[-1]
        interval_rows.append({'interval_id':i,'V_first_sample':r0['V'],'V_last_sample':r1['V'],
         'number_of_samples':len(seg),'ground_symmetry':r0['symmetry'],
         'ground_dim':len(r0['gs']),'ground_multiplet_json':json.dumps([r['labels'] for r in r0['gs']]),
         'diagnostic_evidence':summarize_evidence(seg,meta.get('cluster','ring3')),
         'physical_phase':'UNDETERMINED (finite-size symmetry sector, not thermodynamic phase)'})
    write_csv(out/'intervals.csv',interval_rows)
    change_rows=[]
    for i,c in enumerate(changes):
        change_rows.append({'transition_id':i+1,'V_lower':c['V_left'],'V_upper':c['V_right'],
         'from':c['from'],'to':c['to'],'linear_interpolation_hint':c.get('linear_estimate',float('nan')),
         'refined_lower':c.get('refined_lower',float('nan')),'refined_upper':c.get('refined_upper',float('nan')),
         'warning':c.get('warning',''),
         'note':'Brackets mark where sampled ground-state symmetry changes; not an established phase transition'})
    write_csv(out/'crossing_brackets.csv',change_rows)
    with (out/'intervals.md').open('w',encoding='utf-8') as f:
        f.write('# Ruby fixed-filling ED symmetry regimes (finite torus)\n\n')
        f.write('**This is a ground-state symmetry interval diagram, NOT a certified CO/LC thermodynamic phase diagram.**\n')
        f.write('A symmetry switch is a finite-cluster level crossing or near-degeneracy change; check a finer V grid, order parameters and size dependence.\n\n')
        f.write('## Observed contiguous intervals\n\n')
        for row in interval_rows:
            f.write(f'- Segment {row["interval_id"]}: V={row["V_first_sample"]:g}..{row["V_last_sample"]:g}, '
                    f'{row["ground_symmetry"]}; {row["diagnostic_evidence"]}.\n')
        f.write('\n## Unresolved change brackets\n\n')
        for row in change_rows:
            a,b=row['V_lower'],row['V_upper']
            f.write(f'- {row["from"]} -> {row["to"]}, between V={a:g} and {b:g}. ')
            v=row['linear_interpolation_hint']
            if np.isfinite(v):f.write(f'Linear estimate {v:.6g} (not a calculated crossing). ')
            if np.isfinite(row['refined_lower']):f.write(f'Refined bracket [{row["refined_lower"]:.8g}, {row["refined_upper"]:.8g}]. ')
            f.write('\n')
        f.write('\nSource: existing sectors.json / responses.json. All energies in t units.\n')
    write_csv(out/'energy_slopes.csv',energy_slope_records(points))
    nc=4 if meta.get('cluster')=='2x2' else 3 if meta.get('cluster') in ('ring3','3x1') else 1
    plot_results(out,points,interval_rows,changes,num_cells=nc)
    return out,interval_rows,change_rows


def energy_slope_records(points):
    """Secant dE0/dV computed only within the same ground symmetry interval.

    Never interpolate the derivative across an observed level crossing.
    """
    out=[]
    for left,right in zip(points[:-1],points[1:]):
        same=left['signature']==right['signature']
        dv=right['V']-left['V']
        out.append({'V_mid':(left['V']+right['V'])/2,
                    'V_left':left['V'],'V_right':right['V'],
                    'E0_left':left['E0'],'E0_right':right['E0'],
                    'same_ground_signature':bool(same),
                    'slope_dE0_dV':((right['E0']-left['E0'])/dv if same else float('nan')),
                    'ground_symmetry':left['symmetry'] if same else 'CROSSING BRACKET'})
    return out


def plot_results(out,points,intervals,changes,num_cells=1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs=np.array([p['V'] for p in points])
    # Show relevant sectors: all ground multiplets or within 0.15t anywhere.
    interesting={r['key'] for p in points for r in p['rows'] if r['E']-p['E0']<.18}
    interesting=sorted(interesting,key=str)
    fig,ax=plt.subplots(figsize=(10,6))
    colors=plt.cm.tab20(np.linspace(0,1,max(3,len(interesting))))
    for j,k in enumerate(interesting):
        ys=[min((r['E']-p['E0'] for r in p['rows'] if r['key']==k),default=np.nan) for p in points]
        label=next(r['label'] for p in points for r in p['rows'] if r['key']==k)
        ax.plot(xs,ys,'-o',markersize=3,linewidth=1.3,color=colors[j],label=label)
    for c in changes:ax.axvspan(c['V_left'],c['V_right'],color='grey',alpha=.13)
    ax.set_xlabel('V / t');ax.set_ylabel('Sector energy - E0 (units of t)')
    ax.set_title('Exact finite-cluster low-energy symmetry-sector levels')
    ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2,loc='best')
    fig.tight_layout();fig.savefig(out/'symmetry_levels.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,2.6))
    labels=[r['ground_symmetry'] for r in intervals]
    cmap=plt.cm.Set2(np.linspace(0,1,len(intervals)))
    left=float(xs[0]);right=float(xs[-1]);step=(right-left)/max(1,len(xs)-1)
    for i,part in enumerate(intervals):
        vl,vr=part['V_first_sample'],part['V_last_sample']
        # Observed region only, not an extrapolation outside sampled points.
        ax.plot([vl,vr],[.5,.5],lw=17,color=cmap[i],solid_capstyle='butt')
        ax.scatter([vl,vr],[.5,.5],s=30,color=cmap[i],edgecolors='black',linewidths=.4,zorder=3)
        ax.text((vl+vr)/2,.67,chr(65+i) if i<26 else str(i+1),ha='center',fontsize=11,weight='bold')
    for c in changes:
        ax.axvspan(c['V_left'],c['V_right'],alpha=.16,color='#999999')
        ax.text((c['V_left']+c['V_right'])/2,.22,'crossing\nbracket',ha='center',fontsize=8)
    ax.set_xlim(left-.05*step,right+.05*step);ax.set_ylim(0,1.0);ax.set_yticks([])
    ax.set_xlabel('V/t');ax.set_title('Observed ground-state symmetry intervals (not physical phase labels)')
    desc='   '.join((chr(65+i)+': '+z) for i,z in enumerate(labels))
    fig.text(.02,.02,desc,fontsize=7)
    fig.tight_layout(rect=(0,.11,1,1));fig.savefig(out/'symmetry_regions.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6))
    ax.plot(xs,[p['gap'] for p in points],'-o',label='First gap above ground multiplet')
    for c in changes:ax.axvspan(c['V_left'],c['V_right'],color='grey',alpha=.13)
    ax.set_xlabel('V/t');ax.set_ylabel('Energy / t');ax.set_title('Finite-size excitation gap and symmetry change brackets')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'excitation_gaps.png',dpi=180);plt.close(fig)

    # The large extensive E0(V) trend can conceal a cusp. Show both E0
    # and per-primitive-cell energy without changing raw numerical data.
    en=np.asarray([p['E0'] for p in points],float)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,scale,ylabel in zip(axes,(1.,float(num_cells)),('Total E0 / t','E0 / (Nc t)')):
        ax.plot(xs,en/scale,'-o',ms=4,color='#226e9e')
        for c in changes:ax.axvspan(c['V_left'],c['V_right'],color='grey',alpha=.16)
        ax.set_xlabel('V / t');ax.set_ylabel(ylabel);ax.grid(alpha=.25)
    axes[0].set_title('Finite-cluster ground-state energy')
    axes[1].set_title('Ground-state energy per primitive cell')
    fig.tight_layout();fig.savefig(out/'ground_energy.png',dpi=180);plt.close(fig)

    # Between-sample slopes are not exact Hellmann-Feynman expectation
    # values. Skip segments spanning a symmetry switch.
    fig,ax=plt.subplots(figsize=(9,4.7))
    for row in energy_slope_records(points):
        if not row['same_ground_signature']:continue
        ax.hlines(row['slope_dE0_dV'],row['V_left'],row['V_right'],
                  colors='#226e9e',linewidth=2.2)
        ax.plot(row['V_mid'],row['slope_dE0_dV'],'o',color='#226e9e',ms=4)
    for c in changes:ax.axvspan(c['V_left'],c['V_right'],color='grey',alpha=.16)
    ax.set_xlabel('V / t');ax.set_ylabel('Within-interval secant Delta E0 / Delta V')
    ax.set_title('Energy slope (do not fit across level crossings)')
    ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'ground_energy_slope.png',dpi=180);plt.close(fig)



def refine_brackets(meta,changes,delta_v,max_steps):
    if not changes:return
    from geometry import RubyPBC
    from scan_susceptibility import VScanSolver
    cluster=meta.get('cluster','ring3');nf=int(meta['nf'])
    g=RubyPBC(cluster,t=meta.get('t',1.),t1=meta.get('t1',.5),t2=meta.get('t2',.5))
    solver=VScanSolver(g,nf,meta.get('symmetry','point'),int(meta.get('low',5)),int(meta.get('max_basis',70000)))
    tol=meta.get('degeneracy_tol',1e-7)
    cache={}
    def check(V):
        if V not in cache:
            H,E0,psis,rec,incomplete,residual=solver.solve(V,tol)
            if incomplete:raise RuntimeError('Incomplete ground multiplet in refinement')
            tmp=extract(rec,tol,cluster)
            cache[V]=tmp['signature']
        return cache[V]
    for c in changes:
        v0,v1=c['V_left'],c['V_right'];a,b=c['left_signature'],c['right_signature']
        # Binary localization is valid for a SINGLE transition; newly
        # encountered extra ground signatures are reported explicitly.
        mid_signatures=[]
        for step in range(max_steps):
            if v1-v0<=delta_v:break
            vm=.5*(v0+v1)
            sig=check(vm)
            if sig==a:v0=vm
            elif sig==b:v1=vm
            else:
                mid_signatures.append({'V':vm,'signature':str(sig)})
                break
        c['refined_lower']=v0;c['refined_upper']=v1
        if mid_signatures:c['warning']='Additional intermediate multiplet found: '+str(mid_signatures)
    return cache


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Fixed-n scan root with V_*/sectors.json')
    p.add_argument('--degeneracy-tol',type=float,default=None)
    p.add_argument('--refine',action='store_true',help='Recompute energy sectors ONLY at V midpoints')
    p.add_argument('--refine-tol',type=float,default=.02,help='Target V width for crossing bracket')
    p.add_argument('--refine-steps',type=int,default=8)
    args=p.parse_args(argv)
    if args.refine_tol<=0 or args.refine_steps<0:raise ValueError('Invalid refine options')
    meta,tol,pts=read_scan(args.input,args.degeneracy_tol)
    parts,changes=make_segments(pts)
    for c in changes:
        c['point_left']=next(p for p in pts if p['V']==c['V_left'])
        c['point_right']=next(p for p in pts if p['V']==c['V_right'])
        c['linear_estimate']=estimate_intersection(c)
    if args.refine:refine_brackets(meta,changes,args.refine_tol,args.refine_steps)
    out,intervals,rows=prepare_reports(args.input,meta,tol,pts,parts,changes)
    print('Scan:',args.input,'| samples:',len(pts),'| intervals:',len(intervals))
    for row in intervals:
        print(f'  [{row["V_first_sample"]:g}, {row["V_last_sample"]:g}] ',row['ground_symmetry'])
    for row in rows:
        print(f'  SWITCH bracket [{row["V_lower"]:g}, {row["V_upper"]:g}]',row['from'],'->',row['to'])
        if math.isfinite(row['refined_lower']):print('    refined',row['refined_lower'],row['refined_upper'])
    print('Wrote:',out)


if __name__=='__main__':main()