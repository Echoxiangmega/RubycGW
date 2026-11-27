#!/usr/bin/env python3
"""Scan V_tri vs filling n for the periodic, spinless Ruby hex Hartree--Fock model.

V_t1=V_t2=0 BY CONSTRUCTION (the t1/t2 hopping is retained).  This script
imports RubyHexHF and the translation-aware phase diagnostics supplied with
the previous hex-HF package.  A converged minimum-F state is selected at each
fixed-filling point, and all progress is saved before continuing.

Example:
    python scan_ruby_hex_vtri_filling.py --period ring3 --symmetry none \\
      --t 1 --t1 0.5 --t2 0.5 --nk 10 10 \\
      --Vtri-min 0 --Vtri-max 3 --Vtri-step 0.25 \\
      --n-min 1 --n-max 3 --n-step 0.25 \\
      --temperature 0.04 --output scan_hex_Vtri_n

Works with local ruby_hex_hf.py and plot_ruby_hex_phase.py (corrected version).
No pairing is included.  n is spinless electrons per 6-site primitive cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ruby_hex_hf import RubyHexHF, save_solution
from plot_ruby_hex_phase import LABELS, COLORS, SHORT, order_diagnostics


def axis_points(start: float, stop: float, step: float, *, name: str) -> list[float]:
    if not all(np.isfinite([start, stop, step])) or step <= 0 or stop < start:
        raise ValueError(f"Invalid {name} range: {start}, {stop}, {step}")
    n = int(round((stop - start) / step))
    if not np.isclose(start + n * step, stop, rtol=0, atol=2e-9):
        raise ValueError(f"{name}: (max-min) must be an integer multiple of step")
    return [float(round(start + i * step, 12)) for i in range(n + 1)]


def unique_points(values: list[float], *, name: str) -> list[float]:
    result = [float(v) for v in values]
    if not result or any(not math.isfinite(v) for v in result):
        raise ValueError(f"Invalid {name} values")
    if len({round(v, 12) for v in result}) != len(result):
        raise ValueError(f"Duplicate {name} values")
    return sorted(result)


def config_for_scan(a: argparse.Namespace, vs: list[float], ns: list[float]) -> dict:
    return dict(
        solver='ruby_hex_hf', geometry=a.period, symmetry=a.symmetry,
        nk=[int(i) for i in a.nk], t=float(a.t), t1=float(a.t1), t2=float(a.t2),
        V_t1=0., V_t2=0., Vtri_values=vs, n_values=ns,
        temperature=float(a.temperature), mixing=float(a.mix),
        tol=float(a.tol), max_iter=int(a.max_iter), seed_amplitude=float(a.seed_amplitude),
        seeds=list(a.seeds), random_starts=int(a.random_starts),
        auto_lc_modes=not a.no_auto_lc_modes, continuation=not a.no_continuation,
        rng_seed=int(a.seed),
    )


def stable_key(v: float, n: float) -> str:
    return f'{v:.12g}|{n:.12g}'


def record_path(root: Path, ip: int) -> Path:
    return root / f'point_{ip:04d}'


def load_restart_state(root: Path, ip: int) -> dict | None:
    state = record_path(root, ip) / 'best' / 'state.npz'
    if not state.is_file():
        return None
    with np.load(state, allow_pickle=False) as z:
        return {'n': np.array(z['n'], dtype=float),
                'chi': np.array(z['chi'], dtype=complex)}


def shift_density(initial: dict, filling: float) -> dict:
    """Preserve modulations when continuing between different target fillings."""
    n = np.asarray(initial['n'], dtype=float).copy()
    n += filling / 6. - float(n.mean())
    n = np.clip(n, 1e-8, 1. - 1e-8)
    return {'n': n, 'chi': np.asarray(initial['chi'], dtype=complex).copy()}


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(path)


def persist(root: Path, rows: dict[str, dict]) -> None:
    allrows = sorted(rows.values(), key=lambda r: int(r['point']))
    atomic_write(root/'scan.json', json.dumps(allrows, indent=2, allow_nan=False))
    cols=list(dict.fromkeys(k for row in allrows for k in row))
    tmp=root/'scan.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    tmp.replace(root/'scan.csv')


def classify(m: RubyHexHF, result: dict, a: argparse.Namespace) -> tuple[int, dict]:
    # This calls the corrected translation-aware classifier, not raw bond std.
    return order_diagnostics(m, {'n': result['n'], 'chi': result['chi']}, result, a)


def edges(values: list[float]) -> np.ndarray:
    x=np.array(values, dtype=float)
    if len(x)==1:
        return np.array([x[0] - .5, x[0] + .5])
    mids=.5*(x[:-1]+x[1:])
    return np.r_[x[0] - (mids[0]-x[0]), mids, x[-1]+(x[-1]-mids[-1])]


def plot_maps(root: Path, vs: list[float], ns: list[float], a: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    path = root/'scan.json'
    if not path.is_file():
        print('[plot] No scan.json, cannot draw plots')
        return
    rows=json.loads(path.read_text(encoding='utf-8'))
    loc={(round(float(r['V_tri']),12),round(float(r['filling_per_primitive']),12)):r for r in rows}
    labels=np.zeros((len(ns),len(vs)), dtype=int)
    quantity={k:np.full(labels.shape,np.nan) for k in (
        'charge_q_rms','real_bond_q_rms','current_rms_recomputed',
        'J_tri_rms','F_per_primitive','sd_isd_form_overlap',
        'pseudospin_co_even_amplitude','pseudospin_co_odd_amplitude',
        'pseudospin_co_amplitude')}
    for iv,v in enumerate(vs):
        for jn,n in enumerate(ns):
            r=loc.get((round(v,12),round(n,12)))
            if not r or not r.get('valid',False):continue
            labels[jn,iv]=int(r.get('phase_code',0))
            for key in quantity:
                if r.get(key) is not None:quantity[key][jn,iv]=float(r[key])
    # Respect actual physical coordinates even on nonuniform grids.
    X,Y=np.meshgrid(edges(vs),edges(ns))
    fig,ax=plt.subplots(figsize=(max(7, .55*len(vs)+2),max(5,.42*len(ns)+2)),layout='constrained')
    cmap=ListedColormap(COLORS)
    norm=BoundaryNorm(np.arange(-.5,len(COLORS)+.5),len(COLORS))
    ax.pcolormesh(X,Y,labels,cmap=cmap,norm=norm,shading='flat')
    ax.set(xlabel=r'$V_{\rm tri}$',ylabel=r'$n$ (per 6-site primitive cell)',
           title=f'Ruby {a.period} HF: Vt1=Vt2=0, {a.symmetry.upper()} constraint')
    if a.annotate=='phase':
        for j,n in enumerate(ns):
            for i,v in enumerate(vs):
                if labels[j,i]>0:ax.text(v,n,SHORT[labels[j,i]],ha='center',va='center',fontsize=7)
    present=sorted(set(int(x) for x in labels.flat))
    patches=[Patch(facecolor=COLORS[i],label=LABELS[i]) for i in present]
    ax.legend(handles=patches,loc='upper left',bbox_to_anchor=(1.015,1),frameon=False,fontsize=9)
    for ext in ('png','svg'):
        fig.savefig(root/f'phase.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

    labels_data=[('charge_q_rms',r'Finite-$Q$ charge RMS'),
                 ('real_bond_q_rms',r'Finite-$Q$ real bond RMS'),
                 ('pseudospin_co_even_amplitude',r'In-plane pseudospin CO even'),
                 ('pseudospin_co_odd_amplitude',r'In-plane pseudospin CO odd'),
                 ('current_rms_recomputed',r'Bond current RMS'),
                 ('J_tri_rms',r'Triangle circulation RMS'),
                 ('F_per_primitive',r'$F$ per primitive cell')]
    for key,title in labels_data:
        z=np.ma.masked_invalid(quantity[key])
        fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
        vmin=None
        if key != 'F_per_primitive':vmin=0
        im=ax.pcolormesh(X,Y,z,cmap='viridis',shading='flat',vmin=vmin)
        fig.colorbar(im,ax=ax,label=title)
        ax.set(xlabel=r'$V_{\rm tri}$',ylabel=r'$n$ (per 6-site primitive cell)',title=title)
        fig.savefig(root/f'{key}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    print(f'[plot] Saved phase.png, phase.svg and order-amplitude maps under {root}',flush=True)


def write_candidates(point: Path, trials: list[dict]) -> None:
    cols=['seed','converged','iterations','residual','symmetry_residual','F','Omega',
          'E','Ekin','EH','EFock','Npart','mu','charge_std','charge_mod_max',
          'J_A','J_B','J_plus','J_minus','J_tri_rms','current_rms','max_divergence',
          'has_inplane_pseudospin_co','pseudospin_x_even','pseudospin_x_odd',
          'pseudospin_y_even','pseudospin_y_odd','pseudospin_co_even_amplitude',
          'pseudospin_co_odd_amplitude','pseudospin_co_amplitude']
    tmp=point/'candidates.csv.tmp'
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in trials:w.writerow({k:r.get(k) for k in cols})
    tmp.replace(point/'candidates.csv')


def execute(a: argparse.Namespace) -> None:
    vs=unique_points(a.Vtri_values if a.Vtri_values is not None else
                     axis_points(a.Vtri_min,a.Vtri_max,a.Vtri_step,name='Vtri'),name='Vtri')
    ns=unique_points(a.n_values if a.n_values is not None else
                     axis_points(a.n_min,a.n_max,a.n_step,name='n'),name='n')
    if not all(0. < n < 6. for n in ns):
        raise ValueError('Fixed-filling grand-canonical HF requires 0 < n < 6')
    if a.random_starts < 0 or a.max_iter < 1 or not 0 < a.mix <= 1.:
        raise ValueError('Invalid solver settings')
    if a.symmetry=='c6' and abs(a.t1-a.t2)>1e-12:
        raise ValueError('C6 requires t1=t2; choose c3 or none for unequal hopping')
    root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    conf=config_for_scan(a,vs,ns)
    confpath=root/'scan_settings.json'
    if a.plot_only:
        if not confpath.exists():
            raise FileNotFoundError(f'Missing scan settings: {confpath}')
        existing=json.loads(confpath.read_text(encoding='utf-8'))
        vs=[float(x) for x in existing['Vtri_values']]
        ns=[float(x) for x in existing['n_values']]
        a.period=existing['geometry']
        a.symmetry=existing['symmetry']
        scanpath=root/'scan.json'
        if scanpath.is_file():
            oldrows=json.loads(scanpath.read_text(encoding='utf-8'))
            refreshed={}
            for row in oldrows:
                if row.get('valid'):
                    model=RubyHexHF(row['period'],row['symmetry'],
                        (int(row['Nk1']),int(row['Nk2'])),row['t'],row['t1'],row['t2'],
                        row['V_tri'],row['V_t1'],row['V_t2'])
                    state=record_path(root,int(row['point']))/'best'/'state.npz'
                    if state.is_file():
                        with np.load(state,allow_pickle=False) as z:
                            code,diag=order_diagnostics(model,z,row,a)
                        row.update(diag)
                        row['phase_code']=int(code)
                        row['phase']=LABELS[code]
                refreshed[stable_key(float(row['V_tri']),
                    float(row['filling_per_primitive']))]=row
            persist(root,refreshed)
        plot_maps(root,vs,ns,a)
        return
    if confpath.exists():
        old=json.loads(confpath.read_text(encoding='utf-8'))
        if old!=conf:
            raise ValueError('Existing output directory has incompatible scan_settings.json; use a new --output')
        if not a.resume and (root/'scan.json').exists():
            raise FileExistsError('Existing scan results; pass --resume to continue safely or change --output')
    else:
        if a.resume and (root/'scan.json').exists():
            raise RuntimeError('Cannot safely resume results without scan_settings.json')
        atomic_write(confpath,json.dumps(conf,indent=2))
    rows={}
    if a.resume and (root/'scan.json').exists():
        data=json.loads((root/'scan.json').read_text(encoding='utf-8'))
        rows={stable_key(float(r['V_tri']),float(r['filling_per_primitive'])):r for r in data}
    # Ordering: filling-major and alternating V direction to keep neighbor warm starts nearby.
    order=[(v,n) for j,n in enumerate(ns) for v in (vs if j%2==0 else list(reversed(vs)))]
    point_of={stable_key(v,n):i for i,(v,n) in enumerate(order)}
    successful={}
    print(f'[scan] {len(vs)} Vtri x {len(ns)} fillings = {len(order)} points',flush=True)
    print('[scan] V_t1 = V_t2 = 0 (interaction only; t1/t2 hoppings remain)',flush=True)
    for ip,(v,n) in enumerate(order):
        key=stable_key(v,n)
        previous=rows.get(key)
        point=record_path(root,ip)
        if a.resume and previous and previous.get('valid') and load_restart_state(root,ip) is not None:
            successful[key]=ip
            print(f'[resume] {ip+1}/{len(order)} Vtri={v:g} n={n:g} recovered best',flush=True)
            continue
        if a.resume and previous and not previous.get('valid') and not a.retry_invalid:
            print(f'[resume] {ip+1}/{len(order)} Vtri={v:g} n={n:g} invalid; use --retry-invalid to recompute',flush=True)
            continue
        model=RubyHexHF(a.period,a.symmetry,tuple(a.nk),a.t,a.t1,a.t2,v,0.,0.)
        point.mkdir(parents=True,exist_ok=True)
        seeds=list(a.seeds) + ['random']*a.random_starts
        if not a.no_auto_lc_modes:
            seeds.extend(f'lc_mode{k}' for k in range(model.current_null.shape[1]))
        # Use 2D warm starts: previous V point and same V on adjacent filling.
        neighbors=[]
        if not a.no_continuation:
            # Last valid point in current serpentine sweep (including previous filling boundary).
            if successful:
                last=next(reversed(successful))
                neighbors.append(('continuation_previous',last))
            jn=ns.index(n)
            if jn>0:
                nk=stable_key(v,ns[jn-1])
                if nk in successful and all(x[1]!=nk for x in neighbors):
                    neighbors.append(('continuation_filling',nk))
        print(f'[point {ip+1}/{len(order)}] Vtri={v:.6g}, n={n:.6g}, '
              f'Nk={a.nk}, seeds={len(seeds)}+{len(neighbors)}',flush=True)
        trials=[]
        tasks=[(name,None) for name in seeds]
        for name,nkey in neighbors:
            state=load_restart_state(root,successful[nkey])
            if state is not None:tasks.insert(0,(name,shift_density(state,n)))
        for si,(name,initial) in enumerate(tasks):
            orig='normal' if initial is not None else name
            try:
                result=model.solve(name=orig,target=model.ncell*n,mu=0.,T=a.temperature,
                      amp=a.seed_amplitude,max_iter=a.max_iter,mix=a.mix,tol=a.tol,
                      seed=a.seed+ip*1009+si,initial=initial,verbose=a.verbose)
            except Exception as exc:
                print(f'  [failed] {name}: {type(exc).__name__}: {exc}',flush=True)
                continue
            result['seed']=name
            if result['converged']:
                _,diag=classify(model,result,a)
                result.update(diag)
            trials.append(result)
            print(f'  {name:25} converged={str(result["converged"]):5} '
                  f'res={result["residual"]:.2e} F={result["F"]:.9g} '
                  f'Jtri={result["J_tri_rms"]:.2e} CDW={result["charge_mod_max"]:.2e} '
                  f'COe/o={result.get("pseudospin_co_even_amplitude",0.):.2e}/'
                  f'{result.get("pseudospin_co_odd_amplitude",0.):.2e}',flush=True)
            if a.save_candidates and result['converged']:
                save_solution(model,result,point/'candidate_states'/f'{si:02d}_{name}')
        write_candidates(point,trials)
        good=[r for r in trials if r['converged'] and np.isfinite(r['F'])]
        row={
            'point':int(ip),'V_tri':float(v),'V_t1':0.,'V_t2':0.,
            'filling_per_primitive':float(n),'Ntarget':float(model.ncell*n),
            'valid':bool(good),'period':a.period,'symmetry':a.symmetry,
            'Nk1':int(a.nk[0]),'Nk2':int(a.nk[1]),
            'temperature':float(a.temperature),'t':float(a.t),'t1':float(a.t1),'t2':float(a.t2),
            'mu_fixed':None,'trials':len(trials),'converged_trials':len(good),
        }
        if good:
            best=min(good,key=lambda r:float(r['F']))
            save_solution(model,best,point/'best')
            code,diag=classify(model,best,a)
            row.update({k:(float(best[k]) if k not in ('seed',) else str(best[k]))
                for k in ('seed','F','Omega','E','mu','Npart','J_A','J_B','J_plus','J_minus',
                          'J_tri_rms','current_rms','max_divergence','charge_std',
                          'charge_mod_max','real_bond_std','residual','symmetry_residual')})
            row.update({k:float(vv) if isinstance(vv,(float,np.floating)) else int(vv)
                        for k,vv in diag.items()})
            row['phase_code']=int(code);row['phase']=LABELS[code]
            row['F_per_primitive']=float(best['F'])/model.ncell
            successful[key]=ip
            print(f'  [BEST] seed={best["seed"]} F={best["F"]:.9g} '
                  f'phase={LABELS[code]}',flush=True)
        else:
            row['phase_code']=0;row['phase']=LABELS[0]
            print('  [NO SOLUTION] No converged candidate. Phase remains unknown.',flush=True)
        rows[key]=row
        persist(root,rows)
        if a.plot_every>0 and (ip+1)%a.plot_every==0:
            plot_maps(root,vs,ns,a)
    plot_maps(root,vs,ns,a)
    print('[done] Use scan.csv to find point ID and point_XXXX/best/state.npz for its pattern.',flush=True)


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--period',choices=['ring3','hex2x2'],default='ring3')
    p.add_argument('--symmetry',choices=['c6','c3','none'],default='none',
                   help='Default none allows the in-plane pseudospin CO seeds to break C3')
    p.add_argument('--nk',nargs=2,type=int,default=[8,8])
    p.add_argument('--t',type=float,default=1.)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--Vtri-min',type=float,default=0.)
    p.add_argument('--Vtri-max',type=float,default=3.)
    p.add_argument('--Vtri-step',type=float,default=.25)
    p.add_argument('--Vtri-values',nargs='+',type=float,default=None,
                   help='Explicit Vtri grid; overrides min,max,step')
    p.add_argument('--n-min',type=float,default=1.)
    p.add_argument('--n-max',type=float,default=3.)
    p.add_argument('--n-step',type=float,default=.25)
    p.add_argument('--n-values',nargs='+',type=float,default=None,
                   help='Explicit fillings; overrides n-min,max,step')
    p.add_argument('--temperature',type=float,default=.04)
    p.add_argument('--seeds',nargs='+',default=['normal','x_even','x_odd','y_even','y_odd',
       'cdw','sd','isd','real_bond','lc_same','lc_opposite','random'])
    p.add_argument('--random-starts',type=int,default=1)
    p.add_argument('--no-auto-lc-modes',action='store_true')
    p.add_argument('--no-continuation',action='store_true')
    p.add_argument('--save-candidates',action='store_true')
    p.add_argument('--seed-amplitude',type=float,default=.06)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-iter',type=int,default=400)
    p.add_argument('--tol',type=float,default=1e-7)
    p.add_argument('--mix',type=float,default=.25)
    p.add_argument('--charge-threshold',type=float,default=1e-4)
    p.add_argument('--bond-threshold',type=float,default=2e-3)
    p.add_argument('--current-threshold',type=float,default=1e-4)
    p.add_argument('--min-form-overlap',type=float,default=.45)
    p.add_argument('--annotate',choices=['none','phase'],default='none')
    p.add_argument('--plot-every',type=int,default=0,
                   help='Redraw maps every N computed points; 0 means only at end')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-invalid',action='store_true')
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--output',default='scan_hex_Vtri_n')
    return p


if __name__=='__main__':
    execute(parser().parse_args())
