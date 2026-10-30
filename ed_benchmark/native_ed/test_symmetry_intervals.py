"""Run as: python test_symmetry_intervals.py; NumPy only + Matplotlib for plots."""
import json
import tempfile
from pathlib import Path

import numpy as np
from analyze_symmetry_intervals import (read_scan, make_segments, estimate_intersection,
                                        extract, canonical_label, multiplet_summary,
                                        prepare_reports, sector_minimum)


def test_sectors_not_just_ground_dim():
    label1={'K':[1],'rotation':['C3',0],'mirror':0,'signed_rotation':False}
    label2={'K':[2],'rotation':['C3',0],'mirror':0,'signed_rotation':False}
    label3={'K':[0],'rotation':['C6',3],'mirror':0,'signed_rotation':False}
    d1=[{'labels':label1,'energies':[-3,0]},{'labels':label2,'energies':[-3,0]},
        {'labels':label3,'energies':[-2.99,1]}]
    d2=[{'labels':label1,'energies':[-2.94,0]},{'labels':label2,'energies':[-2.94,0]},
        {'labels':label3,'energies':[-3,1]}]
    p=extract(d1,1e-7,'ring3');q=extract(d2,1e-7,'ring3')
    assert len(p['gs'])==2 and len(q['gs'])==1
    assert p['symmetry']=="K/K' | rot=C3:m0 | d=2"
    assert q['symmetry']=='Gamma | rot=C6:m3 | d=1'
    p.update(V=1.0,responses={'CO_leading_q1':.333,'CO_leading_q2':.333},incomplete=False)
    q.update(V=2.0,responses={'CO_leading_q1':.333,'CO_leading_q2':.333},incomplete=False)
    segments,changes=make_segments([p,q]);assert len(segments)==2
    c=changes[0];c['point_left']=p;c['point_right']=q
    assert abs(estimate_intersection(c)-1.1428571428571423)<1e-8


def test_cache_and_report():
    label1={'K':[1],'rotation':['C3',0],'mirror':0,'signed_rotation':False}
    label2={'K':[2],'rotation':['C3',0],'mirror':0,'signed_rotation':False}
    label3={'K':[0],'rotation':['C6',3],'mirror':0,'signed_rotation':False}
    for invalid in (False,True):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            metadata={'cluster':'ring3','nf':6,'degeneracy_tol':1e-7,'t':1.,'t1':.5,'t2':.5}
            (root/'scan_metadata.json').write_text(json.dumps(metadata))
            for v,ek,eg in ((1.,-3.,-2.9),(2.,-3.,-2.95),(2.5,-2.92,-3.)):
                folder=root/f'V_{v:+.9f}';folder.mkdir()
                d=[{'labels':label1,'energies':[ek,0.]},
                   {'labels':label2,'energies':[ek,0.]},
                   {'labels':label3,'energies':[eg,1.]}]
                if invalid and v==2.5:
                    d[0]['energies'][0]=-3.00000001
                (folder/'sectors.json').write_text(json.dumps(d))
                (folder/'responses.json').write_text(json.dumps({'V':v,'row':{
                   'ground_dim':2 if ek<eg else 1,
                   'CO_leading_q1':.3333,'CO_leading_q2':.3333,
                   'LC_opposite_ground_Jmax_abs':0.2 if ek<eg else 0.}}))
            if invalid:
                try:read_scan(root)
                except RuntimeError as e:assert 'ground count' in str(e)
                else:raise AssertionError('Missing consistency check')
            else:
                meta,tol,pts=read_scan(root)
                seg,chg=make_segments(pts)
                assert len(seg)==2 and len(chg)==1
                for c in chg:
                    c['point_left']=next(p for p in pts if p['V']==c['V_left'])
                    c['point_right']=next(p for p in pts if p['V']==c['V_right'])
                    c['linear_estimate']=estimate_intersection(c)
                out,intervals,changes=prepare_reports(root,meta,tol,pts,seg,chg)
                for fn in ('symmetry_levels.png','symmetry_regions.png',
                           'excitation_gaps.png','symmetry_points.csv','intervals.md',
                           'intervals.csv','sector_levels.csv','crossing_brackets.csv'):
                    assert (out/fn).exists(),fn
                assert intervals[0]['ground_dim']==2
                assert changes[0]['V_lower']==2.0


if __name__=='__main__':
    test_sectors_not_just_ground_dim()
    test_cache_and_report()
    print('All fixed-filling symmetry interval tests passed')