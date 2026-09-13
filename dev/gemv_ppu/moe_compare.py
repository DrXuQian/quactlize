"""Bounded indexed Q4 SIMT/TC comparison, preserving measured parent recall."""
import csv
import json
from pathlib import Path

from dev.gemv_ppu import moe_s1
from dev.gemv_ppu.run_moe_s1 import cases, case_id
from quactlize.runtime.compiler import sha, validate_parent
from quactlize.runtime.candidates import PARENT_FIELDS
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'quactlize.q4-moe-compare.v1'
REVIEW = ROOT/'docs/measurements/q4_moe_s1_20260913/summary.tsv'
TACTICS = ROOT/'policies/kpack_zw810_tactics.json'
SIMT_BUNDLE = ROOT/'prebuilt/ppu0010/q4-moe-s1-v2'
TC_BUNDLE = ROOT/'prebuilt/ppu0010/q4-moe-compare-v1'


def historical_parents():
    model = json.loads(TACTICS.read_text())
    families = {}
    for n, k in moe_s1.SHAPES:
        source_n = n//2 if n == 1024 else n
        key = digest((12, 'fq-grouped', source_n, k, 32, 256))
        names = {p['parent'] for p in model['anchors'][key]
                 if p['features'][0] <= 64 and p['features'][1] <= 8}
        if not names:
            raise ValueError(f'no historical decode anchor: {n,k}')
        families[f'{n}x{k}'] = sorted(names)
    parents = {name:{f:model['parents'][name][f] for f in PARENT_FIELDS}
               for names in families.values() for name in names}
    for p in parents.values():
        validate_parent(p)
    return parents, families


def prior_simt():
    records = list(csv.DictReader(REVIEW.open(), delimiter='\t'))
    expected = {case_id(n,k,t,ch,c) for n,k in moe_s1.SHAPES for t,ch,c in cases()}
    if len(records) != len(expected) or {r['case'] for r in records} != expected:
        raise ValueError('prior SIMT cases missing/duplicated')
    return {r['case']: [r['selected'], r['runner']] for r in records}


def tc_key(parent, split, grid_b=0, grid_mode=0):
    return f'{parent}:s{split}:b{grid_b}:g{grid_mode}'


def tc_inventory(manifest, n, k):
    pool = manifest['families'][f'{n}x{k}']
    out = []
    for record in manifest['modules']:
        p = record['parent']
        if p['symbol'] not in pool:
            continue
        for split in (1,2,4,8):
            grids = [(b,g) for b in (1,2,4) for g in (2,3)] if p['persistent'] else [(0,0)]
            grids += [(s['grid_b'],s['grid_mode']) for s in manifest['selection']
                      if s['parent']==p['symbol'] and s['split']==split]
            for b,g in sorted(set(grids)):
                reason = None
                if k % (p['tk']*split) or k//(p['tk']*split) < p['stages']-1:
                    reason = 'INSUFFICIENT_K_TILES_PER_PIPELINE_SLICE'
                out.append(dict(key=tc_key(p['symbol'],split,b,g),parent=p['symbol'],
                                split=split,grid_b=b,grid_mode=g,reason=reason))
    return out


def verify(bundle=TC_BUNDLE):
    bundle = Path(bundle)
    m = json.loads((bundle/'manifest.json').read_text())
    if m.get('schema') != SCHEMA or m.get('production_changed') is not False:
        raise ValueError('wrong comparison manifest')
    history, families = historical_parents()
    parents = {r['parent']['symbol']:r['parent'] for r in m['modules']}
    if len(parents) != len(m['modules']):
        raise ValueError('duplicated TC parent')
    for family, names in families.items():
        if not set(names) <= set(m['families'][family]):
            raise ValueError('historical winner missing: '+family)
    if any(parents.get(name) != p for name,p in history.items()):
        raise ValueError('historical parent tuple changed')
    for r in m['modules']:
        path = (bundle/r['path']).resolve(strict=True)
        if not path.is_relative_to(bundle.resolve()) or sha(path) != r['sha256']:
            raise ValueError('TC module payload differs')
    if sha(bundle/'libq4_moe_io.so') != m['adapter_sha256']:
        raise ValueError('GPU endpoint adapter differs')
    for p, value in m['source_hashes'].items():
        if sha(ROOT/p) != value:
            raise ValueError('comparison build source differs: '+p)
    if sha(REVIEW) != m['prior_simt_sha256'] or sha(TACTICS) != m['tactics_sha256']:
        raise ValueError('measurement inventory differs')
    for selection in m['selection']:
        if selection['status'] != 'SELECTED':
            raise ValueError('policy control was omitted')
        _,_,_,n,k,_,_ = selection['request']
        if selection['parent'] not in m['families'][f'{n}x{k}']:
            raise ValueError('policy parent missing')
    moe_s1.verify(SIMT_BUNDLE)
    return m
