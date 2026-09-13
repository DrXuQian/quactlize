#!/usr/bin/env python3
"""Compile a bounded TC comparison closure; reuse the admitted SIMT images."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_ppu import moe_compare as spec,moe_s1
from quactlize.runtime.compiler import Compiler,FLAGS,LIBRARIES,sha
from tools.build_kpack_dispatch import plan
from tools.build_kpack_decode_sweep import parents as decode_parents


def build(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sdk=a.sdk.resolve(strict=True);start=time.monotonic()
    requests=[(12,2,t*8,n,k,256,t) for n,k in moe_s1.SHAPES for t in moe_s1.TOKENS]
    selected,selections=plan(out,requests)
    if any(s['status']!='SELECTED' for s in selections): raise ValueError('policy missed a case')
    parents,families=spec.historical_parents()
    # This is the measured decode-neighborhood pool, not a Cartesian product.
    extra=[p for p in decode_parents() if p['qtype']==12]
    for p in selected+extra: parents[p['symbol']]=p
    for n,k in moe_s1.SHAPES:
        names=set(families[f'{n}x{k}'])|{p['symbol'] for p in extra}
        names|={s['parent'] for s in selections if s['request'][3:5]==(n,k) or list(s['request'][3:5])==[n,k]}
        families[f'{n}x{k}']=sorted(names)
    compiler=Compiler(sdk,a.cache,a.jobs)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in compiler.input_stats if p.is_relative_to(ROOT)}
    for name in ('dev/gemv_ppu/moe_compare_io.cu','dev/gemv_ppu/moe_compare.py',
                 'dev/gemv_ppu/build_moe_compare.py','tools/build_kpack_dispatch.py',
                 'tools/kpack_native_policy.cpp','tools/build_kpack_decode_sweep.py',
                 'quactlize/dispatch/policy.hpp','policies/kpack_zw810_heuristic_v1.hpp'):
        hashes[name]=sha(ROOT/name)
    env=dict(os.environ);env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    os.environ.update({k:env[k] for k in ('PATH','LD_LIBRARY_PATH')})
    print(f'Q4_MOE_COMPARE_BUILD parents={len(parents)} cases=108 jobs={a.jobs}',flush=True)
    records=compiler.compile_only(list(parents.values()),progress=lambda *x:print('Q4_MOE_COMPARE_BUILD',*x,flush=True))
    commands=[ [str(sdk/'bin/hgcc'),*FLAGS,f'-I{ROOT}',f'-I{ROOT}/third_party/actlize/include',
                 '-c',str(ROOT/'dev/gemv_ppu/moe_compare_io.cu'),'-o',str(out/'io.o')],
              ['g++','-shared','-Wl,-Bsymbolic',str(out/'io.o'),'-o',str(out/'libq4_moe_io.so'),
               f'-L{sdk}/lib',*[f'-l{x}' for x in LIBRARIES]] ]
    with (out/'adapter-build.log').open('w') as f:
        for cmd in commands:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
    modules=[]
    for r in records:
        p=out/'modules'/r['key']/'kernel.so';p.parent.mkdir(parents=True)
        shutil.copy2(r['path'],p);modules.append(r|dict(path=str(p.relative_to(out))))
    if any(sha(ROOT/p)!=value for p,value in hashes.items()):raise ValueError('build source changed')
    m=dict(schema=spec.SCHEMA,production_changed=False,device_validated=False,
        modules=modules,families=families,selection=selections,source_hashes=hashes,
        adapter_sha256=sha(out/'libq4_moe_io.so'),prior_simt_sha256=sha(spec.REVIEW),
        tactics_sha256=sha(spec.TACTICS),runtime={f'lib{x}.so':sha(sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},
        seconds=time.monotonic()-start,commands=commands)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');spec.verify(out)
    print(f'Q4_MOE_COMPARE_BUILD COMPILED parents={len(modules)} seconds={m["seconds"]:.1f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args()
    if a.jobs<1:p.error('jobs must be positive')
    build(a)
