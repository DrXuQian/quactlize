#!/usr/bin/env python3
"""Numerical-only CUDA execution of the PPU S1 source, in an isolated adapter.

Build support is mechanical runtime spelling substitution; device arithmetic,
row indexing, layouts and launch/config validation remain the real source.
The development adapter is never linked into a PPU or llama.cpp library.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


class Call(C.Structure):
    _fields_=[(x,C.c_uint32) for x in ('version','size')]+[(x,C.c_int32) for x in
        ('qtype','n','k','experts','rows','mode','input_type','channels','topk')]+[(x,C.c_int64) for x in
        ('a_row_stride','a_token_stride','ids_stride','out_row_stride')]+[(x,C.c_void_p) for x in
        ('a','low','high','units','offsets','ids','output','workspace')]+[('workspace_bytes',C.c_uint64),('stream',C.c_void_p)]


class Config(C.Structure):
    _fields_=[(x,C.c_uint32) for x in ('version','size')]+[(x,C.c_int32) for x in ('reader','variant','warps','values')]


class Arrangement(C.Structure):
    _fields_=[(x,C.c_int32) for x in ('version','layout','bits','high_bits','artifact_tile_k',
        'transport_tile_k','group_size','reserved')]+[('mapping_id',C.c_uint64)]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(a):
    root=a.package; m=json.loads((root/'manifest.json').read_text())
    for path,sha in m['files'].items():
        if digest(root/path)!=sha: raise ValueError('input source/fixture differs: '+path)
    command=[str(a.nvcc),'-std=c++17','-O3','-arch=sm_120','--shared','-Xcompiler=-fPIC']
    for n,k in m['shapes']:
        for arm in ('legacy','fixed'):
            out=root/f'{arm}-n{n}-k{k}.so'
            cmd=command+['-I'+str(root/arm),'-I'+str(root/'include'),str(root/f'{arm}-n{n}-k{k}.cu'),'-o',str(out)]
            with (root/f'build-{arm}-n{n}-k{k}.log').open('w') as log:
                rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            if rc: raise ValueError(f'CUDA build failed: {arm} {n} {k}')
            print(f'Q4_MOE_CUDA_BUILD arm={arm} shape={n}x{k} sha256={digest(out)}',flush=True)


def run(a):
    root=a.package; m=json.loads((root/'manifest.json').read_text())
    records=[]
    for n,k in m['shapes']:
        with np.load(root/f'fixture-n{n}-k{k}.npz',allow_pickle=False) as f:
            arrays={name:np.ascontiguousarray(f[name]) for name in f.files}
        low,units=arrays['low'],arrays['units']
        e=low.shape[0]
        for arm in ('legacy','fixed'):
            lib=C.CDLL(str(root/f'{arm}-n{n}-k{k}.so'),mode=C.RTLD_LOCAL)
            # cudart is linked statically into each diagnostic DSO. Its
            # exported driver wrappers avoid any host CUDA library discovery.
            alloc=lib.diag_alloc; alloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
            copy=lib.diag_copy; copy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
            free=lib.diag_free; free.argtypes=[C.c_void_p]
            fn=lib.quactlize_q4_s1_run_v1; fn.argtypes=[C.POINTER(Call),C.POINTER(Config),C.POINTER(Arrangement)]
            for f in (alloc,copy,free,fn,lib.diag_sync): f.restype=C.c_int
            pointers=[]
            def check(rc):
                if rc: raise ValueError('CUDA runtime/launch rc='+str(rc))
            def upload(arr):
                p=C.c_void_p(); check(alloc(C.byref(p),arr.nbytes)); pointers.append(p)
                check(copy(p,arr.ctypes.data,arr.nbytes,1)); return p.value
            lp,up=upload(low),upload(units)
            arrangement=Arrangement(2,1,4,0,0,64,32,0,0x51344b5034540001)
            try:
                for t in range(1,9):
                    for ch in (1,8):
                        prefix=f't{t}_ch{ch}_'; av=arrays[prefix+'a']; ids=arrays[prefix+'ids']
                        gold,denom=arrays[prefix+'golden'],arrays[prefix+'denom']
                        rows=t*8; shape=(rows,n+8)
                        poison=np.full((rows*(n+8)+8,),0xa5a5a5a5,dtype='<u4')
                        op=upload(poison); aid=upload(ids)
                        outputs={}
                        case_pointers=len(pointers)-2
                        for inp in (0,1):
                            act=np.ascontiguousarray(av.astype('<f2' if inp==0 else '<f4'))
                            ap=upload(act)
                            c=Call(version=1,size=C.sizeof(Call),qtype=12,n=n,k=k,experts=e,rows=rows,
                                mode=2,input_type=inp,channels=ch,topk=8,a_row_stride=k+8,a_token_stride=ch*(k+8),
                                ids_stride=11,out_row_stride=n+8,a=ap,ids=aid,low=lp,units=up,output=op+16)
                            for r in m['recipes']:
                                cfg=Config(1,C.sizeof(Config),r['reader'],r['variant'],r['warps'],r['values'])
                                check(copy(op,poison.ctypes.data,poison.nbytes,1)); check(fn(C.byref(c),C.byref(cfg),C.byref(arrangement))); check(lib.diag_sync())
                                host=np.empty_like(poison); check(copy(host.ctypes.data,op,host.nbytes,2))
                                result=host[4:-4].view('<f4').reshape(shape)
                                guard=bool(np.all(host[:4]==0xa5a5a5a5) and np.all(host[-4:]==0xa5a5a5a5) and np.all(result[:,n:].view('<u4')==0xa5a5a5a5))
                                got=result[:,:n].copy()
                                err=float(np.max(np.abs(got.astype('f8')-gold)/np.maximum(denom,1e-30))) if np.isfinite(got).all() else None
                                bits=None
                                if inp==0: outputs[r['key']]=got
                                else: bits=bool(np.array_equal(got.view('<u4'),outputs[r['key']].view('<u4')))
                                expected_red=arm=='legacy' and r['reader']!=0
                                passed=guard and err is not None and ((err>.05) if expected_red else (err<.005)) and bits is not False
                                row=dict(arm=arm,n=n,k=k,tokens=t,channels=ch,input_type=inp,recipe=r['key'],error=err,
                                    guard=guard,f16_f32_exact=bits,expected_red=expected_red,status='PASS' if passed else 'FAIL')
                                records.append(row)
                                print('Q4_MOE_CUDA_CELL '+json.dumps(row),flush=True)
                        for p in pointers[case_pointers:]: check(free(p))
                        del pointers[case_pointers:]
            finally:
                for p in pointers: check(free(p))
    summary=dict(status='PASS' if records and all(r['status']=='PASS' for r in records) else 'FAIL',
        scope='RTX5070_NUMERICAL_ONLY_NOT_PPU_ADMISSION_OR_PERFORMANCE',cells=len(records),records=records,
        package_sha256=digest(root/'manifest.json'),libraries={p.name:digest(p) for p in root.glob('*.so')})
    (root/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Q4_MOE_CUDA_DONE '+json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--nvcc',type=Path,default=Path('/usr/local/cuda-12.8/bin/nvcc'))
    p.add_argument('--build',action='store_true')
    a=p.parse_args()
    if a.build: build(a)
    raise SystemExit(run(a))
