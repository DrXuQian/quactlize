#!/usr/bin/env python3
"""Export source-bound legacy/fixed Q4 MoE SIMT numerical checks for CUDA."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_ppu import moe_s1 as spec
from dev.gemv_ppu.moe_s1_bench import fixture
from tools.kpack_execution_fixture import IndexedWeights

LEGACY='be2bc98'


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    include=out/'include'; include.mkdir()
    sources={}
    def copy_header(path):
        if path in sources: return
        text=path.read_text(); sources[path]=text
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"',text,re.M):
            candidates=[x for x in (path.parent/name,ROOT/'quactlize/include'/name,ROOT/'quactlize/execution'/name) if x.is_file()]
            if not candidates: raise ValueError('local include unresolved: '+name)
            copy_header(candidates[0])
    copy_header(ROOT/'quactlize/execution/q4_s1_kernel.cuh')
    for path,text in sources.items():
        (include/path.name).write_text(text.replace('hggc','cuda'))
    # Only the reader code differs between the two DSOs. The caller, fixture,
    # source geometry, activation view and validation are otherwise shared.
    for arm in ('legacy','fixed'):
        target=out/arm; target.mkdir()
        for name in ('q4_s1_kernel.cuh','q4_s1_readers.cuh','q4_s1_helpers.cuh','q4_s1_activation.cuh'):
            path=ROOT/'quactlize/execution'/name
            text=subprocess.check_output(['git','show',f'{LEGACY}:quactlize/execution/{name}'],cwd=ROOT,text=True) if arm=='legacy' else path.read_text()
            (target/name).write_text(text.replace('hggc','cuda'))
    extra='''
extern "C" int diag_alloc(void** p,size_t n) { return int(cudaMalloc(p,n)); }
extern "C" int diag_copy(void* dst,void const* src,size_t n,int kind) { return int(cudaMemcpy(dst,src,n,cudaMemcpyKind(kind))); }
extern "C" int diag_free(void* p) { return int(cudaFree(p)); }
extern "C" int diag_sync() { return int(cudaDeviceSynchronize()); }
'''
    for n,k in spec.SHAPES:
        for arm in ('legacy','fixed'):
            (out/f'{arm}-n{n}-k{k}.cu').write_text(spec.source(n,k).replace('hggc','cuda')+extra)
        w=IndexedWeights(12,n,k,16)
        data={name:w.planes[name] for name in ('low','units')}
        for t in range(1,9):
            for ch in (1,8):
                d=fixture(w,t,ch)
                for name in ('a','ids','golden','denom'): data[f't{t}_ch{ch}_{name}']=d[name]
        np.savez_compressed(out/f'fixture-n{n}-k{k}.npz',**data)
        print(f'Q4_MOE_CUDA_FIXTURE shape={n}x{k} experts=16 cases=16',flush=True)
    (out/'check.py').write_text((ROOT/'dev/gemv_cuda/check_moe_s1.py').read_text())
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()}
    m=dict(schema='quactlize.q4-moe-s1-cuda-check.v1',files=files,source_hashes=source_hashes,
        legacy=LEGACY,shapes=spec.SHAPES,experts=16,tokens=list(range(1,9)),input_types=[0,1],
        recipes=[asdict(r)|dict(key=r.key) for r in spec.inventory()],
        scope='NUMERICAL_ONLY_SAME_DEVICE_BODIES_RUNTIME_SPELLING_ADAPTER_NOT_PPU_ADMISSION')
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
