"""CPU gates for indexed endpoint accounting, inventory recall and recovery."""
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import moe_compare as spec,moe_s1,run_moe_compare as runner
from dev.gemv_ppu.moe_compare_bench import grid_size
from quactlize.runtime.candidates import PARENT_FIELDS
from quactlize.dispatch.native import IndexedIO


def test_history_is_compile_tuple_not_stale_resource_estimate():
    parents,families=spec.historical_parents()
    assert parents and len(families)==6
    assert all(set(p)==set(PARENT_FIELDS) for p in parents.values())
    assert families['1024x2048']==families['512x2048']
    assert families['1024x3072']==families['512x3072']
    assert all(p['qtype']==12 and p['route']=='fq-grouped' for p in parents.values())


def test_prior_simt_all_108_winners_and_runners_preserved():
    rows=spec.prior_simt()
    assert len(rows)==108
    for names in rows.values():
        assert len(names)==len(set(names))==2
        for name in names:moe_s1.lookup(name)


def test_split_and_grid_pool_keeps_unsupported_cells_explicit():
    parent=next(p for p in spec.historical_parents()[0].values() if p['persistent'])
    m=dict(families={'2048x512':[parent['symbol']]},modules=[dict(parent=parent)],
           selection=[dict(parent=parent['symbol'],split=1,grid_b=7,grid_mode=3)])
    candidates=spec.tc_inventory(m,2048,512)
    assert len(candidates)==len({x['key'] for x in candidates})==25
    assert any(c['grid_b']==7 and c['grid_mode']==3 and c['split']==1 for c in candidates)
    assert all(c['reason']=='INSUFFICIENT_K_TILES_PER_PIPELINE_SLICE' for c in candidates if c['split']>2)
    assert all(c['reason'] is None for c in candidates if c['split']<=2)


@pytest.mark.parametrize('tokens',range(1,9))
@pytest.mark.parametrize('mode',(2,3))
def test_persistent_grid_uses_public_token_bound(tokens,mode):
    p=dict(tm=8,tn=64,persistent=1)
    c=dict(split=4,grid_b=4,grid_mode=mode)
    work=tokens*8*8*4
    observed=grid_size(p,c,tokens,512,3,72)
    expected=min(work,216) if mode==2 else (work+(work+215)//216-1)//((work+215)//216)
    assert observed==expected and 0<observed<=work


@pytest.fixture(scope='module')
def rows_library(tmp_path_factory):
    path=tmp_path_factory.mktemp('q4-moe-rows')/'rows.so'
    result=subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',
        '-I'+str(spec.ROOT),'-I'+str(spec.ROOT/'third_party/actlize/include'),
        str(spec.ROOT/'tests/q4_moe_compare_rows.cpp'),'-o',str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    lib=C.CDLL(str(path));lib.rows.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
    lib.rows.restype=C.c_int
    return lib


@pytest.mark.parametrize('tokens',range(1,9))
@pytest.mark.parametrize('cluster',(False,True))
def test_actual_gpu_rank_helpers_cover_64_rows(rows_library,tokens,cluster):
    ids=np.array([(np.arange(8)*17+3+(0 if cluster else t*13))%256 for t in range(tokens)],dtype='i4').reshape(-1)
    ranks=np.zeros(len(ids),dtype='i4');offsets=np.zeros(257,dtype='i4')
    assert rows_library.rows(ids.ctypes.data,len(ids),ranks.ctypes.data,offsets.ctypes.data)==1
    np.testing.assert_array_equal(ranks,np.argsort(np.argsort(ids,kind='stable')))
    np.testing.assert_array_equal(offsets,np.r_[0,np.bincount(ids,minlength=256).cumsum()])
    for value in (-1,256,int(ids[1])):
        fault=ids.copy();fault[0]=value
        assert rows_library.rows(fault.ctypes.data,len(fault),ranks.ctypes.data,offsets.ctypes.data)==0


def proof():
    return dict(error=1e-5,zero_a='PASS',zero_codes='PASS',output_guard='PASS',gpu_ids_replay='PASS')


def test_child_retry_keeps_success_and_does_not_label_failed_tc_slow(tmp_path,monkeypatch):
    configs=[dict(key='simt:s0',arm='simt',recipe='s0',reason=None),dict(key='simt:s1',arm='simt',recipe='s1',reason=None),
             dict(key='tc:p0',arm='tc',parent='p0',reason=None),dict(key='tc:p1',arm='tc',parent='p1',reason=None)]
    counts={c['key']:0 for c in configs};planted=[False]
    ident={'test':1};device={'l2_bytes':67108864}
    class Base:
        def __init__(self,*a):
            self.device=device;self.data={'expert':np.arange(8)};self.copies=32;self.calls_per_graph=64
            self.weight_sha256={'low':'fixture','units':'fixture'}
        def setup(self,*a):pass
        def correctness(self,key):counts['simt:'+key]+=1;return proof()
        def measure(self,key,count):return [1. if key=='s0' else 1.1]*count
        def close(self):pass
    class TC:
        def __init__(self,base,bundle,record,candidate):self.key=candidate['key']
        def correctness(self):
            counts[self.key]+=1
            if self.key=='tc:p0' and not planted[0]:planted[0]=True;raise ValueError('planted numeric failure')
            return proof()
        def measure(self,count):return [2. if self.key=='tc:p0' else 2.1]*count
        def receipt(self):return dict(scope='TC_WITH_ADAPTERS',split=1)
        def close(self):pass
    monkeypatch.setattr(runner,'SimtBench',Base);monkeypatch.setattr(runner,'TensorCore',TC)
    monkeypatch.setattr(runner,'identity',lambda _:ident);monkeypatch.setattr(runner,'catalog',lambda *a:configs)
    args=SimpleNamespace(case=[512,2048,1,1,0],output=tmp_path,sdk=tmp_path,bundle=tmp_path,
                         l2_bytes=0,profile=False,retry_failed=False)
    manifest={'modules':[{'parent':{'symbol':p}} for p in ('p0','p1')]}
    with pytest.raises(ValueError,match='planted'):runner.child(args,manifest)
    assert runner.child(args,manifest)==2
    assert counts=={'simt:s0':1,'simt:s1':1,'tc:p0':1,'tc:p1':1}
    args.retry_failed=True
    assert runner.child(args,manifest)==0
    assert counts=={'simt:s0':1,'simt:s1':1,'tc:p0':2,'tc:p1':1}
    row=json.loads((tmp_path/'n512-k2048-t1-ch1-spread.json').read_text())
    runner.validate_result(row,ident,row['case'],configs)
    assert row['simt_vs_tc_pct']==-50
    for change in ({'best':{}},{'simt_vs_tc_pct':0},{'confirmation':row['confirmation'][:-1]}, {'copies':1}):
        with pytest.raises(ValueError):runner.validate_result(row|change,ident,row['case'],configs)
    bad=copy.deepcopy(row);bad['screen'][0]['samples_us'][0]=float('nan')
    with pytest.raises(ValueError):runner.validate_result(bad,ident,row['case'],configs)


def test_timing_contains_native_call_or_all_fallback_stages():
    from dev.gemv_ppu.moe_compare_bench import TensorCore
    seen=[]
    t=TensorCore.__new__(TensorCore);t.r=SimpleNamespace(stream=9);t.base=SimpleNamespace(k=512,n=2048)
    t.call=SimpleNamespace(a=1,offsets_device=2,output=3);t.io=IndexedIO()
    t.handles=[11];t.negative=12;t.module=SimpleNamespace(run=lambda *a:seen.append('GEMM_REDUCER') or 0)
    t.before=lambda *a:seen.append('GPU_IDS_GATHER') or 0
    t.after=lambda *a:seen.append('SCATTER') or 0
    t.fused=False;assert t.launch()==0 and seen==['GPU_IDS_GATHER','GEMM_REDUCER','SCATTER']
    seen.clear();t.fused=True;assert t.launch()==0 and seen==['GEMM_REDUCER']


def test_new_endpoint_adapter_does_not_change_production_32_row_guard():
    source=(spec.ROOT/'dev/gemv_ppu/moe_compare_io.cu').read_text()
    assert 'Capacity = 64' in source and 'ranked_row(ids,m,tid)' in source
    assert 'hggcMemcpy' not in source and 'cudaDeviceSynchronize' not in source
    production=(spec.ROOT/'quactlize/runtime/module.cuh').read_text()
    assert 'call.m>32' in production


def test_recreated_tc_confirmation_handle_initializes_guards_without_correctness_rerun():
    import inspect
    from dev.gemv_ppu.moe_compare_bench import TensorCore
    init=inspect.getsource(TensorCore.__init__)
    assert init.index('self.poison()')<init.index('base.sdk.synchronize(r.stream)')
    assert 'instance.poison()' in inspect.getsource(runner.child)
