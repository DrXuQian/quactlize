"""Equal F32 indexed endpoints for admitted SIMT and native grouped TC."""
import ctypes as C
from pathlib import Path

import numpy as np

from dev.gemv_ppu import moe_s1,moe_compare as spec
from dev.gemv_ppu.moe_s1_bench import Bench as SimtBench,error
from quactlize.dispatch.native import IndexedIO
from quactlize.runtime.native import Call,Module,Recipe,Resources as Query,checked
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import DeviceCall
from tools.run_kpack_decode_sweep import bind_group
from tools.run_kpack_grouped_decode_probe import Replay


class Unsupported(Exception):
    pass


def grid_size(parent, candidate, tokens, n, occupancy, cus):
    if not parent['persistent']:
        return 0
    rows=tokens*8
    # The host has a token bound, not a D2H expert histogram.
    entries=min((rows+min(rows,256)*(parent['tm']-1))//parent['tm'],
                256*((tokens+parent['tm']-1)//parent['tm']))
    work=entries*((n+parent['tn']-1)//parent['tn'])*candidate['split']
    cap=cus*min(candidate['grid_b'],occupancy)
    if cap<=0: raise Unsupported('NO_PERSISTENT_RESIDENCY')
    if candidate.get('grid_mode',2)==3:
        waves=(work+cap-1)//cap
        return (work+waves-1)//waves
    return min(work,cap)


class TensorCore:
    def __init__(self, base, bundle, record, candidate):
        self.base=base;self.candidate=candidate;self.handles=[];self.graph=None
        self.r=Resources(base.sdk);self.module=None
        try:
            self.module=Module(record|dict(path=str((bundle/record['path']).resolve())))
            self.query,self.prepare=bind_group(self.module)
            p=record['parent'];data=base.data;m=data['rows'];n,k=base.n,base.k
            dev=self.module.device_identity()
            if dev['ordinal']!=base.device['ordinal'] or dev['compute_units']!=72:
                raise ValueError('TC device differs from expected ZW810')
            self.recipe=Recipe(1,C.sizeof(Recipe),p['persistent'],candidate['split'],1 if p['persistent'] else 0)
            self.resources=Query()
            r=self.r
            self.output_bytes=m*n*2;self.compact_guard=r.alloc(self.output_bytes+32)
            self.call=Call(version=1,size=C.sizeof(Call),m=m,n=n,k=k,experts=256,group_size=32,
                device=dev['ordinal'],compute_units=dev['compute_units'],mapping_id=base.arr.mapping_id,
                a=r.alloc(m*k*2),low=base.planes[0]['low'],metadata=base.planes[0]['units'],
                output=self.compact_guard+16,offsets_device=r.alloc(257*4),stream=r.stream.value)
            d=DeviceCall(2,C.sizeof(DeviceCall),self.call,data['tokens'],0)
            rc=self.query(C.byref(d),C.byref(self.recipe),C.byref(self.resources))
            if rc==1:raise Unsupported('DEVICE_QUERY_UNSUPPORTED')
            checked(rc,'TC resource query')
            self.recipe.grid=grid_size(p,candidate,data['tokens'],n,self.resources.occupancy,dev['compute_units'])
            checked(self.query(C.byref(d),C.byref(self.recipe),C.byref(self.resources)),'TC final grid query')
            self.workspace_guard=r.alloc(self.resources.workspace_bytes+256)
            r.fill(self.workspace_guard,0xa5,self.resources.workspace_bytes+256)
            self.call.workspace=self.workspace_guard+128;self.call.workspace_bytes=self.resources.workspace_bytes
            self.io=IndexedIO(1,C.sizeof(IndexedIO),data['tokens'],8,data['channels'],0,
                11,k+8,data['channels']*(k+8),n+8,base.ids,base.a,base.out,r.alloc(m*4))
            self.fused=m<=32
            self.scope='NATIVE_FUSED_INDEXED_PREPARE_TC_REDUCE_SCATTER' if self.fused else 'GPU_RANK_GATHER_METADATA_DIRECTORY_TC_REDUCER_SCATTER'
            if self.fused:
                self.bind=self.module.lib.quactlize_kpack_bind_llama_indexed_v1
                self.bind.argtypes=[C.c_void_p,C.POINTER(IndexedIO)];self.bind.restype=C.c_int
            else:
                self.adapter=C.CDLL(str((bundle/'libq4_moe_io.so').resolve()),mode=C.RTLD_LOCAL)
                self.before=self.adapter.q4_moe_compare_prepare
                self.before.argtypes=[C.POINTER(IndexedIO),C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p]
                self.before.restype=C.c_int
                self.after=self.adapter.q4_moe_compare_finish
                self.after.argtypes=[C.POINTER(IndexedIO),C.c_void_p,C.c_int,C.c_void_p];self.after.restype=C.c_int
            for i in range(base.copies):self.handles.append(self.handle(i))
            self.negative=self.handle(0,base.zero_low);self.handles.append(self.negative)
            self.poison()
            base.sdk.synchronize(r.stream)
        except BaseException:
            self.close();raise

    def handle(self,index,low=None):
        c=Call.from_buffer_copy(self.call)
        c.low=low if low is not None else self.base.planes[index]['low']
        c.metadata=self.base.planes[index]['units']
        d=DeviceCall(2,C.sizeof(DeviceCall),c,self.base.data['tokens'],0)
        h=C.c_void_p();checked(self.prepare(C.byref(d),C.byref(self.recipe),C.byref(h)),'TC handle prepare')
        try:
            if self.fused:checked(self.bind(h,C.byref(self.io)),'native indexed endpoint binding')
        except BaseException:
            self.module.destroy(h);raise
        return h

    def launch(self,index=0,negative=False):
        stream=self.r.stream
        if not self.fused:
            rc=self.before(C.byref(self.io),self.call.a,self.call.offsets_device,self.base.k,256,stream)
            if rc:return rc
        rc=self.module.run(self.negative if negative else self.handles[index],stream)
        if rc or self.fused:return rc
        return self.after(C.byref(self.io),self.call.output,self.base.n,stream)

    def poison(self):
        self.r.fill(self.base.out_base,0xa5,self.base.output_bytes+32)
        self.r.fill(self.compact_guard,0xa5,self.output_bytes+32)

    def read(self):
        self.base.sdk.synchronize(self.r.stream)
        for ptr,size in ((self.workspace_guard,128),(self.call.workspace+self.call.workspace_bytes,128),
                         (self.compact_guard,16),(self.call.output+self.output_bytes,16)):
            if self.base.sdk.download(ptr,size)!=b'\xa5'*size:raise ValueError('TC scratch/output guard differs')
        return self.base.read()

    def correctness(self):
        self.poison();checked(self.launch(),'TC eager');got=self.read();err=error(got,self.base.data)
        if err>=.005:raise ValueError(f'TC independent GGUF error {err:.9g}')
        self.poison();checked(self.launch(negative=True),'TC zero-code plant')
        if error(self.read(),self.base.data)<=.005:raise ValueError('TC zero codes escaped oracle')
        replay=Replay(self.base.sdk,self.r.stream,self.launch,1)
        d=self.base.data; original=d['ids']; changed=original.copy();changed[:,:8]=np.roll(changed[:,:8],1,axis=1)
        try:
            upload(self.base.sdk,self.base.ids,changed)
            self.poison();checked(replay(),'TC graph changed IDs');new=self.read()
            positions=[int(np.flatnonzero(self.base.w.categories[0]==g)[0]) for g in range(4)]
            values=d['ah'][:,positions].astype('f8');experts=changed[:,:8].reshape(-1)
            truth=dict(golden=np.stack([values[d['arows'][i]]@self.base.w.sums[e] for i,e in enumerate(experts)]),
                       denom=np.stack([np.abs(values[d['arows'][i]])@self.base.w.abs_sums[e] for i,e in enumerate(experts)]))
            if error(new,truth)>=.005 or error(new,d)<=.005:raise ValueError('TC graph ID replay failed')
            upload(self.base.sdk,self.base.ids,original)
            self.poison();checked(replay(),'TC restored graph')
            if not np.array_equal(got.view('u4'),self.read().view('u4')):raise ValueError('TC eager/restored graph bits differ')
            upload(self.base.sdk,self.base.a,np.zeros_like(d['a']))
            self.poison();checked(replay(),'TC zero A')
            if np.any(self.read()!=0):raise ValueError('TC zero A failed')
        finally:
            upload(self.base.sdk,self.base.ids,original);upload(self.base.sdk,self.base.a,d['a']);replay.close()
        self.poison();checked(self.launch(),'TC restored input');self.read()
        return dict(error=err,zero_codes='PASS',zero_a='PASS',output_guard='PASS',
                    gpu_ids_replay='PASS',eager_graph_bits='PASS')

    def replay(self):
        if self.graph is None:
            index=0
            def invoke():
                nonlocal index
                rc=self.launch(index%self.base.copies);index+=1;return rc
            self.graph=Replay(self.base.sdk,self.r.stream,invoke,self.base.calls_per_graph)
            checked(self.graph(),'TC graph upload excluded');self.base.sdk.synchronize(self.r.stream)
        return self.graph

    def measure(self,count):
        values=[x/self.base.calls_per_graph for x in self.r.samples(self.replay(),count)]
        if error(self.read(),self.base.data)>=.005:raise ValueError('TC rotating graph output differs')
        return values

    def receipt(self):
        return dict(scope=self.scope,split=self.recipe.split,grid=self.recipe.grid,
                    grid_b=self.candidate['grid_b'],shared_bytes=self.resources.shared_bytes,
                    workspace_bytes=self.resources.workspace_bytes,occupancy=self.resources.occupancy,
                    endpoints='F32_A_ROUNDED_F16_F32_OUTPUT',output_rounding='F16_THEN_F32',
                    gpu_routing=True,host_routing=False,weights_moved=False)

    def close(self):
        if self.r:
            self.base.sdk.synchronize(self.r.stream)
            if self.graph:self.graph.close();self.graph=None
            if self.module:
                for h in self.handles:self.module.destroy(h)
            self.handles=[];self.r.close();self.r=None


def upload(sdk,pointer,array):
    checked(sdk.lib.hggcMemcpy(pointer,array.ctypes.data,array.nbytes,1),'untimed fixture upload')
    sdk.synchronize(None)
