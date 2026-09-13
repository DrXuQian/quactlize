// Development-only endpoint adapter for the existing unfused TC path.
// No weights are moved. The native fused indexed ABI remains capped at32 rows.
#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
#include "quactlize/runtime/indexed_rows.hpp"
#include "quactlize/integrations/llama/indexed.h"

namespace q4_moe_compare {
using Half = cutlass::half_t;
constexpr int Capacity = 64;

__global__ void prepare(qk_llama_indexed_v1 io, Half* a, int* offsets, int k, int experts) {
    __shared__ int ids[Capacity], ranks[Capacity];
    int tid=threadIdx.x, m=io.tokens*io.topk;
    if(tid<m) ids[tid]=io.ids[int64_t(tid/io.topk)*io.ids_stride+tid%io.topk];
    __syncthreads();
    bool invalid=false;
    if(tid<m) {
        invalid=ids[tid]<0 || ids[tid]>=experts;
        for(int j=tid-tid%io.topk;j<tid;++j) invalid|=ids[j]==ids[tid];
        ranks[tid]=quactlize::runtime::ranked_row(ids,m,tid);
    }
    bool valid=__syncthreads_or(invalid)==0;
    if(blockIdx.x==0) {
        for(int e=tid;e<=experts;e+=blockDim.x)
            offsets[e]=valid ? quactlize::runtime::expert_begin(ids,m,e) : 0;
        if(tid<m) io.row_ids[valid?ranks[tid]:tid]=valid?tid:-1;
    }
    if(!valid) return;
    int r=blockIdx.x;
    int64_t from=int64_t(r/io.topk)*io.a_token_stride+(r%io.topk%io.channels)*io.a_row_stride;
    for(int col=tid;col<k;col+=blockDim.x) a[int64_t(ranks[r])*k+col]=Half(io.a[from+col]);
}

__global__ void finish(qk_llama_indexed_v1 io, Half const* compact, int n) {
    int row=blockIdx.y, to=io.row_ids[row];
    for(int col=blockIdx.x*blockDim.x+threadIdx.x;col<n;col+=gridDim.x*blockDim.x) {
        float value=to>=0 ? float(compact[int64_t(row)*n+col]) : __int_as_float(0x7fc00000);
        io.output[int64_t(to>=0?to:row)*io.out_row_stride+col]=value;
    }
}

bool valid(qk_llama_indexed_v1 const* p, int columns) {
    return p && p->version==1 && p->size==sizeof(*p) && !p->reserved &&
        p->tokens>0 && p->topk==8 && p->tokens<=8 && p->tokens*p->topk<=Capacity &&
        (p->channels==1 || p->channels==8) && p->ids_stride>=8 &&
        p->ids && p->row_ids && p->a && p->output && columns>0;
}
}

extern "C" int q4_moe_compare_prepare(qk_llama_indexed_v1 const* io,void* a,int* offsets,
                                     int k,int experts,void* stream) {
    if(!q4_moe_compare::valid(io,k) || !a || !offsets || experts!=256 ||
       io->a_row_stride<k || io->a_token_stride<io->channels*io->a_row_stride) return 2;
    if(hggcGetLastError()!=hggcSuccess) return 3;
    q4_moe_compare::prepare<<<io->tokens*io->topk,256,0,static_cast<hggcStream_t>(stream)>>>(
        *io,static_cast<q4_moe_compare::Half*>(a),offsets,k,experts);
    return hggcGetLastError()==hggcSuccess ? 0 : 3;
}
extern "C" int q4_moe_compare_finish(qk_llama_indexed_v1 const* io,void const* output,
                                    int n,void* stream) {
    if(!q4_moe_compare::valid(io,n) || !output || io->out_row_stride<n) return 2;
    if(hggcGetLastError()!=hggcSuccess) return 3;
    q4_moe_compare::finish<<<dim3((n+255)/256,io->tokens*io->topk),256,0,static_cast<hggcStream_t>(stream)>>>(
        *io,static_cast<q4_moe_compare::Half const*>(output),n);
    return hggcGetLastError()==hggcSuccess ? 0 : 3;
}
