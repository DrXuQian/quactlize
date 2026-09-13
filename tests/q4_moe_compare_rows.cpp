#include "quactlize/runtime/indexed_rows.hpp"
extern "C" int rows(int const* ids,int m,int* ranks,int* offsets) {
    if(m<=0 || m>64 || m%8) return 0;
    using namespace quactlize::runtime;
    if(!valid_route(ids,m,8,256)) return 0;
    for(int i=0;i<m;++i) ranks[i]=ranked_row(ids,m,i);
    for(int e=0;e<=256;++e) offsets[e]=expert_begin(ids,m,e);
    return 1;
}
