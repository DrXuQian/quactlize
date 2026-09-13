import copy
import pytest

from dev.gemv_ppu.review_moe_compare import validate_profile


def profile(tokens=1, split=4):
    case = dict(tokens=tokens, shape=[tokens * 8, 512, 2048],
                device=dict(l2_bytes=67108864),
                best=dict(tc=dict(key='tc:parent:s4'), simt=dict(key='simt:r2-v6-w8-p8')),
                screen=[dict(key='tc:parent:s4', execution=dict(split=split, grid=0, shared_bytes=38912))])
    parent = dict(tm=16, tn=64, wm=16, wn=16, persistent=False)
    names = ['void q4_s1::indexed_kernel<1, 2, 6, 8, 8, 512, 2048>(call)']
    if tokens == 1:
        names += ['runtime::indexed_prepare<16>', 'cutlass::device_kernel<subject>', 'runtime::indexed_finish<4>']
    else:
        names += ['q4_moe_compare::prepare', 'runtime::grouped_metadata<int>',
                  'moe_directory::build_kernel<8>', 'cutlass::device_kernel<subject>']
        if split > 1:
            names.append('CompactReductionKernel')
        names.append('q4_moe_compare::finish')
    rows = [dict(ID=str(i), **{'Kernel Name': n, 'device__attribute_cu_count': '72',
            'device__attribute_llc_cache_size': '67108864', 'ppu__time_duration.sum': '1000'})
            for i, n in enumerate(names)]
    rows[0].update({'launch__grid_size': str(tokens * 128), 'launch__block_size': '256',
                    'launch__shared_mem_per_block': '1024'})
    tc = next(r for r in rows if r['Kernel Name'].startswith('cutlass::'))
    tc.update({'launch__grid_size': str(tokens * 64 * split), 'launch__block_size': '128',
               'launch__shared_mem_per_block': '38912'})
    return rows, case, parent


@pytest.mark.parametrize('tokens,split', [(1, 4), (8, 1), (8, 4)])
def test_complete_call_profiles(tokens, split):
    validate_profile(*profile(tokens, split))


@pytest.mark.parametrize('plant', ['missing_reduce', 'swapped', 'geometry', 'cu', 'duration', 'simt_recipe', 'duplicate'])
def test_profile_negatives(plant):
    rows, case, parent = copy.deepcopy(profile(8, 4))
    if plant == 'missing_reduce':
        rows.pop(-2)
    elif plant == 'swapped':
        rows[0], rows[1] = rows[1], rows[0]
    elif plant == 'geometry':
        rows[4]['launch__grid_size'] = '1'
    elif plant == 'cu':
        rows[1]['device__attribute_cu_count'] = '1'
    elif plant == 'duration':
        rows[1]['ppu__time_duration.sum'] = 'nan'
    elif plant == 'simt_recipe':
        rows[0]['Kernel Name'] = rows[0]['Kernel Name'].replace('6, 8, 8', '7, 8, 8')
    elif plant == 'duplicate':
        rows[1]['ID'] = rows[0]['ID']
    with pytest.raises(ValueError):
        validate_profile(rows, case, parent)
