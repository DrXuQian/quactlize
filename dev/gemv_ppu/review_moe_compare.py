#!/usr/bin/env python3
"""Review a returned indexed SIMT/TC gate and import ACU without a GPU."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_ppu import moe_compare as spec, moe_s1
from dev.gemv_ppu.run_moe_compare import identity, validate_result, catalog
from dev.gemv_ppu.run_moe_s1 import cases, case_id
from quactlize.runtime.compiler import sha


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def profile_kind(name):
    for text, kind in (
        ('q4_s1::indexed_kernel', 'simt'),
        ('q4_moe_compare::prepare', 'fallback-gather'),
        ('runtime::indexed_prepare', 'fused-prepare'),
        ('runtime::indexed_finish', 'fused-reduce-scatter'),
        ('q4_moe_compare::finish', 'fallback-scatter'),
        ('moe_directory::build_kernel', 'directory'),
        ('cutlass::device_kernel', 'tc'),
    ):
        if text in name:
            return kind
    if 'grouped_' in name and 'metadata<' in name:
        return 'metadata'
    if 'Reduction' in name or 'reduction' in name:
        return 'reducer'
    raise ValueError('unrecognized profiled kernel: ' + name)


def validate_profile(rows, case, parent):
    chosen = case['best']['tc']['key']
    receipt = next(x['execution'] for x in case['screen'] if x['key'] == chosen)
    expected = ['simt', 'fused-prepare', 'tc', 'fused-reduce-scatter']
    if case['tokens'] > 4:
        expected = ['simt', 'fallback-gather', 'metadata', 'directory', 'tc']
        if receipt['split'] > 1:
            expected.append('reducer')
        expected.append('fallback-scatter')
    require([profile_kind(x['Kernel Name']) for x in rows] == expected,
            'profile complete-call sequence differs')
    require(len({x['ID'] for x in rows}) == len(rows), 'duplicate profiled kernel')
    for r in rows:
        require(float(r['device__attribute_cu_count']) == 72, 'ACU CU identity differs')
        require(float(r['device__attribute_llc_cache_size']) == case['device']['l2_bytes'],
                'ACU L2 identity differs')
        duration = float(r['ppu__time_duration.sum'])
        require(math.isfinite(duration) and duration > 0, 'invalid profile duration')
    r = rows[0]
    recipe = moe_s1.lookup(case['best']['simt']['key'].split(':', 1)[1])
    _, n, k = case['shape']
    signature = f'indexed_kernel<1, {recipe.reader}, {recipe.variant}, {recipe.warps}, {recipe.values}, {n}, {k}>'
    require(signature in r['Kernel Name'], 'SIMT profile specialization differs')
    geometry = recipe.geometry(n, k, case['shape'][0])
    for field, key in (('launch__grid_size', 'grid'), ('launch__block_size', 'threads'),
                       ('launch__shared_mem_per_block', 'shared_bytes')):
        require(float(r[field]) == geometry[key], 'SIMT profile geometry differs: ' + field)
    tc = next(r for r in rows if profile_kind(r['Kernel Name']) == 'tc')
    block = 32 * (parent['tm'] // parent['wm']) * (parent['tn'] // parent['wn'])
    grid = receipt['grid'] if parent['persistent'] else (
        case['shape'][0] * ((n + parent['tn'] - 1) // parent['tn']) * receipt['split'])
    for field, value in (('launch__block_size', block), ('launch__grid_size', grid),
                         ('launch__shared_mem_per_block', receipt['shared_bytes'])):
        require(float(tc[field]) == value, 'TC profile geometry differs: ' + field)


def review(folder):
    manifest = spec.verify()
    result = json.loads((folder / 'result.json').read_text())
    authority = json.loads((folder / 'authority.json').read_text())
    device = json.loads((folder / 'device.json').read_text())
    require(authority == identity(SimpleNamespace(bundle=spec.TC_BUNDLE, l2_bytes=0)),
            'source/package identity differs')
    require(result['status'] == 'PASS' and not result['failures'] and
            result['cases'] == result['expected'] == 108, 'incomplete campaign')
    files = {p.name for p in folder.iterdir() if p.is_file()} - {'result.json', 'console.log'}
    require(files == set(result['files']), 'result file denominator differs')
    for name, value in result['files'].items():
        p = (folder / name).resolve(strict=True)
        require(p.parent == folder.resolve() and sha(p) == value, 'result hash differs: ' + name)
    checked = []
    counts = Counter()
    errors = {'simt': [], 'tc': []}
    warnings = []
    for n, k in moe_s1.SHAPES:
        for t, ch, cluster in cases():
            key = case_id(n, k, t, ch, cluster)
            c = json.loads((folder / (key + '.json')).read_text())
            validate_result(c, authority, key, catalog(manifest, n, k, t, ch, cluster))
            require(c['shape'] == [t * 8, n, k] and c['tokens'] == t and
                    c['channels'] == ch and c['router'] == ('cluster' if cluster else 'spread')
                    and c['device'] == device, 'case identity differs')
            partial = json.loads((folder / (key + '.partial.json')).read_text())
            require(all(partial[f] == c[f] for f in ('screen', 'confirmation', 'failed')),
                    'checkpoint and final result differ')
            logs = (folder / (key + '.log')).read_text().splitlines()
            terminal = [json.loads(s.split(' ', 1)[1]) for s in logs
                        if s.startswith('Q4_MOE_COMPARE_RESULT ')]
            require(terminal and terminal[-1]['status'] == 'PASS' and
                    terminal[-1]['best'] == c['best'], 'final log receipt differs')
            for x in c['screen']:
                counts[x['arm'] + '/' + x['status']] += 1
                if x['status'] == 'PASS':
                    errors[x['arm']].append(x['correctness']['error'])
                    counts['samples'] += len(x['samples_us'])
            counts['confirmation_records'] += len(c['confirmation'])
            counts['samples'] += sum(len(x['samples_us']) for x in c['confirmation'])
            for arm, best in c['best'].items():
                rounds = [x['median_us'] for x in c['confirmation'] if x['key'] == best['key']]
                span = 100 * (max(rounds) - min(rounds)) / statistics.median(rounds)
                require(rounds == best['round_medians_us'] and span == best['round_span_pct'],
                        'round spread differs')
                if span > 5:
                    warnings.append(dict(case=key, arm=arm, round_span_pct=span))
            checked.append(c)
    summary = list(csv.DictReader((folder / 'summary.tsv').open(), delimiter='\t'))
    require(len(summary) == 108, 'TSV denominator differs')
    for c, row in zip(checked, summary):
        require(row['case'] == c['case'] and row['status'] == c['status'] and
                float(row['simt_vs_tc_pct']) == c['simt_vs_tc_pct'], 'TSV case/delta differs')
        for arm in ('simt', 'tc'):
            require(float(row[arm + '_us']) == c['best'][arm]['median_us'] and
                    row[arm + '_key'] == c['best'][arm]['key'], 'TSV selected timing differs')
    expected = {case_id(n, k, t, ch, c) for n, k in moe_s1.SHAPES
                for t, ch, c in ((1, 1, False), (8, 8, True))}
    require(len(result['profiles']) == 12 and {x['case'] for x in result['profiles']} == expected,
            'ACU profile denominator differs')
    for profile in result['profiles']:
        require(profile['identity'] == authority, 'ACU harness identity differs')
        for field in ('report', 'log'):
            require(sha(folder / profile[field]) == profile[field + '_sha256'], 'ACU hash differs')
    receipt = dict(schema='quactlize.q4-moe-compare-review.v1', authority=authority,
                   device=device, result_sha256=sha(folder / 'result.json'),
                   verified_files=len(files), package_sources=len(manifest['source_hashes']),
                   counts=dict(counts), max_conditioned_error={a: max(v) for a, v in errors.items()},
                   simt_wins=sum(c['simt_vs_tc_pct'] < 0 for c in checked),
                   tc_wins=sum(c['simt_vs_tc_pct'] > 0 for c in checked),
                   within_5pct=sum(abs(c['simt_vs_tc_pct']) <= 5 for c in checked),
                   noisy_rounds=warnings, seconds=result['seconds'], production_changed=False)
    return receipt, checked, result['profiles']


def main(args):
    folder = args.results.resolve(strict=True)
    receipt, cases_, profiles = review(folder)
    command = ([str(args.loader), '--library-path', args.library_path] if args.loader else []) + [str(args.acu)]
    bycase = {c['case']: c for c in cases_}
    parents = {r['parent']['symbol']: r['parent'] for r in spec.verify()['modules']}

    def import_one(profile):
        process = subprocess.run(command + ['--import', str(folder / profile['report']),
                                 '--page', 'raw', '--csv'], check=True, capture_output=True, text=True)
        metrics = list(csv.DictReader(io.StringIO(process.stdout)))
        case = bycase[profile['case']]
        validate_profile(metrics, case, parents[case['best']['tc']['key'].split(':')[1]])
        owners = sorted(set(re.findall(r'^\s*(\d+):([^\n]+)$',
                                       (folder / profile['log']).read_text(), re.M)))
        return dict(case=profile['case'], report_sha256=profile['report_sha256'],
                    metrics=metrics, context_owners=owners)

    with ThreadPoolExecutor(max_workers=3) as pool:
        imported = list(pool.map(import_one, profiles))
    receipt.update(archive_sha256=sha(args.archive), profiles=len(imported),
                   profiled_kernels=sum(len(p['metrics']) for p in imported),
                   interference='CONTEXT_OWNER_WARNING_NOT_EXCLUSIVITY_OR_CONCURRENT_WORK_PROOF',
                   context_owners=sorted({tuple(o) for p in imported for o in p['context_owners']}))
    args.output.mkdir(parents=True, exist_ok=False)
    metrics_file = args.output / 'acu-metrics.json.gz'
    metrics_file.write_bytes(gzip.compress(json.dumps(imported).encode(), mtime=0))
    receipt['acu_metrics_sha256'] = sha(metrics_file)
    with (args.output / 'summary.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['case', 'tokens', 'channels', 'router', 'active_experts', 'max_rows',
                         'simt_us', 'tc_us', 'simt_vs_tc_pct', 'winner', 'simt_key', 'tc_key',
                         'simt_round_span_pct', 'tc_round_span_pct', 'tc_scope'])
        for c in cases_:
            chosen = c['best']['tc']['key']
            scope = next(x['execution']['scope'] for x in c['screen'] if x['key'] == chosen)
            writer.writerow([c['case'], c['tokens'], c['channels'], c['router'],
                             c['active_experts'], c['max_rows'],
                             *[c['best'][a]['median_us'] for a in ('simt', 'tc')],
                             c['simt_vs_tc_pct'], 'SIMT' if c['simt_vs_tc_pct'] < 0 else 'TC',
                             *[c['best'][a]['key'] for a in ('simt', 'tc')],
                             *[c['best'][a]['round_span_pct'] for a in ('simt', 'tc')], scope])
    receipt['summary_sha256'] = sha(args.output / 'summary.tsv')
    (args.output / 'review.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Q4_MOE_COMPARE_REVIEW VERIFIED ' + json.dumps({k: receipt[k] for k in
          ('counts', 'max_conditioned_error', 'simt_wins', 'tc_wins', 'profiles', 'profiled_kernels')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acu', type=Path, required=True)
    parser.add_argument('--loader', type=Path)
    parser.add_argument('--library-path')
    args = parser.parse_args()
    if bool(args.loader) != bool(args.library_path):
        parser.error('a private loader needs its matching library path')
    main(args)
