#!/usr/bin/env python3
"""Compare cold indexed SIMT and TC calls; no device compilation or policy update."""
import argparse
import csv
import ctypes as C
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_ppu import moe_compare as spec,moe_s1
from dev.gemv_ppu.run_moe_s1 import cases,case_id,fingerprint as simt_fingerprint
from dev.gemv_ppu.moe_compare_bench import SimtBench,TensorCore,Unsupported
from dev.gemv_ppu.moe_s1_bench import error
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK,checked
from dev.gemv_ppu.run import query_l2_attribute,resolve_l2
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange,acu_launch_command


def identity(args):
    files=['dev/gemv_ppu/run_moe_compare.py','dev/gemv_ppu/moe_compare_bench.py',
           'quactlize/dispatch/native.py','tools/run_kpack_decode_sweep.py',
           'tools/run_kpack_pack_gate.py']
    return dict(schema=spec.SCHEMA,tc_manifest=sha(args.bundle/'manifest.json'),
        simt=simt_fingerprint(SimpleNamespace(bundle=spec.SIMT_BUNDLE,l2_bytes=args.l2_bytes),None),
        harness={p:sha(ROOT/p) for p in files},screen_samples=5,rounds=6,samples=15,
        cases=108,scope='F32_INDEXED_COMPLETE_CALL_COLD_WEIGHTS',production_changed=False)


def validate_samples(r,count):
    xs=r['samples_us']
    if len(xs)!=count or any(not math.isfinite(x) or x<=0 for x in xs):raise ValueError('sample denominator/value differs')
    if r['median_us']!=statistics.median(xs):raise ValueError('sample median differs')


def finalists(screen):
    selected=[]
    for arm in ('simt','tc'):
        valid=[r for r in screen if r['arm']==arm and r['status']=='PASS']
        selected += [r['key'] for r in sorted(valid,key=lambda r:(r['median_us'],r['key']))[:2]]
    return selected


def validate_partial(p, ident, key, inventory):
    if p['identity']!=ident or p['case']!=key:raise ValueError('resume identity differs')
    allowed={x['key']:x for x in inventory}
    seen=set()
    for row in p['screen']:
        name=row['key']
        if name not in allowed or name in seen or row['arm']!=allowed[name]['arm']:raise ValueError('screen cell identity differs')
        seen.add(name)
        if row['status']=='PASS':
            validate_samples(row,5)
            proof=row['correctness']
            if not 0<=proof['error']<.005:raise ValueError('correctness missing')
            for control in ('zero_codes','zero_a','output_guard','gpu_ids_replay'):
                if proof.get(control)!='PASS':raise ValueError('correctness control missing')
        elif row['status']=='STRUCTURAL':
            if row.get('samples_us') or row.get('reason') not in ('INSUFFICIENT_K_TILES_PER_PIPELINE_SLICE','DEVICE_QUERY_UNSUPPORTED','NO_PERSISTENT_RESIDENCY'):
                raise ValueError('invalid structural record')
        else:raise ValueError('unrecognized successful checkpoint row')
    if set(p['failed']) & seen:raise ValueError('failed/success overlap')
    if not set(p['failed'])<=set(allowed):raise ValueError('unknown failed cell')
    for x in p.get('confirmation',[]):validate_samples(x,15)


def validate_result(r,ident,key,inventory):
    validate_partial(r,ident,key,inventory)
    if r.get('status')!='PASS' or r['failed']:raise ValueError('case is incomplete')
    if {x['key'] for x in r['screen']}!={x['key'] for x in inventory}:raise ValueError('screen incomplete')
    chosen=finalists(r['screen'])
    if len(r['confirmation'])!=6*len(chosen) or {(x['round'],x['key']) for x in r['confirmation']}!={(i,k) for i in range(6) for k in chosen}:
        raise ValueError('confirmation denominator differs')
    if set(r['best'])!={'simt','tc'}:raise ValueError('missing comparison arm')
    med={k:statistics.median(x['median_us'] for x in r['confirmation'] if x['key']==k) for k in chosen}
    for arm in ('simt','tc'):
        win=min((k for k in chosen if k.startswith(arm+':')),key=lambda k:(med[k],k))
        if r['best'][arm]['key']!=win or r['best'][arm]['median_us']!=med[win]:raise ValueError('best differs')
    delta=100*(r['best']['simt']['median_us']/r['best']['tc']['median_us']-1)
    if r['simt_vs_tc_pct']!=delta:raise ValueError('relative timing differs')
    _,n,k=r['shape']
    if r['copies']*8*n*k*9//16<2.25*r['device']['l2_bytes'] or r['calls_per_graph']%r['copies']:
        raise ValueError('cold active-weight ring differs')
    return r


def probe(args):
    sdk=SDK(args.sdk)
    lib=C.CDLL(str(spec.SIMT_BUNDLE/moe_s1.payload(*moe_s1.SHAPES[0])),mode=C.RTLD_LOCAL)
    fn=lib.q4_ppu_probe;fn.argtypes=[C.POINTER(C.c_int)]*3+[C.c_char_p];fn.restype=C.c_int
    l2,sm,warp,name=C.c_int(),C.c_int(),C.c_int(),C.create_string_buffer(256)
    checked(fn(C.byref(l2),C.byref(sm),C.byref(warp),name),'campaign device probe')
    return device_identity(sdk)|resolve_l2(l2.value,args.l2_bytes,query_l2_attribute(sdk.lib))|dict(
        name=name.value.decode(),properties_sm=sm.value,warp=warp.value)


def catalog(manifest,n,k,t,ch,cluster):
    key=case_id(n,k,t,ch,cluster)
    simt=[dict(key='simt:'+r,arm='simt',recipe=r,reason=None) for r in spec.prior_simt()[key]]
    tc=[dict(c,arm='tc',key='tc:'+c['key']) for c in spec.tc_inventory(manifest,n,k)]
    return simt+tc


def child(args,m):
    n,k,t,ch,cluster=args.case;cluster=bool(cluster)
    key=case_id(n,k,t,ch,cluster); ident=identity(args); inventory=catalog(m,n,k,t,ch,cluster)
    checkpoint=args.output/(key+'.partial.json');target=args.output/(key+'.json')
    p=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(identity=ident,case=key,screen=[],failed={},confirmation=[])
    validate_partial(p,ident,key,inventory)
    if args.retry_failed:
        p['failure_history']=p.get('failure_history',[])+list(p['failed'].values());p['failed']={}
        p['confirmation']=[]
    def save():checkpoint.write_text(json.dumps(p,indent=2)+'\n')
    base=None;active={};current=None
    try:
        base=SimtBench(SimpleNamespace(sdk=args.sdk,bundle=spec.SIMT_BUNDLE,l2_bytes=args.l2_bytes),n,k)
        base.setup(t,ch,cluster)
        if p.get('device',base.device)!=base.device:raise ValueError('resume physical device differs')
        device_file=args.output/'device.json'
        if device_file.exists() and json.loads(device_file.read_text())!=base.device:raise ValueError('campaign physical device differs')
        p['device']=base.device;save()
        records={r['parent']['symbol']:r for r in m['modules']}
        def make(c):
            if c['arm']=='simt':return None
            return TensorCore(base,args.bundle,records[c['parent']],c)
        def measure(c,instance,count):
            return base.measure(c['recipe'],count) if c['arm']=='simt' else instance.measure(count)
        if not args.profile:
            seen={r['key'] for r in p['screen']}
            for i,c in enumerate(inventory):
                if c['key'] in seen or c['key'] in p['failed']:continue
                current=c['key'];instance=None
                print(f'Q4_MOE_COMPARE_PROGRESS case={key} phase=screen cell={i+1}/{len(inventory)} key={current}',flush=True)
                try:
                    if c['reason']:raise Unsupported(c['reason'])
                    instance=make(c)
                    proof=base.correctness(c['recipe']) if c['arm']=='simt' else instance.correctness()
                    samples=measure(c,instance,5)
                    record=dict(key=current,arm=c['arm'],status='PASS',correctness=proof,samples_us=samples,median_us=statistics.median(samples),
                        execution=dict(scope='INDEXED_S1_NO_GATHER_SCATTER',split=1) if instance is None else instance.receipt())
                except Unsupported as e:record=dict(key=current,arm=c['arm'],status='STRUCTURAL',reason=str(e),samples_us=[])
                finally:
                    if instance:instance.close()
                p['screen'].append(record);save()
                print('Q4_MOE_COMPARE_SCREEN '+json.dumps({k:v for k,v in record.items() if k not in ('samples_us','correctness')}),flush=True)
            chosen=finalists(p['screen'])
            expected={(rr,name) for rr in range(6) for name in chosen}
            # A resumed screen can choose different finalists; keep old evidence
            # separately but never mix it into the current denominator.
            if any((x['round'],x['key']) not in expected for x in p['confirmation']):
                p['previous_confirmation']=p.get('previous_confirmation',[])+p['confirmation'];p['confirmation']=[];save()
        else:
            result=json.loads(target.read_text())
            validate_result(result,ident,key,inventory)
            chosen=[r['key'] for r in result['best'].values()]
        configs={c['key']:c for c in inventory}
        for name in chosen:
            current=name;active[name]=make(configs[name])
            if args.profile:
                c=configs[name];instance=active[name]
                if instance is None:base.case_r.fill(base.out_base,0xa5,base.output_bytes+32)
                else:instance.poison()
                launch=(lambda:base.invoke(c['recipe'])) if instance is None else instance.launch
                stream=base.case_r.stream if instance is None else instance.r.stream
                checked(launch(),'profile warmup excluded');base.sdk.synchronize(stream)
                with AcuRange(base.sdk):checked(launch(),'profile call');base.sdk.synchronize(stream)
                got=base.read() if instance is None else instance.read()
                if error(got,base.data)>=.005:raise ValueError('profile numeric failure')
                print('Q4_MOE_COMPARE_PROFILE '+json.dumps(dict(case=key,key=name,status='PASS')),flush=True)
        if args.profile:return 0
        done={(x['round'],x['key']) for x in p['confirmation']}
        for rr in range(6):
            for name in chosen[::1 if rr%2==0 else -1]:
                if (rr,name) in done:continue
                current=name;samples=measure(configs[name],active[name],15)
                p['confirmation'].append(dict(key=name,round=rr,samples_us=samples,median_us=statistics.median(samples)))
                save()
            print(f'Q4_MOE_COMPARE_PROGRESS case={key} phase=confirm round={rr+1}/6',flush=True)
        bykey={name:statistics.median(x['median_us'] for x in p['confirmation'] if x['key']==name) for name in chosen}
        best={}
        for arm in ('simt','tc'):
            keys=[name for name in chosen if configs[name]['arm']==arm]
            if keys:
                win=min(keys,key=lambda name:(bykey[name],name));med=bykey[win]
                rounds=[x['median_us'] for x in p['confirmation'] if x['key']==win]
                best[arm]=dict(key=win,median_us=med,round_medians_us=rounds,round_span_pct=100*(max(rounds)-min(rounds))/med)
        result=dict(p,best=best,status='PASS' if not p['failed'] and len(best)==2 else 'INCOMPLETE',
            shape=[t*8,n,k],tokens=t,channels=ch,router='cluster' if cluster else 'spread',
            active_experts=int(np.unique(base.data['expert']).size),max_rows=int(np.bincount(base.data['expert']).max()),
            copies=base.copies,calls_per_graph=base.calls_per_graph,weight_sha256=base.weight_sha256,
            first_launch='EXCLUDED',cache='ROTATING_ACTIVE_EXPERT_WEIGHTS_AT_LEAST_2_25_L2',production_changed=False)
        if len(best)==2:result['simt_vs_tc_pct']=100*(best['simt']['median_us']/best['tc']['median_us']-1)
        if result['status']=='PASS':validate_result(result,ident,key,inventory)
        target.write_text(json.dumps(result,indent=2)+'\n')
        print('Q4_MOE_COMPARE_RESULT '+json.dumps({k:result[k] for k in ('case','status','best','simt_vs_tc_pct') if k in result}),flush=True)
        return 0 if result['status']=='PASS' else 2
    except BaseException as e:
        if current and not args.profile:
            p['failed'][current]=dict(key=current,error=str(e),phase='screen_or_confirmation')
            p['screen']=[r for r in p['screen'] if r['key']!=current]
            p['confirmation']=[r for r in p['confirmation'] if r['key']!=current];save()
        raise
    finally:
        for x in active.values():
            if x:x.close()
        if base:base.close()


def main(args):
    m=spec.verify(args.bundle)
    for name,value in m['runtime'].items():
        if sha(args.sdk/'lib'/name)!=value:raise ValueError('runtime library differs: '+name)
    if args.case:return child(args,m)
    args.output.mkdir(parents=True,exist_ok=True);ident=identity(args);authority=args.output/'authority.json'
    if authority.exists() and json.loads(authority.read_text())!=ident:raise ValueError('campaign resume differs')
    authority.write_text(json.dumps(ident,indent=2)+'\n')
    device=probe(args);device_file=args.output/'device.json'
    if device_file.exists() and json.loads(device_file.read_text())!=device:raise ValueError('resume physical device differs')
    device_file.write_text(json.dumps(device,indent=2)+'\n')
    started=time.monotonic();failures=[];profiles=[];completed=0;measured=0
    for n,k in moe_s1.SHAPES:
        for t,ch,cluster in cases():
            key=case_id(n,k,t,ch,cluster);target=args.output/(key+'.json')
            command=[sys.executable,'-u',__file__,'--case',str(n),str(k),str(t),str(ch),str(int(cluster)),
                     '--sdk',str(args.sdk),'--bundle',str(args.bundle),'--output',str(args.output),'--l2-bytes',str(args.l2_bytes)]
            old=json.loads(target.read_text()) if target.exists() else None
            if old and old.get('identity')==ident and old.get('status')=='PASS':
                validate_result(old,ident,key,catalog(m,n,k,t,ch,cluster))
                if old['device']!=device:raise ValueError('completed case physical device differs')
                rc=0
                print(f'Q4_MOE_COMPARE_RESUME case={key}',flush=True)
            else:
                retry=True;previous=None
                for attempt in range(len(catalog(m,n,k,t,ch,cluster))+2):
                    with (args.output/(key+'.log')).open('a') as f:
                        process=subprocess.Popen(command+(['--retry-failed'] if retry else []),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        for line in process.stdout:f.write(line);f.flush();print(line,end='',flush=True)
                        rc=process.wait()
                    retry=False
                    if rc in (0,2):break
                    path=args.output/(key+'.partial.json')
                    checkpoint=json.loads(path.read_text()) if path.exists() else {}
                    progress=(len(checkpoint.get('screen',[])),len(checkpoint.get('failed',{})))
                    if progress==previous or not progress[1]:break
                    previous=progress
                    print(f'Q4_MOE_COMPARE_RESTART case={key} fresh_process=1 failed_cells_preserved=1',flush=True)
                measured+=1
            if rc:failures.append(dict(case=key,rc=rc))
            completed+=1;elapsed=time.monotonic()-started
            remaining=(elapsed/measured*(108-completed)) if measured else None
            print('Q4_MOE_COMPARE_ETA '+json.dumps(dict(completed=completed,total=108,elapsed_minutes=elapsed/60,
                remaining_minutes=None if remaining is None else remaining/60,method='OBSERVED_MEAN_CASE_WALL',advisory_only=True)),flush=True)
            if rc==0 and args.acu and (t,ch,cluster) in ((1,1,False),(8,8,True)):
                report=args.output/(key+'.acurep');log=args.output/(key+'.acu.log');receipt=args.output/(key+'.acu.json')
                oldprofile=json.loads(receipt.read_text()) if receipt.exists() else None
                if oldprofile and oldprofile.get('identity')==ident and all((args.output/oldprofile[f]).is_file() and sha(args.output/oldprofile[f])==oldprofile[f+'_sha256'] for f in ('report','log')):
                    profiles.append(oldprofile);continue
                if report.exists():report=args.output/(key+f'.{time.time_ns()}.acurep')
                with log.open('w') as f:prc=subprocess.run(acu_launch_command(args.acu,report,command+['--profile']),stdout=f,stderr=subprocess.STDOUT).returncode
                reports=[p for p in args.output.glob(report.name+'*') if p.is_file() and p.stat().st_size]
                entries=[json.loads(s.split(' ',1)[1]) for s in log.read_text(errors='replace').splitlines() if s.startswith('Q4_MOE_COMPARE_PROFILE ')]
                current_result=json.loads(target.read_text())
                expected_profile={v['key'] for v in current_result['best'].values()}
                if prc or len(reports)!=1 or len(entries)!=2 or {x.get('key') for x in entries}!=expected_profile or any(x.get('status')!='PASS' or x.get('case')!=key for x in entries):
                    failures.append(dict(case=key,phase='acu',rc=prc))
                else:
                    proof=dict(identity=ident,case=key,entries=entries,report=reports[0].name,report_sha256=sha(reports[0]),log=log.name,log_sha256=sha(log))
                    receipt.write_text(json.dumps(proof,indent=2)+'\n');profiles.append(proof)
    rows=[]
    for n,k in moe_s1.SHAPES:
        for t,ch,c in cases():
            p=args.output/(case_id(n,k,t,ch,c)+'.json')
            if p.exists():rows.append(json.loads(p.read_text()))
    with (args.output/'summary.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['case','status','simt_us','tc_us','simt_vs_tc_pct','simt_key','tc_key'])
        for r in rows:w.writerow([r['case'],r['status'],*[r['best'].get(a,{}).get('median_us') for a in ('simt','tc')],r.get('simt_vs_tc_pct'),*[r['best'].get(a,{}).get('key') for a in ('simt','tc')]])
    result=dict(status='PASS' if not failures and len(rows)==108 else 'INCOMPLETE',cases=len(rows),expected=108,
                failures=failures,profiles=profiles,seconds=time.monotonic()-started,production_changed=False,
                files={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name not in ('result.json','console.log')})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Q4_MOE_COMPARE_DONE '+json.dumps(result|dict(files='IN_RESULT_JSON')),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--bundle',type=Path,default=spec.TC_BUNDLE)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--case',nargs=5,type=int);p.add_argument('--acu',type=Path)
    p.add_argument('--profile',action='store_true');p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if a.case:
        n,k,t,ch,c=a.case
        if (n,k) not in moe_s1.SHAPES or (t,ch,bool(c)) not in cases() or c not in (0,1):p.error('undeclared case')
    try:raise SystemExit(main(a))
    except Exception:traceback.print_exc();raise SystemExit(1)
