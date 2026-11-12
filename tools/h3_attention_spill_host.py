"""Shared-lock launcher for bounded, isolated attention spill experiments."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time

from rdnascope_h3_host import ROOT,WORKSPACE,LEASES,REGISTRY,choose,sha,write,stop_own_container
from rdnascope_h3_ab_host import OwnedMonitor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',required=True)
    parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args()
    if Path(args.session).name!=args.session or args.session in ('.','..'):
        parser.error('session must be a single new name')
    session=ROOT/'runs'/args.session
    session.mkdir(parents=True,exist_ok=False)
    index=ROOT/'archive/legacy-20261111/minimax_h3_multi_gpu/results/sol-s00-diag-20261106-pair26/replay-input-index.json'
    fixture=json.loads(index.read_text())
    payloads={str(index.relative_to(ROOT)):sha(index)}
    for row in fixture['cases']:
        p=index.parent/row['folder']/row['qkv']['file']
        assert sha(p)==row['qkv']['sha256']
        payloads[str(p.relative_to(ROOT))]=row['qkv']['sha256']
    write(session/'plan.json',{'goal':'Test fewer query groups per wave and BR512 control; fixed real shape/type/frozen math',
        'hypothesis':'Reduce long-lived FP32 accumulators/Q fragments and spills while retaining KV tile order',
        'variants':['installed128','private128','BR512','QG1_BR128','QG1_BR256'],
        'gate':'raw BF16 bits exact for every case; no timing for failing candidates',
        'cases':'6 real TP2 rank fixtures x4 head groups;6 full28-head adapters',
        'timing':'trace off,3 warmups,4 forward/reverse groups;7 core or3 module samples per method;installed control before/after each group',
        'AA_screen_percent':2,'expected_effect_percent':5,'budget_seconds':1200,'max_output_bytes':3221225472,
        'default_changed':False,'full_Standard':False})
    write(session/'source-manifest.json',{'fixtures':payloads,'sources':{str(p.relative_to(ROOT)):sha(p) for p in
        [ROOT/'tools/h3_attention_spill_probe.py',ROOT/'tools/h3_attention_spill_host.py',
         ROOT/'tools/h3_attention_spill_entry.hip',ROOT/'tools/h3_attention_spill_build.py',
         *ROOT.joinpath('src/minimax_h3_attention').rglob('*.py')]},
        'build_manifest':{'path':str(args.build.resolve()/'build.json'),'sha256':sha(args.build/'build.json')}})
    with ExitStack() as stack:
        pair,snapshot=choose(stack,session)
        devices=[REGISTRY['gpu_map'][str(i)] for i in pair]
        write(session/'hardware.json',{'physical_gpus':pair,'devices':devices,'preflight':snapshot,'excluded':[3]})
        lease=LEASES/f'gpu-reservation-h3-attention-spill-{os.getpid()}.json'
        row={'owner':f'h3-attention-spill-{os.getpid()}','status':'active','physical_gpus':pair,
             'purpose':'isolated real attention spill replay','evidence':str(session),'updated_unix_ns':time.time_ns()}
        write(lease,row)
        output=session/'probe'
        env={'ROCR_VISIBLE_DEVICES':','.join(d['uuid'] for d in devices),'HIP_VISIBLE_DEVICES':'0,1',
             'H3_DIAG_HARDWARE':str(session/'hardware.json'),
             'PYTHONPATH':':'.join(map(str,[ROOT/'src',ROOT/'.runtime/deps/standard-rocm10'])),
             'PYTHONPYCACHEPREFIX':str(ROOT/'.runtime/cache/h3-attention-spill/pycache'),
             'TRITON_CACHE_DIR':str(ROOT/'.runtime/cache/h3-swiglu-standard/triton'),
             'OMP_NUM_THREADS':'8'}
        command=[str(WORKSPACE/'rocm10-launch.sh'),'env',*[f'{k}={v}' for k,v in env.items()],
                 'python',str(ROOT/'tools/h3_attention_spill_probe.py'),'--build',str(args.build.resolve()),
                 '--index',str(index),'--output',str(output)]
        write(session/'launch.json',{'command':command,'trace_enabled':False})
        try:
            with OwnedMonitor(ROOT,REGISTRY,list(map(str,pair)),session/'monitor') as monitor:
                monitor.container_path=output/'container-id'
                start=time.monotonic()
                reason=None
                with (session/'console.log').open('x') as stream:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        latest=json.loads((session/'monitor/latest.json').read_text())
                        if latest['external_pids'] or latest['errors'] or time.time_ns()-latest['unix_ns']>2_000_000_000:
                            reason='resource_contamination'
                        elif time.monotonic()-start>1200:
                            reason='timeout'
                        elif sum(p.stat().st_size for p in session.rglob('*') if p.is_file())>3221225472:
                            reason='output_budget'
                        if reason:
                            stop_own_container(monitor.container_path)
                            process.terminate()
                            break
                        time.sleep(.5)
                    code=process.wait(timeout=30)
                write(session/'execution.json',{'returncode':code,'reason':reason,'elapsed_seconds':time.monotonic()-start})
                assert code==0 and reason is None
                assert json.loads((output/'probe.json').read_text())['status']=='complete'
                print('SPILL_REPLAY_COMPLETE',session,flush=True)
        finally:
            write(lease,{**row,'status':'released','updated_unix_ns':time.time_ns()})


if __name__=='__main__':
    main()
