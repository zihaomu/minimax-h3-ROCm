"""Trace-off, bitwise-gated real H3 attention replay; never changes defaults."""
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from safetensors.torch import load_file, save_file
from minimax_h3_attention._frozen.amd_sage_attention import quantize_qkv
from minimax_h3_attention.ix import backend
from minimax_h3_attention.native_tile.adapter import prepare, original
from minimax_h3_attention.dispatch import attention
from minimax_h3_attention.config import AttentionConfig

METHODS = {'private128':0, 'BR512':1, 'QG1_BR128':2, 'QG1_BR256':3}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def restore(cpu, meta):
    span = max(r['storage_offset']+sum((a-1)*s for a,s in zip(r['shape'],r['stride']))+1
               for r in meta.values())
    base = torch.empty(span, dtype=torch.bfloat16, device='cuda')
    values = []
    for name in ('q','k','v'):
        r = meta[name]
        view = base.as_strided(r['shape'], r['stride'], r['storage_offset'])
        view.copy_(cpu[name])
        values.append(view)
    return values


def compare(actual, expected):
    a, b = actual.contiguous(), expected.contiguous()
    delta = a.float()-b.float()
    return {'bits_equal':bool(torch.equal(a.view(torch.uint8),b.view(torch.uint8))),
        'finite':bool(torch.isfinite(a).all()),
        'relative_l2':float(delta.norm()/b.float().norm().clamp_min(1e-30)),
        'max_abs':float(delta.abs().max())}


def saved(folder, name, tensor):
    path = folder/(name+'.safetensors')
    save_file({'output':tensor.detach().cpu().contiguous()}, str(path))
    return {'file':path.name, 'sha256':sha(path)}


def timing(fn, repeats):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    for start, stop in zip(starts, stops):
        start.record()
        value = fn()
        stop.record()
    stops[-1].synchronize()
    wall_stop = time.perf_counter_ns()
    values = [a.elapsed_time(b) for a,b in zip(starts, stops)]
    return {'samples_ms':values,'median_ms':statistics.median(values),
        'wall_ms_per_call':(wall_stop-wall_start)/1e6/repeats}


def interleaved(funcs, repeats, groups=4):
    for fn in funcs.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    data = []
    names = [n for n in funcs if n != 'installed128']
    for group in range(groups):
        order = names if group%2==0 else names[::-1]
        sample = {'group':group, 'before':timing(funcs['installed128'],repeats),'candidates':{}}
        for name in order:
            sample['candidates'][name] = timing(funcs[name],repeats)
        sample['after'] = timing(funcs['installed128'],repeats)
        aa = sample['after']['median_ms']/sample['before']['median_ms']
        sample['AA_change_percent'] = (aa-1)*100
        sample['paired_speedups'] = {n:(sample['before']['median_ms']+sample['after']['median_ms'])/2/t['median_ms']
                                    for n,t in sample['candidates'].items()}
        data.append(sample)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'container-id').write_text(os.environ['HOSTNAME'])
    lock = json.loads((ROOT/'configs/runtime-lock.json').read_text())['runtime']
    assert torch.__version__==lock['torch'] and torch.version.hip==lock['hip']
    assert importlib.metadata.version('triton')==lock['triton']
    assert os.environ['DEEPINFER_IMAGE_ID']==lock['container_image_id']
    assert 'ROCP_TOOL_LIBRARIES' not in os.environ and 'RDNASCOPE_RUN_DIR' not in os.environ
    hardware = json.loads(Path(os.environ['H3_DIAG_HARDWARE']).read_text())
    assert torch.cuda.device_count()==2
    hip = ctypes.CDLL('libamdhip64.so.7')
    for i, d in enumerate(hardware['devices']):
        pci = ctypes.create_string_buffer(64)
        assert hip.hipDeviceGetPCIBusId(pci,64,i)==0 and pci.value.decode().lower()==d['pci']
        assert torch.cuda.get_device_properties(i).gcnArchName.split(':')[0]=='gfx1201'
    torch.set_num_threads(8)
    build = json.loads((args.build/'build.json').read_text())
    for ref, expected in build['files_sha256'].items():
        assert sha(ROOT/ref)==expected, ref
    spec = importlib.util.spec_from_file_location('h3_attention_spill_v1',build['extension'])
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    backend.load()
    index = json.loads(args.index.read_text())
    assert index['status']=='current_SP2_replay_inputs_qualified'
    report = {'status':'running','scope':'isolated real TP2 rank inputs, no communication or Standard',
        'trace_enabled':False,'hardware':hardware,'runtime':lock,'index_sha256':sha(args.index),
        'build_sha256':sha(args.build/'build.json'),'gate':'exact BF16 bits; rejected candidates are not timed',
        'expected_effect_percent':5,'AA_screen_percent':2,'cases':[],'modules':[],
        'resources':{n:dict(ext.resources(v)) for n,v in METHODS.items()}}
    output = args.output/'probe.json'
    write(output,report)
    try:
        with torch.inference_mode():
            for fixture in index['cases']:
                rank = fixture['rank']
                torch.cuda.set_device(rank)
                path = args.index.parent/fixture['folder']/fixture['qkv']['file']
                assert sha(path)==fixture['qkv']['sha256']
                q,k,v = restore(load_file(str(path)),fixture['qkv']['tensors'])
                assert tuple(q.shape)==(1,37787,28,128)
                valid_names = set(METHODS)
                fixture_id = f'r{rank}-nfe{fixture["nfe"]}-b{fixture["block"]}'
                for group in range(4):
                    begin = group*7
                    qg,kg,vg = [x[:,:,begin:begin+7] for x in (q,k,v)]
                    packed = quantize_qkv(qg,kg,vg,'sageplus')
                    prepared = prepare(packed)
                    assert prepared[0].shape==(1,7,37888,128)
                    reference = original(prepared).clone()
                    reference_full = reference[:,:,:37787].transpose(1,2).contiguous()
                    wrapped = attention(qg,kg,vg,AttentionConfig())
                    assert compare(wrapped,reference_full)['bits_equal']
                    row = {'id':fixture_id+f'-h{begin:02}','logical_rank':rank,
                        'physical_gpu':hardware['physical_gpus'][rank],
                        'source_sha256':fixture['qkv']['sha256'],'input_shape':list(qg.shape),
                        'input_stride':list(qg.stride()),'gates':{},'outputs':{}}
                    folder = args.output/row['id']
                    retain = fixture['nfe']==0 and group==0
                    if retain:
                        folder.mkdir()
                        row['outputs']['installed128'] = saved(folder,'installed128',reference_full)
                    funcs = {'installed128':lambda:original(prepared)}
                    for name,variant in METHODS.items():
                        fn = lambda variant=variant:ext.tile(*prepared,variant)
                        fn()
                        actual = prepared[3]
                        gate = compare(actual,reference)
                        row['gates'][name] = gate
                        if retain:
                            row['outputs'][name] = saved(folder,name,actual[:,:,:37787].transpose(1,2))
                        assert name!='private128' or gate['bits_equal'], 'Private BR128 control changed math'
                        if gate['bits_equal'] and gate['finite']:
                            funcs[name] = fn
                        else:
                            valid_names.discard(name)
                    row['timing'] = interleaved(funcs,repeats=7)
                    assert compare(original(prepared),reference)['bits_equal'], 'Reference changed after timing'
                    report['cases'].append(row)
                    write(output,report)
                    print('SPILL_CORE',row['id'],json.dumps({n:statistics.median(t['paired_speedups'][n] for t in row['timing']) for n in funcs if n!='installed128'}),flush=True)
                    del packed,prepared,reference,reference_full,wrapped,funcs,actual,fn
                    gc.collect()
                def candidate(variant):
                    result = torch.empty_like(q)
                    for start in range(0,28,7):
                        p = prepare(quantize_qkv(q[:,:,start:start+7],k[:,:,start:start+7],v[:,:,start:start+7],'sageplus'))
                        ext.tile(*p,variant)
                        result[:,:,start:start+7].copy_(p[3][:,:,:37787].transpose(1,2))
                    return result
                original_fn = lambda:attention(q,k,v,AttentionConfig())
                golden = original_fn()
                module = {'id':fixture_id,'gates':{},'outputs':{}}
                funcs = {'installed128':original_fn}
                for name in METHODS:
                    if name not in valid_names:
                        continue
                    fn = lambda variant=METHODS[name]:candidate(variant)
                    actual = fn()
                    gate = compare(actual,golden)
                    module['gates'][name] = gate
                    if gate['bits_equal'] and gate['finite']:
                        funcs[name] = fn
                module['timing'] = interleaved(funcs,repeats=3)
                assert compare(original_fn(),golden)['bits_equal']
                report['modules'].append(module)
                write(output,report)
                print('SPILL_MODULE',fixture_id,json.dumps({n:statistics.median(t['paired_speedups'][n] for t in module['timing']) for n in funcs if n!='installed128'}),flush=True)
                del q,k,v,qg,kg,vg,funcs,golden,actual,fn,candidate,original_fn
                gc.collect()
                torch.cuda.empty_cache()
        report['status']='complete'
    except BaseException:
        report.update(status='failed',traceback=traceback.format_exc())
        raise
    finally:
        write(output,report)


if __name__=='__main__':
    main()
