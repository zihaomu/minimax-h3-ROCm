"""CPU-only private Ix query scheduling; frozen encoding/math is gated by replay."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torch.utils.cpp_extension import load
    lock = json.loads((ROOT/'configs/runtime-lock.json').read_text())['runtime']
    assert torch.__version__ == lock['torch'] and torch.version.hip == lock['hip']
    assert importlib.metadata.version('triton') == lock['triton']
    assert os.environ['DEEPINFER_IMAGE_ID'] == lock['container_image_id']
    assert not torch.cuda.is_available(), 'Build with no GPU devices exposed'
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=False)
    work = cache/'source'
    work.mkdir()
    vendor = ROOT/'third_party/sageattention_rdna4'
    original = vendor/'csrc/qattn/qk_int_sv_gfx12_native.cu'
    raw = original.read_text()
    marker = '} // namespace'
    assert raw.count(marker) == 1
    prefix = raw[:raw.index(marker)+len(marker)]+'\n'
    prefix = prefix.replace('#include "../reduction_utils.cuh"', '#include "reduction_utils.cuh"')
    # The upstream template permits QGroups=1, but the non-streamed FP8 PV
    # branch still unconditionally indexes group1. Make only that group access
    # conditional; QGroups=2 retains the same expressions and operation order.
    start = prefix.index('__global__ __launch_bounds__(BlockRows * (2 / QGroupsParam), 1) void qk_int8_sv_f8_native_2q_kernel(')
    pv_start = prefix.index('      if constexpr (!UseStreamedFp8Pv) {',start)
    pv_end = prefix.index('\n    } else {',pv_start)
    pv = prefix[pv_start:pv_end]
    changes = {
        'bool q1_fully_future =\n            ApplyCausalMask && k_col_start >= q_start[1] + RM;':
        'bool q1_fully_future = true;\n        if constexpr (QGroups == 2) q1_fully_future = ApplyCausalMask && k_col_start >= q_start[1] + RM;',
        'acc1[elem] = out_frag[1][vdt][elem];':
        'if constexpr (QGroups == 2) acc1[elem] = out_frag[1][vdt][elem];',
        'if (!q1_fully_future) {':
        'if constexpr (QGroups == 2) if (!q1_fully_future) {',
        'out_frag[1][vdt][elem] = acc1[elem];':
        'if constexpr (QGroups == 2) out_frag[1][vdt][elem] = acc1[elem];',
    }
    for old,new in changes.items():
        assert pv.count(old)==1,old
        pv=pv.replace(old,new)
    prefix=prefix[:pv_start]+pv+prefix[pv_end:]
    (work/'native_prefix.cuh').write_text(prefix)
    shutil.copyfile(vendor/'csrc/reduction_utils.cuh', work/'reduction_utils.cuh')
    shutil.copyfile(ROOT/'tools/h3_attention_spill_entry.hip', work/'entry.cu')
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    flags = ['-O3','-ffast-math','-fgpu-flush-denormals-to-zero','-fno-offload-uniform-block',
        '-D__HIP_PLATFORM_AMD__=1','-U__HIP_NO_HALF_OPERATORS__','-U__HIP_NO_HALF_CONVERSIONS__',
        f'-D_GLIBCXX_USE_CXX11_ABI={abi}','-mllvm','--lsr-drop-solution=1',
        '-mllvm','-enable-post-misched=1','-mllvm','-amdgpu-early-inline-all=true',
        '-mllvm','-amdgpu-function-calls=false']
    build = cache/'build'
    build.mkdir()
    start = time.time_ns()
    module = load(name='h3_attention_spill_v1',sources=[str(work/'entry.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=flags,extra_include_paths=[str(work)],
        build_directory=str(build),with_cuda=True,verbose=True)
    files = [Path(__file__),ROOT/'tools/h3_attention_spill_entry.hip',original,
        vendor/'csrc/reduction_utils.cuh',*work.iterdir(),build/'build.ninja',Path(module.__file__)]
    record = {'status':'compiled_CPU_only','runtime':lock,'GPU_work_issued':False,
        'extension':str(Path(module.__file__).resolve()),'flags':flags,
        'started_unix_ns':start,'finished_unix_ns':time.time_ns(),
        'private_source_changes':['include path','four compile-time guards for group1 access in non-streamed PV'],
        'QGroups2_expressions_and_KV_order_retained':True,
        'variants':{'private128':[128,2],'BR512':[512,2],'QG1_BR128':[128,1],'QG1_BR256':[256,1]},
        'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    (cache/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print('SPILL_BUILD_COMPLETE',module.__file__,flush=True)


if __name__ == '__main__':
    main()
