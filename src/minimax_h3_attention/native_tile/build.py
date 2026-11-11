# Modified for minimax-h3-ROCm: package/build paths only; operator math preserved.
"""Compile only three original kernel instantiations, no GPU devices required."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=Path(os.environ.get('H3_NATIVE_TILE_CACHE', ROOT/'.runtime/cache/native-tile-v1'))
VENDOR=ROOT/'third_party/sageattention_rdna4'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import torch
    from torch.utils.cpp_extension import load
    frozen=json.loads((ROOT/'configs/runtime-lock.json').read_text())['runtime']
    assert torch.__version__==frozen['torch'] and torch.version.hip==frozen['hip']
    assert importlib.metadata.version('triton')==frozen['triton']
    assert not torch.cuda.is_available(),'CPU-only build expected'
    work=CACHE/'source';work.mkdir(parents=True,exist_ok=True)
    original=VENDOR/'csrc/qattn/qk_int_sv_gfx12_native.cu'
    raw=original.read_text();marker='} // namespace'
    assert raw.count(marker)==1
    prefix=raw[:raw.index(marker)+len(marker)]+'\n'
    # Only replace this relative include; the kernel definitions stay byte-exact.
    prefix=prefix.replace('#include "../reduction_utils.cuh"','#include "reduction_utils.cuh"')
    (work/'native_prefix.cuh').write_text(prefix)
    shutil.copyfile(VENDOR/'csrc/reduction_utils.cuh',work/'reduction_utils.cuh')
    shutil.copyfile(HERE/'entry.hip',work/'tile.cu')
    abi=int(torch._C._GLIBCXX_USE_CXX11_ABI)
    flags=['-O3','-ffast-math','-fgpu-flush-denormals-to-zero','-fno-offload-uniform-block',
           '-D__HIP_PLATFORM_AMD__=1','-U__HIP_NO_HALF_OPERATORS__','-U__HIP_NO_HALF_CONVERSIONS__',
           f'-D_GLIBCXX_USE_CXX11_ABI={abi}','-mllvm','--lsr-drop-solution=1',
           '-mllvm','-enable-post-misched=1','-mllvm','-amdgpu-early-inline-all=true',
           '-mllvm','-amdgpu-function-calls=false']
    build=CACHE/'build';build.mkdir(exist_ok=True)
    before=time.time_ns()
    module=load(name='h3_s09_tile_v1',sources=[str(work/'tile.cu')],extra_cflags=['-O3'],
                extra_cuda_cflags=flags,extra_include_paths=[str(work)],build_directory=str(build),
                with_cuda=True,verbose=True)
    files=[HERE/'entry.hip',Path(__file__),original,VENDOR/'csrc/reduction_utils.cuh',
           *sorted(work.glob('*')),build/'build.ninja',Path(module.__file__)]
    report={'status':'compiled_CPU_only','runtime':frozen,'GPU_work_issued':False,
            'started_unix_ns':before,'finished_unix_ns':time.time_ns(),'flags':flags,
            'extension':str(module.__file__),'source_shader_prefix_unchanged_except_include':True,
            'source_prefix_lines':len(prefix.splitlines()),
            'files_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    (CACHE/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S09_BUILD_COMPLETE',module.__file__,flush=True)


if __name__=='__main__':main()
