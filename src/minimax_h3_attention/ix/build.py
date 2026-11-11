# Modified for minimax-h3-ROCm: package/build paths only; operator math preserved.
"""Build the pinned Ix sources against the locked runtime, without changing site-packages."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_INSTALL = ROOT / '.runtime/cache/standard-rocm10/gfx1201/ixmx_rdna4/install'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    frozen = json.loads((ROOT / 'configs/runtime-lock.json').read_text())
    runtime = frozen['runtime']
    assert torch.__version__ == runtime['torch'] and torch.version.hip == runtime['hip']
    assert importlib.metadata.version('triton') == runtime['triton']
    assert os.environ['DEEPINFER_IMAGE_ID'] == runtime['container_image_id']
    upstream = json.loads((HERE / 'upstream.json').read_text())
    for file, expected in upstream['files_sha256'].items():
        assert digest(ROOT / 'third_party/sageattention_rdna4' / file) == expected, file
    install = Path(os.environ.get('IXMX_RDNA4_INSTALL', DEFAULT_INSTALL)).resolve()
    work = install.parent / 'source-build'
    if not work.exists():
        shutil.copytree(ROOT / 'third_party/sageattention_rdna4', work)
    else:
        for file, expected in upstream['files_sha256'].items():
            assert digest(work / file) == expected, ('Stale modified build source', file)
    # Explicit architecture avoids setup.py's GPU discovery. No GPU kernels run in this builder.
    env = dict(os.environ, PYTORCH_ROCM_ARCH='gfx1201', GPU_ARCHS='gfx1201',
               MAX_JOBS='6', EXT_PARALLEL='1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='')
    started = time.time_ns()
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=work, env=env, check=True)
    subprocess.run([sys.executable, 'tools/build_hsaco.py', '--out-dir',
                    str(work / 'sageattention/sk1_backend')], cwd=work, env=env, check=True)
    # Source installation preserves the build stamp, CPython suffixes, and all rebuilt HSACOs.
    # setup.py's wheel metadata excludes Python 3.13; build_ext itself works without changing it.
    install.mkdir(parents=True, exist_ok=True)
    package = install / 'sageattention'
    if package.exists():
        raise FileExistsError(f'{package} already exists; choose another IXMX_RDNA4_INSTALL')
    shutil.copytree(work / 'sageattention', package,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    binaries = sorted(p for p in package.rglob('*') if p.suffix in ('.so', '.hsaco'))
    assert len([p for p in binaries if p.suffix == '.so']) == 2
    assert len([p for p in binaries if p.suffix == '.hsaco']) == 7
    manifest = {'status': 'built', 'install': str(install), 'upstream': upstream['repository'],
                'commit': upstream['commit'], 'runtime': runtime, 'python': platform.python_version(),
                'image_id': os.environ['DEEPINFER_IMAGE_ID'], 'build_started_ns': started,
                'build_finished_ns': time.time_ns(), 'build_gpu_work': False,
                'packaging': 'isolated source installation; no pip dependency changes or metadata bypass',
                'source_sha256': upstream['files_sha256'],
                'installed_sha256': {str(p.relative_to(install)): digest(p) for p in package.rglob('*')
                                     if p.is_file()},
                'binaries': {str(p.relative_to(install)): digest(p) for p in binaries}}
    (install / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('IXMX_BUILD_COMPLETE', install, flush=True)


if __name__ == '__main__':
    main()
