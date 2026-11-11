# Modified for minimax-h3-ROCm: package/build paths only; operator math preserved.
"""Two explicit BF16 HIP paths. Unsupported calls fail; no automatic precision fallback."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch

from .._frozen.amd_sage_attention import _validate
from .build import DEFAULT_INSTALL, HERE, ROOT

PATHS = ('int8_qk_fp8_pv', 'fp8_qk_fp8_pv')
_package = None
_manifest = None
_sk1 = {}


def load():
    global _package, _manifest
    if _package is not None:
        return _package
    install = Path(os.environ.get('IXMX_RDNA4_INSTALL', DEFAULT_INSTALL)).resolve()
    manifest = json.loads((install / 'build.json').read_text())
    upstream = json.loads((HERE / 'upstream.json').read_text())
    frozen = json.loads((ROOT / 'configs/runtime-lock.json').read_text())
    if manifest['commit'] != upstream['commit'] or manifest['runtime'] != frozen['runtime']:
        raise RuntimeError('Ix source/build runtime identity mismatch; rebuild the isolated install')
    if os.environ.get('DEEPINFER_IMAGE_ID') != manifest['image_id']:
        raise RuntimeError('Use the locked ROCm 10.1 launcher for Ix attention')
    if torch.__version__ != manifest['runtime']['torch'] or torch.version.hip != manifest['runtime']['hip']:
        raise RuntimeError('Ix Torch/HIP ABI mismatch')
    for file, expected in manifest['installed_sha256'].items():
        if hashlib.sha256((install / file).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Installed Ix artifact changed: {file}')
    if 'sageattention' in sys.modules:
        origin = Path(sys.modules['sageattention'].__file__).resolve()
        if origin.parent != install / 'sageattention':
            raise RuntimeError(f'Another sageattention package is already imported: {origin}')
    sys.path.insert(0, str(install))
    package = importlib.import_module('sageattention')
    if Path(package.__file__).resolve().parent != install / 'sageattention':
        raise RuntimeError('Resolved the wrong sageattention package')
    if not package.GFX12_NATIVE_ENABLED:
        raise RuntimeError('Ix native HIP extension did not load')
    _package, _manifest = package, manifest
    return package


def _check(q, k, v, profile):
    if profile != 'sageplus':
        raise ValueError('The Ix H3 adapter accepts only the explicitly selected sageplus experiment')
    _validate(q, k, v)
    if q.dtype != torch.bfloat16:
        raise ValueError('The integrated H3 paths require BF16 Q/K/V and return BF16')


def int8_qk_fp8_pv(q, k, v, profile='sageplus'):
    """Ix's inherited PR #368 INT8 QK / FP8 PV; SK1 cannot intercept this entry."""
    _check(q, k, v, profile)
    package = load()
    with torch.cuda.device(q.device):
        return package.sageattn_qk_int8_pv_gfx12_native(
            q, k, v, tensor_layout='NHD', is_causal=False, qk_quant_gran='per_warp',
            sm_scale=128 ** -.5, pv_accum_dtype='fp32+fp16', value_dtype='fp8',
            smooth_k=True, smooth_v=False, return_lse=False, sk1_backend=False)


def _load_sk1(device):
    load()
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _sk1:
        from sageattention.sk1_backend.sk1_loader import HipRuntime, Sk1Attn
        from sageattention.sk1_backend.hip_prologue import HipPrologue
        base = Path(_package.__file__).parent / 'sk1_backend'
        runtime = HipRuntime()
        if not runtime.same_as_torch['handle_equal']:
            raise RuntimeError('SK1 and Torch resolved different HIP runtime images')
        attention = Sk1Attn(str(base / 'attn_fp8qk_bf16_strided.gfx1201.hsaco'), rt=runtime,
                           symbols=('attn_fp8qk_bf16_strided_fwd',
                                    'attn_fp8qk_bf16_strided_fwd_causal'),
                           with_s=True, strided=True, head_dim=128, bm=128,
                           qo_dtype=torch.bfloat16)
        prologue = HipPrologue(str(base / 'kv_prologue.gfx1201.hsaco'), rt=runtime)
        _sk1[index] = (attention, prologue)
    return _sk1[index]


def fp8_qk_fp8_pv(q, k, v, profile='sageplus'):
    """Ix SK1: raw BF16 Q/O, FP8 QK / FP8 PV, HIP KV prologue, no fallback."""
    _check(q, k, v, profile)
    with torch.cuda.device(q.device):
        attention, prologue = _load_sk1(q.device)
        b, n, h, d = q.shape
        n_pad = (n + 63) // 64 * 64
        qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
        packed = prologue.quant_kv(kh, vh, n_pad=n_pad, smooth_k=True, int8=False)
        if packed is None:
            raise RuntimeError(f'Ix HIP KV prologue refused the call: {dict(prologue.refusal_reasons)}')
        k8, sk, v8t, sv = packed
        # Required by upstream for V outliers above 448; preserve the compensating epilogue S.
        s = sv.amax(dim=1).clamp_min(1.0)
        sv.div_(s.unsqueeze(1))
        output = torch.empty((b, n, h, d), dtype=q.dtype, device=q.device)
        attention.launch_strided(qh, k8, sk, v8t, sv, None, output.transpose(1, 2),
                                 h, n, n_pad, d ** -.5, False, s=s)
    return output


def select(path):
    if path not in PATHS:
        raise ValueError(f'Unknown Ix attention path: {path}; choose {PATHS}')
    return globals()[path]


def identity(path):
    select(path)
    load()
    return {'path': path, 'input_output_dtype': 'bfloat16', 'layout': 'NHD',
            'upstream_commit': _manifest['commit'], 'install': _manifest['install'],
            'runtime': _manifest['runtime'], 'binary_sha256': _manifest['binaries'],
            'automatic_fallback': False, 'smooth_k': True, 'smooth_v': False,
            'causal': False, 'head_dim': 128,
            'implementation': 'PR #368 inherited native HIP' if path == PATHS[0] else 'Ix SK1 BF16 strided native HIP',
            'precision_semantics': 'Candidate algorithm; output equivalence to frozen SagePlus requires validation'}
