# Modified for minimax-h3-ROCm: package/build paths only; operator math preserved.
"""Explicit native BR choice, unchanged frozen encoding / BC64 / output ABI."""
import importlib.util
import json
from pathlib import Path

from .build import CACHE,ROOT,digest

_extension=None


def extension():
    global _extension
    if _extension is None:
        record=json.loads((CACHE/'build.json').read_text())
        for ref,expected in record['files_sha256'].items():assert digest(ROOT/ref)==expected,ref
        spec=importlib.util.spec_from_file_location('h3_s09_tile_v1',record['extension'])
        _extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(_extension)
    return _extension


def prepare(packed, q_multiple=128):
    import torch
    import torch.nn.functional as F
    qi,ki,v8,qs,ks,vs=packed
    b,h,n,d=qi.shape
    qpad=(n+q_multiple-1)//q_multiple*q_multiple;kpad=(n+63)//64*64
    qp=F.pad(qi,(0,0,0,qpad-n)).contiguous()
    kp=F.pad(ki,(0,0,0,kpad-n)).contiguous()
    vp=F.pad(v8.view(torch.uint8),(0,0,0,kpad-n)).transpose(2,3).contiguous()
    qsp=F.pad(qs,(0,qpad//32-qs.shape[-1]),value=1e-7/127).contiguous()
    out=torch.empty((b,h,qpad,d),device=qi.device,dtype=torch.bfloat16)
    return qp,kp,vp,out,qsp,ks,vs,n


def original(prepared):
    import importlib
    q,k,v,out,qs,ks,vs,n=prepared
    native=importlib.import_module('sageattention.core')._require_gfx12_native()
    native.qk_int8_sv_f8_scaled_native_attn(q,k,v,out,qs,ks,vs,1,0,128**-.5,n)
    return out


def core(prepared,br):
    assert br in (128,256,512)
    extension().tile(*prepared,br)
    return prepared[3]


def attention(q,k,v,br):
    import torch
    from .._frozen.amd_sage_attention import quantize_qkv
    from ..ix import backend
    backend._check(q,k,v,'sageplus');backend.load()
    with torch.cuda.device(q.device):
        p=prepare(quantize_qkv(q,k,v,'sageplus'))
        core(p,br)
        return p[3][:,:,:q.shape[1]].transpose(1,2).contiguous()
