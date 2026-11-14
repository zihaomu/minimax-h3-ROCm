# Upstream sources

| Component | Repository / version | License / local copy |
|---|---|---|
| MultiGPU core / ComfyUI node | [Kaihui-AMD/ComfyUI-MiniMaxH3-MultiGPU](https://github.com/Kaihui-AMD/ComfyUI-MiniMaxH3-MultiGPU), `ebf3953f6385360ef211d93e4ecf23696599eaa3` | [MIT](licenses/MultiGPU-MIT.txt); src/minimax_h3_amd_sp/ |
| RDNA4 attention | [IxMxAMAR/SageAttention-RDNA4](https://github.com/IxMxAMAR/SageAttention-RDNA4), `7ead14d2744620429602da6c58e7803daf518321` | [Apache-2.0](sageattention_rdna4/LICENSE), [NOTICE](sageattention_rdna4/NOTICE); unchanged vendor source |
| Model / sampling / VAE framework | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI), `0764232429b8cfb10b79b6f186c8cb23e0b22897` | External dependency, not vendored |
| Checkpoint metadata | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), revision `54472221d27c695fca9621b0c3ed52f06c4af707` | Metadata only; weights stay in ignored models/ |
| Optional diffusion cache | [vipshop/cache-dit](https://github.com/vipshop/cache-dit), PyPI `cache-dit==1.5.3` with `diffusers==0.40.0` | External pure-Python wheels in ignored `.runtime/`; no upstream source vendored |

Ix inherits the INT8 path from SageAttention PR368. INT8 and FP8 controls remain
separate. The user-selected reduced-error composition uses the retained frozen
QKV encoding, native FP32 PV accumulation and direct BF16 output.
Frozen local QKV encoding and ROCm NVFP4 decoder adapter are copied byte-for-byte.
The optional Cache-DiT adapter in `src/minimax_h3_amd_sp/cache_dit.py` uses the
upstream DBCache policy and residual application. It adds per-request lifecycle
control and a two-rank element-weighted gate for unequal token shards. It does not
change the frozen attention encoding or the default no-cache path. Exact wheel
hashes and scope are in `configs/provenance/upstreams.json`.

The MultiGPU copy retains prior changes for 4096-row MLP chunks, persistent peer
buffers, shorter temporary lifetimes and failed-peer cleanup. Reorganization
changes worker imports and introduces a standalone attention dispatcher.
The S09 query-tile wrapper changes package/build paths; its HIP entry and vendor
kernel math are retained. Adapted Apache-derived files carry modification notices.

`configs/provenance/upstreams.json` records original/promoted hashes, separating unchanged
copies from adapted files. Models, binaries, old traces and experimental outputs
are excluded from the public tree.
