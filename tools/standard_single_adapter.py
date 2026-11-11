"""One GPU, the current SP2 math and quantizer boundaries, no communication."""
import torch
import comfy.model_management as mm
from types import SimpleNamespace

from minimax_h3_amd_sp.h3_forward import prepare_rank_zero, finish_rank_zero
from minimax_h3_amd_sp.partition import SequencePartition, localize_segments
from minimax_h3_amd_sp.quant_shard import attention_head_indices, shard_linear_rows
from minimax_h3_amd_sp import parallel_blocks as blocks
from minimax_h3_attention.config import AttentionConfig
from minimax_h3_attention.dispatch import attention
from minimax_h3_attention.runtime import STATE


def install(model):
    config = AttentionConfig.from_environment()
    dit = model.get_model_object("diffusion_model")
    assert len(dit.blocks) == 50 and all(b.attn.heads == 56 for b in dit.blocks)
    for block in dit.blocks:
        original = block.attn.qkv_proj
        projections = []
        for rank in (0, 1):
            index, _, _ = attention_head_indices(56, 128, rank, 2, device=original.weight.device)
            projections.append(shard_linear_rows(original, index))
        # Registered submodules preserve the ordinary model-loader/offloader lifecycle.
        # Only QKV weights are split; all other parameters are stored once on this GPU.
        block.attn.qkv_proj = torch.nn.ModuleList(projections)
    STATE["attention_operator_calls"] = 0

    def selected(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        assert mask is None and skip_reshape and heads == 28
        result = attention(*(x.transpose(1, 2) for x in (q, k, v)), config=config)
        STATE["attention_operator_calls"] += 1
        return result.reshape(result.shape[0], result.shape[1], -1)

    blocks.optimized_attention = selected
    last_timestep = None

    def forward(x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        nonlocal last_timestep
        current = float(timestep.flatten()[0])
        if last_timestep is None or current >= last_timestep:
            mm.load_models_gpu([model], memory_required=8 * 1024**3,
                               minimum_memory_required=8 * 1024**3, force_full_load=True)
        last_timestep = current
        options = transformer_options or {}
        hidden, time, rope, segments, state = prepare_rank_zero(dit, x, timestep, context, options, minimax_payload)
        partition = SequencePartition(hidden.shape[0])
        bounds = [partition.bounds(rank) for rank in (0, 1)]
        localized = [localize_segments(segments, start, stop) for start, stop in bounds]
        for block in dit.blocks:
            modulation = [tuple(block.adaln_proj(time)) for _ in (0, 1)]
            normalized = [blocks._scale_shift(block.norm1(hidden[start:stop]), mods[0], mods[1], local)
                          for (start, stop), local, mods in zip(bounds, localized, modulation)]
            # Two token partitions retain exactly the dual run's input-quantizer domains.
            # Execute the exact two 28-head projections and norm/attention geometries.
            proxy = SimpleNamespace(heads=28, head_dim=128,
                                    q_norm=block.attn.q_norm, k_norm=block.attn.k_norm)
            head_outputs = []
            for rank in (0, 1):
                own = block.attn.qkv_proj[rank](normalized[rank])
                peer = block.attn.qkv_proj[rank](normalized[1 - rank])
                qkv = torch.cat((own, peer) if rank == 0 else (peer, own), dim=0)
                del own, peer
                head_outputs.append(blocks.attention_heads(proxy, qkv, rope, options))
                del qkv
            attended = torch.cat(head_outputs, dim=-1)
            del normalized, head_outputs
            for (start, stop), local, mods in zip(bounds, localized, modulation):
                h = hidden[start:stop]
                update = block.attn.out_proj(attended[start:stop].contiguous())
                blocks._gated_residual(h, update, mods[2], local)
                del update
                normalized = blocks._scale_shift(block.norm2(h), mods[3], mods[4], local)
                # Reset 4096-row chunk boundaries independently for each virtual rank.
                update = blocks._chunked_mlp(block.mlp, normalized)
                blocks._gated_residual(h, update, mods[5], local)
                del normalized, update
            del attended
            STATE["attention_calls"] += 1  # One complete 56-head logical DiT block.
        return finish_rank_zero(dit, hidden, time, state, x[0].dtype, x[1].dtype)

    model.add_object_patch("diffusion_model._forward", forward)
    return model
