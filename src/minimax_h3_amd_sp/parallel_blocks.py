import time

import torch
import torch.distributed as dist

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention

from .partition import SequencePartition, localize_segments


_EXCHANGE_BUFFERS = {}
EXCHANGE_CALLS = 0


def _start_peer_exchange(send, receive_shape, peer, group=None):
    global EXCHANGE_CALLS
    key = (send.device, send.dtype, tuple(send.shape), tuple(receive_shape), peer)
    if key not in _EXCHANGE_BUFFERS:
        _EXCHANGE_BUFFERS[key] = (
            torch.empty(send.shape, dtype=send.dtype, device=send.device),
            torch.empty(receive_shape, dtype=send.dtype, device=send.device),
        )
    staging, receive = _EXCHANGE_BUFFERS[key]
    # The default stream's prior wait orders reuse after the preceding transfer;
    # RCCL records this stream before the next P2P batch. Buffers stay resident.
    staging.copy_(send)
    EXCHANGE_CALLS += 1
    operations = [
        dist.P2POp(dist.isend, staging, peer, group=group),
        dist.P2POp(dist.irecv, receive, peer, group=group),
    ]
    requests = dist.batch_isend_irecv(operations)
    return receive, requests


def _wait(requests):
    for request in requests:
        request.wait()


def _scale_shift(hidden, shift, scale, segments):
    for start, stop, row in segments:
        hidden[start:stop].mul_(1.0 + scale[row].to(hidden.dtype))
        hidden[start:stop].add_(shift[row].to(hidden.dtype))
    return hidden


def _gated_residual(hidden, update, gate, segments):
    for start, stop, row in segments:
        hidden[start:stop].addcmul_(update[start:stop], gate[row].to(hidden.dtype))
    return hidden


def _heads_to_local_sequence(attended, partition, rank):
    peer = partition.peer(rank)
    start, stop = partition.bounds(rank)
    peer_start, peer_stop = partition.bounds(peer)
    own_heads = attended[start:stop].contiguous()
    peer_heads, requests = _start_peer_exchange(
        attended[peer_start:peer_stop],
        own_heads.shape,
        peer,
    )
    _wait(requests)
    if rank == 0:
        return torch.cat((own_heads, peer_heads), dim=-1)
    return torch.cat((peer_heads, own_heads), dim=-1)


def attention_heads(attention, qkv, rope_freqs, options=None):
    options = options or {}
    inner = attention.heads * attention.head_dim
    query, key, value = qkv.split(inner, dim=-1)
    sequence = qkv.shape[0]
    value = value.view(sequence, attention.heads, attention.head_dim)

    query = query.view(1, sequence, attention.heads, attention.head_dim)
    key = key.view(1, sequence, attention.heads, attention.head_dim)
    query_weight = comfy.model_management.cast_to(
        attention.q_norm.weight,
        device=qkv.device,
    )
    key_weight = comfy.model_management.cast_to(
        attention.k_norm.weight,
        device=qkv.device,
    )
    rotation_width = rope_freqs.shape[-3] * 2
    comfy.quant_ops.ck.rms_rope_split_half_(
        query,
        key,
        rope_freqs,
        query_weight,
        key_weight,
        epsilon=attention.q_norm.eps,
        rot_dim=rotation_width,
    )

    query = query[0].transpose(0, 1).unsqueeze(0)
    key = key[0].transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    return optimized_attention(
        query,
        key,
        value,
        attention.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=options,
    ).squeeze(0)


def _parallel_attention(attention, local_hidden, rope_freqs, partition, rank, options):
    peer = partition.peer(rank)
    peer_hidden, requests = _start_peer_exchange(
        local_hidden,
        (partition.size(peer), local_hidden.shape[1]),
        peer,
    )

    local_qkv = attention.qkv_proj(local_hidden)
    _wait(requests)
    peer_qkv = attention.qkv_proj(peer_hidden)
    if rank == 0:
        qkv = torch.cat((local_qkv, peer_qkv), dim=0)
    else:
        qkv = torch.cat((peer_qkv, local_qkv), dim=0)
    del local_qkv, peer_qkv, peer_hidden
    attended = attention_heads(attention, qkv, rope_freqs, options)
    del qkv
    local_heads = _heads_to_local_sequence(attended, partition, rank)
    del attended
    return attention.out_proj(local_heads)


def _chunked_mlp(mlp, normalized, chunk_rows=4096):
    # Token rows are independent; retain full K and original ConvRot quantizer.
    # Bound the large gate/up activation at 768p instead of materializing all
    # 18,893/18,894 local rows at once. Never offload an activation to the CPU.
    output = torch.empty_like(normalized)
    for start in range(0, normalized.shape[0], chunk_rows):
        stop = min(start + chunk_rows, normalized.shape[0])
        output[start:stop] = comfy.ops.linear_input_act(
            mlp.fc2, mlp.fc1(normalized[start:stop]), "swiglu"
        )
    return output


def run_blocks(
    blocks,
    local_hidden,
    t_embedding,
    global_segments,
    rope_freqs,
    total_rows,
    rank,
    transformer_options=None,
    profile=False,
):
    partition = SequencePartition(total_rows)
    start, stop = partition.bounds(rank)
    segments = localize_segments(global_segments, start, stop)
    options = transformer_options or {}

    for index, block in enumerate(blocks):
        started = time.perf_counter()
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            block.adaln_proj(t_embedding)
        )
        normalized = _scale_shift(
            block.norm1(local_hidden),
            shift_attn,
            scale_attn,
            segments,
        )
        attention_output = _parallel_attention(
            block.attn,
            normalized,
            rope_freqs,
            partition,
            rank,
            options,
        )
        _gated_residual(local_hidden, attention_output, gate_attn, segments)
        del attention_output, normalized

        normalized = _scale_shift(
            block.norm2(local_hidden),
            shift_mlp,
            scale_mlp,
            segments,
        )
        mlp_output = _chunked_mlp(block.mlp, normalized)
        _gated_residual(local_hidden, mlp_output, gate_mlp, segments)
        del mlp_output, normalized

        if profile:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print(f"[amd-h3-sp] rank={rank} block={index} seconds={elapsed:.4f}")

    return local_hidden
