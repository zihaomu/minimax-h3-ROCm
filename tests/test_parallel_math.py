import torch

from minimax_h3_amd_sp.partition import SequencePartition
from minimax_h3_amd_sp.quant_shard import (
    attention_head_indices,
    shard_linear_rows,
)


def _attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    return torch.matmul(scores.softmax(dim=-1), value)


def test_two_head_shards_reconstruct_dense_attention_projection():
    torch.manual_seed(7)
    sequence = 9
    hidden = 16
    heads = 4
    head_dim = 4
    inner = heads * head_dim
    x = torch.randn(sequence, hidden)
    qkv = torch.nn.Linear(hidden, inner * 3, bias=False)
    output = torch.nn.Linear(inner, hidden, bias=False)

    full_q, full_k, full_v = qkv(x).split(inner, dim=-1)
    full_q = full_q.view(sequence, heads, head_dim).transpose(0, 1)
    full_k = full_k.view(sequence, heads, head_dim).transpose(0, 1)
    full_v = full_v.view(sequence, heads, head_dim).transpose(0, 1)
    expected = output(
        _attention(full_q, full_k, full_v).transpose(0, 1).reshape(sequence, inner)
    )

    head_outputs = []
    for rank in (0, 1):
        index, _, _ = attention_head_indices(heads, head_dim, rank)
        qkv_shard = shard_linear_rows(qkv, index)
        local_heads = heads // 2
        local_inner = inner // 2
        query, key, value = qkv_shard(x).split(local_inner, dim=-1)
        query = query.view(sequence, local_heads, head_dim).transpose(0, 1)
        key = key.view(sequence, local_heads, head_dim).transpose(0, 1)
        value = value.view(sequence, local_heads, head_dim).transpose(0, 1)
        attended = _attention(query, key, value)
        attended = attended.transpose(0, 1).reshape(sequence, local_inner)
        head_outputs.append(attended)

    actual = output(torch.cat(head_outputs, dim=-1))
    torch.testing.assert_close(actual, expected)


def test_peer_reduce_scatter_slices_cover_full_sequence():
    partition = SequencePartition(9)
    assert partition.bounds(0) == (0, 5)
    assert partition.bounds(1) == (5, 9)
    assert partition.size(0) + partition.size(1) == 9
