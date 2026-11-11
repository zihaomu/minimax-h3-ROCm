import torch

from minimax_h3_amd_sp.quant_shard import (
    attention_head_indices,
    shard_linear_columns,
    shard_linear_rows,
)


def test_attention_indices_keep_qkv_sections_in_head_order():
    index, start, stop = attention_head_indices(8, 4, rank=1)
    assert (start, stop) == (16, 32)
    assert index.tolist() == list(range(16, 32)) + list(range(48, 64)) + list(
        range(80, 96)
    )


def test_dense_row_shard_matches_reference_linear():
    layer = torch.nn.Linear(5, 12, bias=True)
    index = torch.tensor([1, 3, 8, 10])
    shard = shard_linear_rows(layer, index)
    x = torch.randn(7, 5)
    torch.testing.assert_close(shard(x), layer(x)[:, index])


def test_dense_column_shards_sum_to_full_projection():
    layer = torch.nn.Linear(8, 5, bias=False)
    left = shard_linear_columns(layer, 0, 4)
    right = shard_linear_columns(layer, 4, 8)
    x = torch.randn(7, 8)
    torch.testing.assert_close(left(x[:, :4]) + right(x[:, 4:]), layer(x))
