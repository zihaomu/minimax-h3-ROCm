"""Independent CPU dense-attention reference for the layout of the SDPA control."""
import torch

from minimax_h3_attention import AttentionConfig
from minimax_h3_attention.dispatch import implementation


def test_sdpa_nhd_matches_independent_dense_reference():
    generator = torch.Generator().manual_seed(9)
    q, k, v = [torch.randn((1, 11, 4, 128), generator=generator, dtype=torch.float64)
               for _ in range(3)]
    expected_heads = []
    for head in range(4):
        scores = q[0, :, head] @ k[0, :, head].T / (128 ** 0.5)
        expected_heads.append(scores.softmax(dim=-1) @ v[0, :, head])
    expected = torch.stack(expected_heads, dim=1).unsqueeze(0)
    actual = implementation(AttentionConfig(backend="torch_sdpa"))(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
