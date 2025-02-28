import pytest
import torch
from wcmha import WeightedCausalMultiheadAttention


@pytest.mark.parametrize(
    "batch_size, T, embed_dim, num_heads, alpha",
    [(2, 8, 16, 4, 1.0), (1, 10, 32, 8, 2.0), (3, 5, 12, 3, 0.5)],
)
def test_similarity_powerlaw(batch_size, T, embed_dim, num_heads, alpha):
    model = WeightedCausalMultiheadAttention(
        embed_dim, num_heads, f_type="spl", alpha=alpha
    )
    x = torch.randn(batch_size, T, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, T, embed_dim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "batch_size, T, embed_dim, num_heads, alpha",
    [(2, 8, 16, 4, 1.0), (1, 10, 32, 8, 2.0), (3, 5, 12, 3, 0.5)],
)
def test_weighted_powerlaw(batch_size, T, embed_dim, num_heads, alpha):
    model = WeightedCausalMultiheadAttention(
        embed_dim, num_heads, f_type="wpl", alpha=alpha
    )
    x = torch.randn(batch_size, T, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, T, embed_dim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "batch_size, T, embed_dim, num_heads, alpha, butterworth_tc",
    [(2, 8, 16, 4, 1.0, 5.0), (1, 10, 32, 8, 2.0, 10.0), (3, 5, 12, 3, 0.5, 3)],
)
def test_butterworth_filter(batch_size, T, embed_dim, num_heads, alpha, butterworth_tc):
    model = WeightedCausalMultiheadAttention(
        embed_dim,
        num_heads,
        f_type="butterworth",
        alpha=alpha,
        butterworth_tc=butterworth_tc,
    )
    x = torch.randn(batch_size, T, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, T, embed_dim)
    assert not torch.isnan(out).any()
