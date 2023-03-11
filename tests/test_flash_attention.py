import torch

from fast_ops.flash_attention.flash_attention import FlashAttentionFunction

def test_flash_attn():
    batch_sz = 1
    n_heads = 1
    seq_len = 16
    dtype = torch.bfloat16

    Q = torch.randn(batch_sz, n_heads, seq_len, seq_len, dtype=dtype)
    K = torch.randn(batch_sz, n_heads, seq_len, seq_len, dtype=dtype)
    V = torch.randn(batch_sz, n_heads, seq_len, seq_len, dtype=dtype)
    out = FlashAttentionFunction.apply(Q, K, V)

    print(out)
