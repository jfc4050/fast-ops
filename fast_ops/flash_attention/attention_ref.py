import math
from typing import Optional

import einops
import torch
from torch import Tensor


def attention_ref(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    query_padding_mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    dropout_mask: Optional[Tensor] = None,
    causal: bool = False,
    bias: Optional[Tensor] = None,
    upcast: bool = True,
    reorder_ops: bool = False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        bias: (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if key_padding_mask is not None:
        scores.masked_fill_(einops.rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(einops.rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
        attention = attention.masked_fill(
            einops.rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)
