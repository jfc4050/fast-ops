# Multi-Head Attention

> 🚨 The CUTLASS FlashAttention implementation described below is still WIP.
in the meantime there's a Triton implementation [here](flash_attention_triton.py).
Unfortunately it's somewhat picky about input sizes
(needs head_dim 128 and seq_len multiple of 128). If it does accept your inputs
its the fastest FlashAttention kernel i'm aware of for sequence lengths > 4096
(the HazyResearch CUDA implementation is still a little faster for shorter
sequence lengths, but doesn't support attention bias).

**See FlashAttention paper:
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)**

As in the FlashAttention paper, this implementation is fully fused and has $O(n)$
memory complexity.
It is written with [CUTLASS](https://github.com/NVIDIA/cutlass).

We prioritize A100 (SM80) GPUs. This doesn't necessarily mean that other GPUs
aren't supported, but to be safe it's recommended to run the unit tests on your
own machine if you are using different hardware.

## Usage
See [unit tests](../../tests/test_flash_attention.py) for more usage examples.
```python
dtype = torch.bfloat16
device = "cuda"
Q = torch.rand(1, 12, 4096, 128, dtype=dtype, device=device)
K = torch.rand(1, 12, 4096, 128, dtype=dtype, device=device)
V = torch.rand(1, 12, 4096, 128, dtype=dtype, device=device)

O = FlashAttentionFunction.apply(Q, K, V)
```
### Tips
* If you are doing causal masking, use the `causal` flag rather than applying
it via the attention bias.
The kernel can use this as a signal to skip blocks it knows will be masked
out anyways and performance will improve by ~2x. We also get to avoid loading
the bias from DRAM.
* If you are doing sequence masking (e.g. sequence length different across
batch elements), use a
[Nested Tensor](https://pytorch.org/docs/stable/nested.html) rather than
masking out padding tokens via the attention bias. The kernel can use this
as a signal to skip computation. We also get to avoid loading the bias from DRAM.

## Limitations
Violating any of these constraints will result in an exception.
* `head_dim` must be <= 128. Otherwise tiles become too large to keep in SRAM.
If you'd like to use larger values of `head_dim`, consider using the
[xFormers](https://github.com/facebookresearch/xformers) implementation.
* `head_dim` must be a multiple of 8 - this is an alignment requirement to support
128b vectorized loads.
* The last two dimensions of `Q`, `K`, `V`, and `attn_bias` must be contiguous and
row-major 2D matrix
* Inputs must be half-precision floating points (float16 and bfloat16)

## Benchmarks
TODO.
