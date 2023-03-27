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
results from running [benchmark_flash_attention.py](../../benchmarks/benchmark_flash_attention.py)
with an A100.

* `CUDA` column refers to the CUDA FlashAttention implementation by Tri Dao in
[HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention).
Empty entries are because of unsupported inputs.
* `Triton` column refers to our Triton implementation, which is based on Tri Dao's in
[HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention). It
adds dropout support and does some additional optimizations.
* `Ref` column refers to a vanilla PyTorch implementation
* Row labels have the format
`(batch_sz, n_heads, seq_len, head_dim), p=dropout_p, bias=True/False, causal=True/False`
* using 12 heads per GPU to mimic a TP arrangement that is representative of our workload
(96 heads split over 8 GPUs).

For encoder use cases, the Triton version is typically faster. For decoder use cases,
which version is faster depends on the sequence length. The break-even point is around
4096, above which the Triton version begins to pull away.

```
[------------------------------------ fwd -------------------------------------]
                                                |   CUDA   |  Triton  |    Ref
48 threads: --------------------------------------------------------------------
      (1,12,2048,128), p=0.0, bias=0, causal=0  |   418.1  |   258.2  |    482.6
      (1,12,2048,128), p=0.0, bias=0, causal=1  |   215.5  |   188.7  |    797.3
      (1,12,2048,128), p=0.1, bias=0, causal=0  |   357.0  |   283.4  |    693.3
      (1,12,2048,128), p=0.1, bias=0, causal=1  |   225.8  |   230.7  |   1071.1
      (1,12,2048,128), p=0.0, bias=1, causal=0  |          |   232.7  |    694.8
      (1,12,2048,128), p=0.0, bias=1, causal=1  |          |   192.1  |   1023.8
      (1,12,2048,128), p=0.1, bias=1, causal=0  |          |   286.4  |    917.0
      (1,12,2048,128), p=0.1, bias=1, causal=1  |          |   233.0  |   1454.8
      (1,12,4096,128), p=0.0, bias=0, causal=0  |  1567.6  |   926.5  |   1720.0
      (1,12,4096,128), p=0.0, bias=0, causal=1  |   733.2  |   548.9  |   3015.5
      (1,12,4096,128), p=0.1, bias=0, causal=0  |  1571.3  |  1102.2  |   2568.1
      (1,12,4096,128), p=0.1, bias=0, causal=1  |   776.1  |   674.3  |   3878.5
      (1,12,4096,128), p=0.0, bias=1, causal=0  |          |   901.7  |   2586.0
      (1,12,4096,128), p=0.0, bias=1, causal=1  |          |   565.9  |   3894.2
      (1,12,4096,128), p=0.1, bias=1, causal=0  |          |  1112.0  |   3446.5
      (1,12,4096,128), p=0.1, bias=1, causal=1  |          |   681.9  |   4753.2
      (1,12,8192,128), p=0.0, bias=0, causal=0  |  6265.5  |  3415.7  |   6528.5
      (1,12,8192,128), p=0.0, bias=0, causal=1  |  3118.1  |  1788.7  |  11742.0
      (1,12,8192,128), p=0.1, bias=0, causal=0  |  6250.8  |  4244.1  |   9981.7
      (1,12,8192,128), p=0.1, bias=0, causal=1  |  3208.5  |  2230.5  |  15178.2
      (1,12,8192,128), p=0.0, bias=1, causal=0  |          |  3411.3  |  10022.5
      (1,12,8192,128), p=0.0, bias=1, causal=1  |          |  1847.7  |  15219.2
      (1,12,8192,128), p=0.1, bias=1, causal=0  |          |  4215.3  |  13450.6
      (1,12,8192,128), p=0.1, bias=1, causal=1  |          |  2258.7  |  18673.7

Times are in microseconds (us).

[------------------------------------- bwd --------------------------------------]
                                                |    CUDA   |   Triton  |    Ref
48 threads: ----------------------------------------------------------------------
      (1,12,2048,128), p=0.0, bias=0, causal=0  |   1168.4  |   1020.9  |   1018.3
      (1,12,2048,128), p=0.0, bias=0, causal=1  |    512.1  |    730.1  |   2117.8
      (1,12,2048,128), p=0.1, bias=0, causal=0  |    901.7  |    826.7  |   1205.2
      (1,12,2048,128), p=0.1, bias=0, causal=1  |    496.6  |    832.7  |   2324.1
      (1,12,2048,128), p=0.0, bias=1, causal=0  |           |    870.6  |   1016.0
      (1,12,2048,128), p=0.0, bias=1, causal=1  |           |    989.8  |   1417.5
      (1,12,2048,128), p=0.1, bias=1, causal=0  |           |    857.1  |   1199.9
      (1,12,2048,128), p=0.1, bias=1, causal=1  |           |    875.8  |   1606.1
      (1,12,4096,128), p=0.0, bias=0, causal=0  |   3451.0  |   2931.5  |   3702.0
      (1,12,4096,128), p=0.0, bias=0, causal=1  |   1695.6  |   2035.1  |   8043.9
      (1,12,4096,128), p=0.1, bias=0, causal=0  |   3306.5  |   2965.1  |   4437.4
      (1,12,4096,128), p=0.1, bias=0, causal=1  |   1658.6  |   1847.9  |   8781.8
      (1,12,4096,128), p=0.0, bias=1, causal=0  |           |   3130.8  |   3702.9
      (1,12,4096,128), p=0.0, bias=1, causal=1  |           |   2453.2  |   5311.4
      (1,12,4096,128), p=0.1, bias=1, causal=0  |           |   3058.4  |   4436.9
      (1,12,4096,128), p=0.1, bias=1, causal=1  |           |   1954.6  |   6046.8
      (1,12,8192,128), p=0.0, bias=0, causal=0  |  13290.3  |  11207.1  |  12955.3
      (1,12,8192,128), p=0.0, bias=0, causal=1  |   6098.6  |   6593.9  |  30240.6
      (1,12,8192,128), p=0.1, bias=0, causal=0  |  12714.6  |  11320.8  |  15860.4
      (1,12,8192,128), p=0.1, bias=0, causal=1  |   5932.4  |   5854.1  |  33147.9
      (1,12,8192,128), p=0.0, bias=1, causal=0  |           |  12037.3  |  12953.2
      (1,12,8192,128), p=0.0, bias=1, causal=1  |           |   7640.0  |  19357.2
      (1,12,8192,128), p=0.1, bias=1, causal=0  |           |  11619.2  |  15858.4
      (1,12,8192,128), p=0.1, bias=1, causal=1  |           |   6188.9  |  22254.6

Times are in microseconds (us).

[----------------------------------- fwd+bwd ------------------------------------]
                                                |    CUDA   |   Triton  |    Ref
48 threads: ----------------------------------------------------------------------
      (1,12,2048,128), p=0.0, bias=0, causal=0  |   1608.3  |   1354.6  |   1514.6
      (1,12,2048,128), p=0.0, bias=0, causal=1  |    729.5  |    938.6  |   2928.1
      (1,12,2048,128), p=0.1, bias=0, causal=0  |   1264.0  |   1132.7  |   1908.1
      (1,12,2048,128), p=0.1, bias=0, causal=1  |    732.9  |   1025.9  |   3340.2
      (1,12,2048,128), p=0.0, bias=1, causal=0  |           |   1129.0  |   1723.5
      (1,12,2048,128), p=0.0, bias=1, causal=1  |           |   1258.1  |   2456.8
      (1,12,2048,128), p=0.1, bias=1, causal=0  |           |   1161.1  |   2129.9
      (1,12,2048,128), p=0.1, bias=1, causal=1  |           |   1191.0  |   2864.4
      (1,12,4096,128), p=0.0, bias=0, causal=0  |   5066.4  |   3886.3  |   5456.5
      (1,12,4096,128), p=0.0, bias=0, causal=1  |   2453.1  |   2609.7  |  11107.8
      (1,12,4096,128), p=0.1, bias=0, causal=0  |   4906.3  |   4084.2  |   7051.3
      (1,12,4096,128), p=0.1, bias=0, causal=1  |   2457.9  |   2539.7  |  12701.2
      (1,12,4096,128), p=0.0, bias=1, causal=0  |           |   4055.6  |   6331.6
      (1,12,4096,128), p=0.0, bias=1, causal=1  |           |   3036.3  |   9245.2
      (1,12,4096,128), p=0.1, bias=1, causal=0  |           |   4186.3  |   7929.8
      (1,12,4096,128), p=0.1, bias=1, causal=1  |           |   2660.6  |  10838.6
      (1,12,8192,128), p=0.0, bias=0, causal=0  |  19637.8  |  14681.4  |  19570.4
      (1,12,8192,128), p=0.0, bias=0, causal=1  |   9292.8  |   8263.1  |  42050.2
      (1,12,8192,128), p=0.1, bias=0, causal=0  |  19050.0  |  15619.1  |  25906.9
      (1,12,8192,128), p=0.1, bias=0, causal=1  |   9185.2  |   8129.3  |  48408.0
      (1,12,8192,128), p=0.0, bias=1, causal=0  |           |  15479.7  |  23050.0
      (1,12,8192,128), p=0.0, bias=1, causal=1  |           |   9533.0  |  34650.7
      (1,12,8192,128), p=0.1, bias=1, causal=0  |           |  15894.8  |  29412.2
      (1,12,8192,128), p=0.1, bias=1, causal=1  |           |   8485.7  |  41020.9

Times are in microseconds (us).
```
