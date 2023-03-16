# Multi-Head Attention

**See FlashAttention paper:
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)**

As in the FlashAttention paper, this implementation is fully fused and has O(n) memory complexity.
It is written with [CUTLASS](https://github.com/NVIDIA/cutlass).

We prioritize A100 (SM80) GPUs. This doesn't necessarily mean that other GPUs
aren't supported, but to be safe it's recommended to run the unit tests on your own machine
if you are using different hardware.

## Usage Tips
* If you are doing causal masking, use the `causal` flag rather than applying it via `attn_bias`.
The kernel can use this as a signal to skip blocks it knows will be masked out anyways and performance will improve by ~2x.

## Limitations
Violating these constraints will result in an exception.
* `head_dim` must be <= 128. Otherwise tiles become too large to keep in SRAM. If you'd like to use larger values of `head_dim`,
consider using the [xFormers](https://github.com/facebookresearch/xformers) implementation.
* `head_dim` must be a multiple of 8 - this is an alignment requirement to support 128b, half-precision, vectorized loads.
* The last two dimensions of `Q`, `K`, `V`, and `attn_bias` must be contiguous 2D matrix.
* Inputs must be half-precision floating points (float16 and bfloat16)

## Benchmarks
TODO.
