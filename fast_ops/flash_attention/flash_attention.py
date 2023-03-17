from typing import Tuple

from torch import Tensor
from torch.autograd import Function
import torch.utils.cpp_extension as cpp_extension

# monkey patching this for now so I can add in my own arch flags.
# for some reason even when building with TORCH_CUDA_ARCH_LIST=8.0
# it builds for an older architecture and thinks it only has 49152B (0xc000)
# of shared memory.
cpp_extension._get_cuda_arch_flags = lambda: []

flash_attention_ext = cpp_extension.load(
    name="flash_attention",
    sources=[
        "fast_ops/flash_attention/flash_attention.cpp",
        "fast_ops/flash_attention/flash_attention_fwd.cu",
    ],
    extra_include_paths=["third-party/cutlass/include"],
    extra_cflags=["-std=c++17"],
    # see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
    extra_cuda_cflags=["--threads", "0", "-std=c++17", "--gpu-architecture=compute_80"],
    with_cuda=True,
    verbose=True,
)


class FlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        :param Q: (B, H, M, D) Query tensor.
        :param K: (B, H, N, D) Key tensor.
        :param V: (B, H, N, D) Value tensor.
        :returns: (B, H, M, D)
        """
        # TODO. handle dynamic dispatch here
        return flash_attention_ext.forward(Q, K, V)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param dO: (B, H, M, D) Output gradient
        """
        raise NotImplementedError


def multi_head_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    """
    See FlashAttention paper:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    https://arxiv.org/pdf/2205.14135.pdf

    for shape documentation we use the following notation:
    - B: batch size
    - H: number of heads
    - M: number of queries (sequence length)
    - N: number of keys (sequence length)
    - D: head dim

    Limitations:
    Violating any of these constraints will result in an exception.
    * `head_dim` must be <= 128. Otherwise tiles become too large to keep in SRAM.
      If you'd like to use larger values of `head_dim`, consider using the
      xFormers implementation (https://github.com/facebookresearch/xformers).
    * `head_dim` must be a multiple of 8 - this is an alignment requirement to support
      128b vectorized loads.
    * The last two dimensions of `Q`, `K`, `V`, and `attn_bias` must be contiguous and
      row-major 2D matrix
    * Inputs must be half-precision floating points (float16 and bfloat16)

    :param Q: (B, H, M, D) Query tensor.
    :param K: (B, H, N, D) Key tensor.
    :param V: (B, H, N, D) Value tensor.
    :returns: (B, H, M, D)
    """
    return FlashAttentionFunction.apply(Q, K, V)
