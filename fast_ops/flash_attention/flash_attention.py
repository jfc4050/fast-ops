from typing import Tuple

from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load


flash_attention_ext = load(
    name="flash_attention",
    sources=[
        "fast_ops/flash_attention/flash_attention.cpp",
        "fast_ops/flash_attention/flash_attention_fwd.cu",
    ],
    extra_include_paths=["third-party/cutlass/include"],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["--threads", "0"],
    with_cuda=True,
    verbose=True,
)


class FlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        # TODO. handle dynamic dispatch here
        return flash_attention_ext.forward(Q, K, V)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError
