from typing import Tuple

from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load

flash_attention_ext = load(
    name="flash_attention",
    sources=["fast_ops/flash_attention/flash_attention.cpp"],
    extra_include_paths=["third-party/cutlass/include"],
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
