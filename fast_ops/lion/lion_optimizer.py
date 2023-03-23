from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
import torch.utils.cpp_extension

# monkey patching this for now so I can add in my own arch flags.
# for some reason even when building with TORCH_CUDA_ARCH_LIST=8.0
# it builds for an older architecture and thinks it only has 49152B (0xc000)
# of shared memory.
torch.utils.cpp_extension._get_cuda_arch_flags = lambda: []

lion_ext = torch.utils.cpp_extension.load(
    name="lion_optimizer",
    sources=[
        "fast_ops/lion/lion.cpp",
        "fast_ops/lion/lion_update.cu",
    ],
    extra_cflags=["-std=c++17"],
    # see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
    extra_cuda_cflags=["--threads", "0", "-std=c++17", "--gpu-architecture=compute_80"],
    with_cuda=True,
    verbose=True,
)


class Lion(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise RuntimeError(f"expected positive LR, got {lr}")
        if not all(0 <= beta <= 1 for beta in betas):
            raise RuntimeError(f"betas must be between 0 and 1. got {betas}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for param_group in self.param_groups:
            lr = param_group["lr"]
            weight_decay = param_group["weight_decay"]
            beta1, beta2 = param_group["betas"]
            for param in filter(lambda p: p.grad is not None, param_group["params"]):
                param: Parameter
                grad: Tensor = param.grad

                state = self.state[param]
                # init state: exponential moving average of gradient value
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg: Tensor = state["exp_avg"]

                lion_ext.lion_update(param, grad, exp_avg, lr, beta1, beta2, weight_decay)
