from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer


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

                # stepweight decay
                param.data.mul_(1 - lr * weight_decay)

                # weight update
                update = exp_avg.clone()
                update.mul_(beta1)
                update.add_(grad, alpha=1 - beta1)
                update.sign_()
                param.add_(update, alpha=-lr)

                # decay momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
