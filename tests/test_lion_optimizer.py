import pytest
import torch

from fast_ops.lion.lion_optimizer import Lion
from fast_ops.lion.lion_optimizer_ref import Lion as LionRef


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
def test_lion(weight_decay: float, dtype: torch.dtype) -> None:
    lr = 1e-4
    betas = (0.9, 0.99)

    master_params = [
        torch.rand(100, requires_grad=True, dtype=dtype, device="cuda") for _ in range(3)
    ]
    params = [p.clone().detach() for p in master_params]
    params_ref = [p.clone().detach() for p in master_params]

    lion = Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
    lion_ref = LionRef(params_ref, lr=lr, betas=betas, weight_decay=weight_decay)

    for _ in range(5):  # simulate a few training steps
        for param, param_ref in zip(params, params_ref):
            grad = torch.randn_like(param, requires_grad=False)
            param.grad = grad
            param_ref.grad = grad

        lion.step()
        lion_ref.step()

        for param, param_ref in zip(params, params_ref):
            assert torch.allclose(param, param_ref, rtol=1e-4, atol=1e-4)
            exp_avg = lion.state[param]["exp_avg"]
            exp_avg_ref = lion_ref.state[param_ref]["exp_avg"]
            assert torch.allclose(exp_avg, exp_avg_ref, rtol=1e-4, atol=1e-4)
