import pytest
import torch

from fast_ops.lion.lion_optimizer import Lion
from fast_ops.lion.lion_optimizer_ref import Lion as LionRef

torch.set_printoptions(sci_mode=False)

@pytest.mark.parametrize("weight_decay", [0.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_lion(weight_decay: float, dtype: torch.dtype) -> None:
    lr = 1e-2
    betas = (0.9, 0.99)

    master_params = [
        torch.rand(32, requires_grad=True, dtype=dtype, device="cuda") for _ in range(3)
    ]
    params = [p.clone().detach() for p in master_params]
    params_ref = [p.clone().detach() for p in master_params]

    lion = Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
    lion_ref = LionRef(params_ref, lr=lr, betas=betas, weight_decay=weight_decay)

    for iter in range(5):  # simulate a few training steps
        for param, param_ref in zip(params, params_ref):
            grad = torch.randn_like(param, requires_grad=False)
            param.grad = grad
            param_ref.grad = grad

        lion.step()
        lion_ref.step()

        for param_idx, (param, param_ref) in enumerate(zip(params, params_ref)):
            # print(param)
            # print(param_ref)
            assert torch.allclose(
                param, param_ref, rtol=1e-3, atol=1e-3
            ), f"param mismatch on param {param_idx}, iter {iter}"
            exp_avg = lion.state[param]["exp_avg"]
            exp_avg_ref = lion_ref.state[param_ref]["exp_avg"]
            # print(exp_avg)
            # print(exp_avg_ref)
            assert torch.allclose(
                exp_avg, exp_avg_ref, rtol=1e-3, atol=1e-3
            ), f"exp_avg mismatch on param {param_idx}, iter {iter}"
