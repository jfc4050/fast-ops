import pytest
import torch
from torch import Tensor

from fast_ops.lion.lion_optimizer import Lion
from fast_ops.lion.lion_optimizer_ref import Lion as LionRef

torch.set_printoptions(sci_mode=False, linewidth=200)

@pytest.mark.parametrize("weight_decay", [0.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_lion(weight_decay: float, dtype: torch.dtype) -> None:

    def __print_tensor_diff(x: Tensor, x_ref: Tensor):
        print(x.view(-1, 8))
        print(x_ref.view(-1, 8))
        print((x - x_ref).abs().view(-1, 8))

    lr = 1e-2
    betas = (0.9, 0.99)

    master_params = [
        torch.randn(32, requires_grad=True, dtype=dtype, device="cuda") for _ in range(3)
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
            passed = torch.allclose(param, param_ref, rtol=1e-4, atol=1e-2)
            if not passed:
                __print_tensor_diff(param, param_ref)
                assert False, f"param mismatch on param {param_idx}, iter {iter}"
            exp_avg = lion.state[param]["exp_avg"]
            exp_avg_ref = lion_ref.state[param_ref]["exp_avg"]
            passed = torch.allclose(exp_avg, exp_avg_ref, rtol=1e-4, atol=1e-2)
            if not passed:
                __print_tensor_diff(exp_avg, exp_avg_ref)
                assert False, f"exp_avg mismatch on param {param_idx}, iter {iter}"
