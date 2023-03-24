import itertools

import torch
from torch.utils.benchmark import Compare, Timer

from fast_ops.lion.lion_optimizer import Lion
from fast_ops.lion.lion_optimizer_ref import Lion as LionRef


sizes = [8192, 8192**2]
dtypes = [torch.float16, torch.bfloat16]

lr = 1e-4
betas = (0.9, 0.99)
weight_decay = 0.1
label = "lion step"

results = []
for size, dtype in itertools.product(sizes, dtypes):
    sub_label = f"{size:.1e}, {dtype}"

    for LionCls, cls_desc in [(Lion, "ours"), (LionRef, "ref")]:
        master_params = [torch.randn(size, requires_grad=True, dtype=dtype, device="cuda")]
        params = [p.clone().detach() for p in master_params]
        for param in params:
            param.grad = torch.randn_like(param, requires_grad=False)
        lion = LionCls(params, lr=lr, betas=betas, weight_decay=weight_decay)

        results.append(
            Timer(
                stmt="lion.step()",
                globals={"lion": lion},
                label=label,
                sub_label=sub_label,
                description=cls_desc,
            ).blocked_autorange()
        )

compare = Compare(results)
compare.print()
