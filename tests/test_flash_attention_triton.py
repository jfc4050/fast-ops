import math
import re

import torch
import numpy as np
from scipy.stats import distributions

import pytest

from einops import rearrange

from fast_ops.flash_attention.flash_attention_triton import flash_attn_triton, triton_dropout_mask

worker_id_pattern = re.compile(r"gw(\d+)")


@pytest.fixture
def gpu_id_for_test(worker_id):
    if worker_id == "master":
        gpu_id = 0
    else:
        gpu_id = int(re.match(worker_id_pattern, worker_id).group(1))

    # assign to GPUs round-robin. be aware that sometimes too many
    # workers per GPU can result in OOMs
    return gpu_id % torch.cuda.device_count()


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    bias=None,
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        bias: (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]), ids=str)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("batch_size,nheads", [(1, 1), (1, 4), (1, 12), (32, 1), (32, 4)])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 256),
        (256, 256),
        (256, 512),
        (1024, 1024),
        (1024, 512),
        (512, 256),
    ],
)
@pytest.mark.parametrize("bias_shape", ([None, "1h1k", "1hqk", "b11k", "b1qk"]))
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("multi_query", ([False, True]))
def test_flash_attn_triton_output(
    gpu_id_for_test,
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    causal,
    dtype,
    bias_shape,
    dropout_p,
    multi_query,
):
    if seqlen_q >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = f"cuda:{gpu_id_for_test}"
    seed = 0

    device_id_before_test = torch.cuda.current_device()

    torch.cuda.set_device(gpu_id_for_test)  # otherwise triton complains

    # set seed
    torch.random.manual_seed(seed)
    testing_dropout = dropout_p > 0

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k, v = torch.randn(
        batch_size, seqlen_k, 2, nheads if not multi_query else 1, d, device=device, dtype=dtype
    ).unbind(dim=2)
    if multi_query:
        k = k.expand(batch_size, seqlen_k, nheads, d)
        v = v.expand(batch_size, seqlen_k, nheads, d)
    if bias_shape == "1h1k":
        bias = torch.randn(1, nheads, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == "1hqk":
        bias = torch.randn(1, nheads, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == "b11k":
        bias = torch.randn(batch_size, 1, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == "b1qk":
        bias = torch.randn(batch_size, 1, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    else:
        bias = None

    q, k, v = [x.detach().requires_grad_() for x in [q, k, v]]

    if testing_dropout:
        torch.random.manual_seed(seed)
    output = flash_attn_triton(q, k, v, bias, causal, dropout_p)

    keep_mask = None
    if testing_dropout:
        torch.random.manual_seed(seed)
        keep_mask = triton_dropout_mask(
            dropout_p, batch_size, nheads, seqlen_q, seqlen_k, device=device
        )

    output_ref, attn_ref = attention_ref(
        q, k, v, bias=bias, causal=causal, dropout_p=dropout_p, dropout_mask=keep_mask
    )
    output_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        bias=bias,
        causal=causal,
        dropout_p=dropout_p,
        dropout_mask=keep_mask,
        upcast=False,
        reorder_ops=True,
    )

    g = torch.randn_like(output)
    dq, dk, dv = torch.autograd.grad(output, (q, k, v), g)
    (
        dq_ref,
        dk_ref,
        dv_ref,
    ) = torch.autograd.grad(output_ref, (q, k, v), g)
    (
        dq_pt,
        dk_pt,
        dv_pt,
    ) = torch.autograd.grad(output_pt, (q, k, v), g)

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    mismatched_outputs = []
    if (output - output_ref).abs().max().item() > 2 * (output_pt - output_ref).abs().max().item():
        mismatched_outputs.append("out")
    # TODO. dropout increases numerical error and dQ fails sometimes.
    err_factor = 3 if testing_dropout else 2
    if (dq - dq_ref).abs().max().item() > err_factor * (dq_pt - dq_ref).abs().max().item():
        mismatched_outputs.append("dQ")
    if (dk - dk_ref).abs().max().item() > err_factor * (dk_pt - dk_ref).abs().max().item():
        mismatched_outputs.append("dK")
    if (dv - dv_ref).abs().max().item() > err_factor * (dv_pt - dv_ref).abs().max().item():
        mismatched_outputs.append("dV")

    failed = len(mismatched_outputs) > 0
    if failed:

        def report_diff(output_: torch.Tensor, ref_output_: torch.Tensor) -> None:
            output_ = output_.flatten()
            ref_output_ = ref_output_.flatten()

            diff = (output_ - ref_output_).abs()
            diff_argmax = diff.argmax()

            print(
                f"  max diff : {diff.max().item()} ({(diff[diff_argmax] / ref_output_[diff_argmax]).abs() * 100:.2f}%)"
            )
            print(f"  mean diff: {diff.mean().item()}")

        for name, flash_result, pt_result, ref_result in [
            ("O", output, output_pt, output_ref),
            ("dQ", dq, dq_pt, dq_ref),
            ("dK", dk, dk_pt, dk_ref),
            ("dV", dv, dv_pt, dv_ref),
        ]:
            print("\033[1m" + name + "\033[0m")
            print("FlashAttention:")
            report_diff(flash_result, ref_result)
            print("PyTorch:")
            report_diff(pt_result, ref_result)
            print()

    torch.cuda.set_device(device_id_before_test)
    assert not failed, mismatched_outputs


def _vec_binom_test(x: np.ndarray, n: int, p: float) -> np.ndarray:
    """
    stolen from xFormers
    https://github.com/facebookresearch/xformers/blob/main/tests/test_mem_eff_attention.py#L679-L703

    vectorized implementation of scipy.stats.binom_test
    this makes our tests much faster
    reference: https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_morestats.py#L2609-L2702
    """
    x = np.atleast_1d(x)
    d = distributions.binom.pmf(x, n, p)[:, None]
    rerr = 1 + 1e-7
    # x < p * n case
    i = np.arange(np.ceil(p * n), n + 1)
    y = np.sum(distributions.binom.pmf(i, n, p) <= d * rerr, axis=1)
    pval1 = distributions.binom.cdf(x, n, p) + distributions.binom.sf(n - y, n, p)

    # other case
    i = np.arange(np.floor(p * n) + 1)
    y = np.sum(distributions.binom.pmf(i, n, p) <= d * rerr, axis=1)
    pval2 = distributions.binom.cdf(y - 1, n, p) + distributions.binom.sf(x - 1, n, p)

    pval = np.where(x < p * n, pval1, pval2)
    pval = np.minimum(1.0, pval)
    return pval


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 256), (256, 512), (512, 512)])
@pytest.mark.parametrize("dropout_p,seed", [(0.17, 123), (0.17, 0)])
def test_flash_attn_triton_dropout_statistics(gpu_id_for_test, seqlen_q, seqlen_k, dropout_p, seed):
    batch_size = 8
    nheads = 4
    dtype = torch.float16
    d = 128

    device_id_before_test = torch.cuda.current_device()
    device = f"cuda:{gpu_id_for_test}"
    torch.cuda.set_device(gpu_id_for_test)  # otherwise triton complains
    torch.random.manual_seed(seed)

    n_trials = 1000

    # setup inputs
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k, v = torch.randn(batch_size, seqlen_k, 2, nheads, d, device=device, dtype=dtype).unbind(dim=2)
    q, k, v = [x.detach().requires_grad_() for x in [q, k, v]]
    bias = None
    causal = False

    go = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    try:
        torch.random.manual_seed(seed)  # reset RNG offset to 0
        keep_masks = []
        for _ in range(n_trials):
            # will increment RNG offset
            keep_mask = triton_dropout_mask(
                dropout_p, batch_size, nheads, seqlen_q, seqlen_k, device=device
            )
            keep_masks.append(keep_mask.cpu())

        # binomial test on rand uniform tensors
        # add up all the "keeps" for each element, then make suure the percentage of keeps
        # for each element is roughly 1 - dropout_prob
        mask_sums = sum(mask.numpy() for mask in keep_masks).flatten()
        keep_prob = 1 - dropout_p
        pvals = _vec_binom_test(mask_sums, n_trials, keep_prob)
        argmin = pvals.argmin()
        assert (
            pvals[argmin] > 1e-8
        ), f"failed binom test with keep_percentage: {mask_sums[argmin] / n_trials}, keep_prob: {keep_prob} and p_val: {pvals[argmin]}"

        # correctness tests using rand uniform tensors
        torch.random.manual_seed(seed)  # reset RNG offset to 0
        for keep_mask_idx, keep_mask in enumerate(keep_masks):
            keep_mask = keep_mask.to(device)
            # will increment RNG offset
            output = flash_attn_triton(q, k, v, bias, causal, dropout_p)
            output_ref, _ = attention_ref(
                q, k, v, bias=bias, causal=causal, dropout_p=dropout_p, dropout_mask=keep_mask
            )

            dq, dk, dv = torch.autograd.grad(output, (q, k, v), go)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(output_ref, (q, k, v), go)

            to_test = {
                "out": (output, output_ref),
                "dQ": (dq, dq_ref),
                "dK": (dk, dk_ref),
                "dV": (dv, dv_ref),
            }

            mismatched = []
            for name, (val, ref) in to_test.items():
                if not torch.allclose(val, ref, atol=4e-3, rtol=4e-4):
                    mismatched.append(name)

            assert len(mismatched) == 0, f"{mismatched} mismatched on {keep_mask_idx}"

    finally:
        torch.cuda.set_device(device_id_before_test)
