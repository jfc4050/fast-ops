**fast-ops** is a (personal project) library containing efficient PyTorch operators.

# Operators
* **[(Flash) Multi-Head Attention](fast_ops/flash_attention/README.md):**
Algorithm from [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf).
Significantly faster than vanilla attention due to fused implementation, and also
has $O(n)$ rather than $O(n^2)$ memory complexity.
* **[(Fused) Lion Optimizer](fast_ops/lion/README.md):**
Optimizer described in [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/pdf/2302.06675.pdf).
Claims some improved convergence properties and optimizer states only consists of half precision momentum,
meaning pretty large memory savings over commonly used optimizers like
[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

# Development

## Dependencies
This project's Python dependencies are managed with [Poetry](https://python-poetry.org/docs/).

You can install dependencies (or subsets for development and testing) using:
```bash
> poetry install
```

## Language Server Support
We use [Bear](https://github.com/rizsotto/Bear) to generate the
[compile_commands.json](compile_commands.json) file that is used for language servers.
If you need to update this file you can run:
```bash
> bear python setup.py develop
```
or
```python
> bear pytest
```

## Formatting
Run `scripts/fmt` to reformat all project files using
[clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
[black](https://black.readthedocs.io/en/stable/).

# Resources
* [\[NVIDIA Doc\] CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
* [\[NVIDIA Doc\] Parallel Thread Execution ISA Version 8.1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
* [\[NVIDIA Doc\] CuTe Tutorials](https://github.com/NVIDIA/cutlass/tree/master/media/docs/cute)
* [\[PyTorch Tutorial\] Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
