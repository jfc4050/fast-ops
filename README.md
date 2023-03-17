**fast-ops** is a (personal project) library containing efficient PyTorch operators.

# Operators
* [(Flash) Multi-Head Attention](fast_ops/flash_attention/README.md)

# Development

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
