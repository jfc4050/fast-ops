**fast-ops** is a (personal project) library containing efficient PyTorch operators.

# Operators

## Flash Attention
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)

# Development

## Code Completion
we use [Bear](https://github.com/rizsotto/Bear) to generate the
[compile_commands.json](compile_commands.json) file that is used for code completion.
if you need to update this file you can run:
```bash
> bear python setup.py develop
```

# Resources
* [\[PyTorch Tutorial\] Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
