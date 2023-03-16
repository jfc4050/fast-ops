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

## Formatting
Run `scripts/fmt` to reformat all project files.

# Resources
* [\[PyTorch Tutorial\] Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
