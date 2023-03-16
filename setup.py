from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fast_ops",
    ext_modules=[
        CppExtension(
            "flash_attention",
            sources=["fast_ops/flash_attention/flash_attention.cpp"],
            cmdclass={"build_ext": BuildExtension},
        ),
    ],
)
