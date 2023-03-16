from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fast_ops",
    ext_modules=[
        CUDAExtension(
            "flash_attention",
            sources=["fast_ops/flash_attention/flash_attention.cpp"],
            include_dirs=["third-party/cutlass/include"],
            cmdclass={"build_ext": BuildExtension},
        ),
    ],
)
