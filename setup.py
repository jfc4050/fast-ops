from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fast_ops",
    ext_modules=[
        CUDAExtension(
            "flash_attention",
            sources=[
                "fast_ops/flash_attention/flash_attention.cpp",
                "fast_ops/flash_attention/flash_attention_fwd.cu",
            ],
            include_dirs=["third-party/cutlass/include"],
            extra_compile_args={"nvcc": ["--threads", "0"]},
            cmdclass={"build_ext": BuildExtension},
        ),
    ],
)
