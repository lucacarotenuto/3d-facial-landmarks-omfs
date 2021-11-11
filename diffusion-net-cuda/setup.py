import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='diffusion_net_cuda',
    ext_modules=[
        CUDAExtension(
            name='diffusion_net_cuda',
            sources=[
                'src/diffusion_net_cuda/module.cpp',
                'src/diffusion_net_cuda/geometry.cpp',
                'src/diffusion_net_cuda/geometry_cuda.cu',
                'src/diffusion_net_cuda/geometry_cuda_kernels.cu',
            ],
        ),        
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)