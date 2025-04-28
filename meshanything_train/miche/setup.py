from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


setup(
    name="michelangelo",
    version="0.4.1",
    author="Zibo Zhao, Wen Liu and Xin Chen",
    author_email="liuwen@shanghaitech.edu.cn",
    description="Michelangelo: a 3D Shape Generation System.",
    packages=find_packages(exclude=("configs", "tests", "scripts", "example_data")),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "cython",
        "tqdm",
    ],
)
