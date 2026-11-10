from setuptools import setup, find_packages

setup(
    name="Environment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib",
    ],
    author="Your Name",
    description="A GPU-accelerated 3D UGV simulation environment using PyTorch SDFs.",
)
