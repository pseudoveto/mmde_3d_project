from setuptools import setup, find_packages

setup(
    name="Agentic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "math"
    ],
    author="A.m.selvaganesh, S. Sankaranarayanan, A. Varsha,",
    description="A GPU-accelerated 3D Framework for agentic control of UGVs",
    changes={
        "Version 0.1.0" : "initial release"}
)
