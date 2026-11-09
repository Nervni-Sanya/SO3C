"""Minimal setup.py — install with:  pip install -e ."""

from setuptools import setup, find_packages

setup(
    name="so3c",
    # Keep in step with so3c.__version__ and the version in CITATION.cff.
    version="1.0.0",
    description=(
        "SO3C: Lorentz-covariant jet tagging with complexified-SO(3) "
        "geodesic flows"
    ),
    author="Panchenko Alexander",
    author_email="sascha.panchenko2018@yandex.ru",
    license="MIT",
    # Installs so3c and the benchmark harness. NOTE: `benchmarks` is a
    # top-level name that the so33 repository also ships, so never install
    # both projects into the same environment.
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchdiffeq>=0.2.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "jupyter"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
