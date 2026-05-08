"""
setup.py for GITT_V2
Installs all Python dependencies required to run GITT_V2.py.

Usage:
    pip install -e .          # editable install (development)
    pip install .             # regular install
    pip install -e ".[dev]"   # + optional testing/linting tools

After installation the command 'gitt' is available on the PATH.
tkinter is NOT on PyPI — install it via your OS package manager:
    Ubuntu/Debian : sudo apt-get install python3-tk
    macOS Homebrew: brew install python-tk
    Windows       : included with the python.org installer
"""
from setuptools import setup
import sys

if sys.version_info < (3, 9):
    sys.exit(
        f"GITT_V2 requires Python 3.9 or later.\n"
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}."
    )

setup(
    name="GITT_V2",
    version="2.2.0",
    description=(
        "GITT analysis GUI — Weppner & Huggins and Kang & Chueh "
        "solid-state diffusion coefficient extraction."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="GITT_V2 contributors",
    python_requires=">=3.9",

    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "matplotlib>=3.6",
    ],

    extras_require={
        "dev": [
            "pytest>=7",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },

    entry_points={
        "console_scripts": [
            "gitt=GITT_V2:main",
        ],
    },

    py_modules=["GITT_V2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "GITT", "electrochemistry", "diffusion coefficient",
        "battery", "solid-state", "Weppner", "Huggins", "Kang", "Chueh",
    ],
)
