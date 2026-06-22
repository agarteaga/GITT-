"""
setup.py for GITT_V2
Installs all Python dependencies required to run GITT_V2.py.

──────────────────────────────────────────────────────────────
QUICK-START (copy these commands into your terminal)
──────────────────────────────────────────────────────────────

1. Install tkinter (only needed once, not on PyPI):
     Ubuntu/Debian : sudo apt-get install python3-tk
     macOS Homebrew: brew install python-tk
     Windows       : already included with python.org installers

2. Install the package + its Python dependencies:
     pip install -e .

3. Run the GUI:
     python GITT_V2.py
   or, after the install above, just:
     gitt

──────────────────────────────────────────────────────────────
"""
import sys
import os
from setuptools import setup

# ── Python version guard ──────────────────────────────────────────────────────
if sys.version_info < (3, 9):
    sys.exit(
        f"GITT_V2 requires Python 3.9 or later.\n"
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# ── Optional long description (won't crash if README.md is absent) ────────────
_readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.isfile(_readme_path):
    with open(_readme_path, encoding="utf-8") as _f:
        _long_desc = _f.read()
    _long_desc_type = "text/markdown"
else:
    _long_desc = (
        "GITT analysis GUI — Weppner & Huggins and Kang & Chueh "
        "solid-state diffusion coefficient extraction."
    )
    _long_desc_type = "text/plain"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup(
    name="GITT_V2",
    version="2.1.0",
    description=(
        "GITT analysis GUI — Weppner & Huggins and Kang & Chueh "
        "solid-state diffusion coefficient extraction."
    ),
    long_description=_long_desc,
    long_description_content_type=_long_desc_type,
    author="GITT_V2 contributors",
    python_requires=">=3.9",

    # ── Runtime dependencies (all available on PyPI) ──────────────────────────
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "matplotlib>=3.6",
    ],

    # ── Optional dev tools ────────────────────────────────────────────────────
    extras_require={
        "dev": [
            "pytest>=7",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },

    # ── CLI entry point ───────────────────────────────────────────────────────
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
