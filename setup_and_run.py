#!/usr/bin/env python3
"""
GITT App — Setup & Launcher
────────────────────────────
Run this script ONCE before using the GITT app.
It will:
  1. Check your Python version
  2. Install all required packages
  3. Check that tkinter is available
  4. Launch the GITT app

Usage:
    python setup_and_run.py
    python setup_and_run.py --data_dir "C:/path/to/your/data"
Author: Ane Gondra 
"""

import sys
import os
import argparse
import subprocess

# ── Colour helpers for terminal output ───────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

def ok(msg):   print(f"  {green('✔')}  {msg}")
def fail(msg): print(f"  {red('✗')}  {msg}")
def warn(msg): print(f"  {yellow('!')}  {msg}")
def info(msg): print(f"  {bold('→')}  {msg}")

# ── Required packages (name to install, name to import) ──────────────────────
REQUIRED = [
    ("numpy",      "numpy"),
    ("scipy",      "scipy"),
    ("matplotlib", "matplotlib"),
]

# ─────────────────────────────────────────────────────────────────────────────
def check_python():
    print(bold("\n[1/4] Checking Python version..."))
    major, minor = sys.version_info[:2]
    version_str = f"Python {major}.{minor}.{sys.version_info[2]}"
    if major < 3 or (major == 3 and minor < 8):
        fail(f"{version_str} — Python 3.8 or newer is required.")
        print(red("\n  Please download a newer Python from https://www.python.org\n"))
        sys.exit(1)
    ok(f"{version_str} — OK")


def install_packages():
    print(bold("\n[2/4] Installing required packages..."))
    all_ok = True
    for pkg_install, pkg_import in REQUIRED:
        try:
            __import__(pkg_import)
            ok(f"{pkg_install} — already installed")
        except ImportError:
            info(f"Installing {pkg_install}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg_install],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                ok(f"{pkg_install} — installed successfully")
            else:
                fail(f"{pkg_install} — installation failed")
                print(red(f"    {result.stderr.strip()}"))
                all_ok = False
    if not all_ok:
        print(red("\n  Some packages failed to install."))
        print(red("  Try running manually:  pip install numpy scipy matplotlib\n"))
        sys.exit(1)


def check_tkinter():
    print(bold("\n[3/4] Checking tkinter (GUI framework)..."))
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        ok("tkinter — available and working")
    except ImportError:
        fail("tkinter is not installed.")
        print(yellow("\n  On Linux, install it with:"))
        print(yellow("    sudo apt-get install python3-tk"))
        print(yellow("\n  On Windows/Mac, reinstall Python from https://www.python.org"))
        print(yellow("  and make sure to check 'tcl/tk and IDLE' during installation.\n"))
        sys.exit(1)
    except Exception as e:
        fail(f"tkinter found but failed to open a window: {e}")
        print(yellow("  This usually means no display is available (e.g. SSH without -X).\n"))
        sys.exit(1)


def launch_gitt(data_dir=None):
    print(bold("\n[4/4] Launching GITT app...\n"))

    # Find GITT_V1.py in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gitt_path  = os.path.join(script_dir, "GITT_V1.py")

    if not os.path.isfile(gitt_path):
        fail(f"GITT_V1.py not found in: {script_dir}")
        print(red("  Make sure GITT.py and setup_and_run.py are in the same folder.\n"))
        sys.exit(1)

    ok(f"Found GITT_V1.py at: {gitt_path}")

    cmd = [sys.executable, gitt_path]
    if data_dir:
        if not os.path.isdir(data_dir):
            warn(f"Data folder not found: {data_dir}")
            warn("Starting without pre-loading data. Use the Browse button in the app.")
        else:
            cmd += ["--data_dir", data_dir]
            ok(f"Data folder: {data_dir}")

    print(f"\n  {bold('Starting...')}\n")
    print("─" * 50)

    # Replace current process with GITT so it runs in the foreground
    os.execv(sys.executable, cmd)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─" * 50)
    print(bold("  GITT App — Setup & Launcher"))
    print("─" * 50)

    parser = argparse.ArgumentParser(description="Setup and launch the GITT app.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help='Path to your data folder, e.g. --data_dir "C:/Users/you/data"'
    )
    args = parser.parse_args()

    check_python()
    install_packages()
    check_tkinter()
    launch_gitt(data_dir=args.data_dir)
