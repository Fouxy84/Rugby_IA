#!/usr/bin/env python
# setup_cicd.py - Cross-platform CI/CD setup

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=None):
    """Run a command and handle errors."""
    if description:
        print(f"[*] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[!] Error: {e.stderr}")
        return False

def main():
    print("=" * 50)
    print("  Rugby IA — CI/CD Setup")
    print("=" * 50)
    print()
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"[!] Python 3.10+ required (got {sys.version_info.major}.{sys.version_info.minor})")
        return 1
    
    # Install dev dependencies
    print("[*] Installing development dependencies...")
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    run_command(
        f"{sys.executable} -m pip install pre-commit",
        "Installing pre-commit"
    )
    
    # Setup pre-commit hooks
    print("[*] Setting up pre-commit hooks...")
    if not run_command(f"{sys.executable} -m pre_commit install", "Installing pre-commit hooks"):
        print("[!] Warning: pre-commit setup had issues, but continuing...")
    
    # Run initial checks (non-blocking)
    print("[*] Running initial pre-commit checks (non-blocking)...")
    run_command(f"{sys.executable} -m pre_commit run --all-files", "Pre-commit checks")
    
    print()
    print("[✓] CI/CD setup complete!")
    print()
    print("Next steps:")
    print("  1. Run tests: pytest tests/ -v")
    print("  2. Check linting: make lint (or python -m pylint src)")
    print("  3. Format code: make format (or black src/)")
    print()
    print("Pre-commit hooks are now active and will run on each commit.")
    print("To bypass: git commit --no-verify")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
