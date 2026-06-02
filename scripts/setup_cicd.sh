#!/bin/bash
# Setup pre-commit hooks pour Rugby IA

set -e

echo "=== Rugby IA — CI/CD Setup ==="
echo ""

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "[!] pre-commit not found. Installing..."
    pip install pre-commit
fi

echo "[*] Installing pre-commit hooks..."
pre-commit install

echo "[*] Running initial checks..."
pre-commit run --all-files || true

echo ""
echo "[✓] CI/CD setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests locally: make test"
echo "  2. Check code quality: make lint"
echo "  3. Format code: make format"
echo ""
echo "Pre-commit hooks are now active and will run on each commit."
echo "To bypass hooks temporarily: git commit --no-verify"
