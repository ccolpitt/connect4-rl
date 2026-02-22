"""
Root conftest.py - Adds project root and src/ to sys.path for all pytest runs.
Place at the project root so pytest discovers it automatically.
"""

import sys
from pathlib import Path

# Ensure project root is on path so `from src.xxx import ...` works
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
