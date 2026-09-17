"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    p = project_root / "data"
    if not p.exists():
        pytest.skip("data/ not present (test set is optional)")
    return p
