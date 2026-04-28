"""Shared test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_project_path():
    """Return a sample project path — adjust to your local dataset."""
    p = Path.home() / "DCIM" / "A_579901304753"
    if not p.exists():
        pytest.skip(f"Sample project not found: {p}")
    return p
