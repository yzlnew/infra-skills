"""
Shared pytest fixtures and configuration for TileLang tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "compilation: mark test as compilation-only test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def tilelang_available():
    """Check if TileLang is available for testing."""
    try:
        import tilelang
        import tilelang.language
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_tilelang(tilelang_available):
    """Skip test if TileLang is not available."""
    if not tilelang_available:
        pytest.skip("TileLang not installed")
