# TileLang Developer Tests

This directory contains unit and integration tests for TileLang kernel implementations based on the examples in `tilelang-developer/references/EXAMPLES.md`.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared pytest fixtures
├── README.md                      # This file
└── tilelang-developer/
    ├── __init__.py
    ├── test_gemm.py              # Matrix Multiplication tests
    ├── test_flash_attention.py   # FlashAttention V2 tests
    └── test_mla_decode.py        # DeepSeek MLA Decoding tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests for a specific module
```bash
pytest tests/tilelang-developer/test_gemm.py
pytest tests/tilelang-developer/test_flash_attention.py
pytest tests/tilelang-developer/test_mla_decode.py
```

### Run tests by marker
```bash
# Run only compilation tests (no CUDA required)
pytest -m compilation

# Run only integration tests (requires CUDA)
pytest -m integration

# Run only CUDA tests
pytest -m cuda

# Skip slow tests
pytest -m "not slow"
```

### Run specific test functions
```bash
pytest tests/tilelang-developer/test_gemm.py::TestMatmulKernel::test_kernel_compilation
pytest tests/tilelang-developer/test_flash_attention.py::TestFlashAttentionV2Kernel::test_small_sequence_length
```

### Verbose output
```bash
pytest -v
pytest -vv  # Extra verbose
```

## Test Categories

### Compilation Tests
Tests that verify kernels compile successfully with various configurations. These tests don't require CUDA hardware.

Examples:
- `test_kernel_compilation()` - Basic compilation
- `test_different_block_sizes()` - Various block configurations
- `test_pipelining_stages()` - Pipeline configuration

### Integration Tests
Tests that run actual kernel computations and verify correctness against reference implementations. These tests require CUDA hardware.

Examples:
- `test_correctness_against_pytorch()` - Compare with PyTorch
- `test_correctness_against_standard_attention()` - Compare with reference implementation

## Dependencies

Required:
- `pytest` - Test framework
- `tilelang` - TileLang compiler

Optional (for integration tests):
- `torch` - PyTorch with CUDA support
- CUDA-capable GPU

Install dependencies:
```bash
# Using uv
uv pip install pytest torch tilelang

# Or using pip
pip install pytest torch tilelang
```

## Test Structure

Each test file follows this structure:

1. **Compilation tests** - Test kernel compilation with various configurations
2. **Integration tests** - Test actual computation and correctness

Each test class:
- `Test<KernelName>Kernel` - Compilation and configuration tests
- `Test<KernelName>Integration` - Integration tests requiring CUDA

## Example Test Files

### test_gemm.py
Tests for Matrix Multiplication (GEMM) kernel:
- Kernel compilation
- Various matrix dimensions (square, non-square, small, large)
- Different block sizes (64x64, 128x128)
- Pipeline stages (2, 3, 4)
- Correctness against PyTorch

### test_flash_attention.py
Tests for FlashAttention V2 kernel:
- Kernel compilation
- Various sequence lengths (512, 2048, 4096)
- Different head dimensions (64, 128)
- Block size configurations
- Correctness against standard attention

### test_mla_decode.py
Tests for DeepSeek MLA Decoding kernel:
- Kernel compilation
- Various KV cache sizes
- Different latent dimensions (256, 512, 1024)
- Block size configurations
- Batched inference
- KV decompression validation

## Adding New Tests

To add tests for a new kernel example:

1. Create a new test file: `tests/tilelang-developer/test_<kernel_name>.py`
2. Import required modules:
   ```python
   import pytest
   import tilelang
   import tilelang.language as T
   ```
3. Create test classes:
   - `Test<KernelName>Kernel` for compilation tests
   - `Test<KernelName>Integration` for integration tests
4. Add appropriate markers (`@pytest.mark.cuda`, `@pytest.mark.integration`, etc.)
5. Update this README with the new test file

## Continuous Integration

Tests can be run in CI pipelines with:

```bash
# Run only compilation tests (no GPU required)
pytest -m "not cuda"

# Run all tests if GPU available
pytest
```

## Troubleshooting

### ImportError: No module named 'tilelang'
Install TileLang: `uv pip install tilelang`

### CUDA tests are skipped
Integration tests require CUDA hardware. Use `-m "not cuda"` to skip them.

### Tests are slow
Use `-m "not slow"` to skip slow-running tests, or run specific test files.
