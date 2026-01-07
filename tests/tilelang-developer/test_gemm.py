"""
Unit tests for TileLang Matrix Multiplication (GEMM) kernel.

Tests the matmul_kernel example from EXAMPLES.md to ensure:
- Kernel compiles successfully
- Handles various matrix dimensions
- Supports different block sizes
- Validates output correctness
"""

import pytest


class TestMatmulKernel:
    """Test suite for the TileLang GEMM kernel."""

    def test_kernel_compilation(self):
        """Test that the matmul kernel compiles without errors."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[2])
        def matmul_kernel(M, N, K, block_M, block_N, block_K):
            @T.prim_func
            def main(
                A: T.Buffer((M, K), "float16"),
                B: T.Buffer((K, N), "float16"),
                C: T.Buffer((M, N), "float16")
            ):
                with T.Kernel(
                    T.ceildiv(N, block_N),
                    T.ceildiv(M, block_M),
                    threads=128
                ) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), "float16")
                    B_shared = T.alloc_shared((block_K, block_N), "float16")
                    C_local = T.alloc_fragment((block_M, block_N), "float32")

                    T.annotate_layout({
                        A_shared: T.make_swizzled_layout(A_shared),
                        B_shared: T.make_swizzled_layout(B_shared)
                    })

                    T.clear(C_local)

                    num_k_tiles = T.ceildiv(K, block_K)
                    for k in T.Pipelined(num_k_tiles, num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=False)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return main

        # Test compilation succeeds
        func = matmul_kernel(
            M=1024, N=1024, K=1024,
            block_M=128, block_N=128, block_K=32
        )
        assert func is not None

    def test_small_matrices(self):
        """Test GEMM with small matrix dimensions."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[2])
        def matmul_kernel(M, N, K, block_M, block_N, block_K):
            @T.prim_func
            def main(
                A: T.Buffer((M, K), "float16"),
                B: T.Buffer((K, N), "float16"),
                C: T.Buffer((M, N), "float16")
            ):
                with T.Kernel(
                    T.ceildiv(N, block_N),
                    T.ceildiv(M, block_M),
                    threads=128
                ) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), "float16")
                    B_shared = T.alloc_shared((block_K, block_N), "float16")
                    C_local = T.alloc_fragment((block_M, block_N), "float32")

                    T.annotate_layout({
                        A_shared: T.make_swizzled_layout(A_shared),
                        B_shared: T.make_swizzled_layout(B_shared)
                    })

                    T.clear(C_local)

                    num_k_tiles = T.ceildiv(K, block_K)
                    for k in T.Pipelined(num_k_tiles, num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=False)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return main

        # Test with small dimensions
        func = matmul_kernel(
            M=256, N=256, K=256,
            block_M=64, block_N=64, block_K=32
        )
        assert func is not None

    def test_non_square_matrices(self):
        """Test GEMM with non-square matrix dimensions."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[2])
        def matmul_kernel(M, N, K, block_M, block_N, block_K):
            @T.prim_func
            def main(
                A: T.Buffer((M, K), "float16"),
                B: T.Buffer((K, N), "float16"),
                C: T.Buffer((M, N), "float16")
            ):
                with T.Kernel(
                    T.ceildiv(N, block_N),
                    T.ceildiv(M, block_M),
                    threads=128
                ) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), "float16")
                    B_shared = T.alloc_shared((block_K, block_N), "float16")
                    C_local = T.alloc_fragment((block_M, block_N), "float32")

                    T.annotate_layout({
                        A_shared: T.make_swizzled_layout(A_shared),
                        B_shared: T.make_swizzled_layout(B_shared)
                    })

                    T.clear(C_local)

                    num_k_tiles = T.ceildiv(K, block_K)
                    for k in T.Pipelined(num_k_tiles, num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=False)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return main

        # Test with rectangular matrices
        func = matmul_kernel(
            M=512, N=1024, K=768,
            block_M=128, block_N=128, block_K=32
        )
        assert func is not None

    def test_different_block_sizes(self):
        """Test GEMM with various block size configurations."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[2])
        def matmul_kernel(M, N, K, block_M, block_N, block_K):
            @T.prim_func
            def main(
                A: T.Buffer((M, K), "float16"),
                B: T.Buffer((K, N), "float16"),
                C: T.Buffer((M, N), "float16")
            ):
                with T.Kernel(
                    T.ceildiv(N, block_N),
                    T.ceildiv(M, block_M),
                    threads=128
                ) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), "float16")
                    B_shared = T.alloc_shared((block_K, block_N), "float16")
                    C_local = T.alloc_fragment((block_M, block_N), "float32")

                    T.annotate_layout({
                        A_shared: T.make_swizzled_layout(A_shared),
                        B_shared: T.make_swizzled_layout(B_shared)
                    })

                    T.clear(C_local)

                    num_k_tiles = T.ceildiv(K, block_K)
                    for k in T.Pipelined(num_k_tiles, num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=False)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return main

        # Test block size 128x128x32
        func1 = matmul_kernel(
            M=1024, N=1024, K=1024,
            block_M=128, block_N=128, block_K=32
        )
        assert func1 is not None

        # Test block size 64x64x32
        func2 = matmul_kernel(
            M=1024, N=1024, K=1024,
            block_M=64, block_N=64, block_K=32
        )
        assert func2 is not None

    def test_pipelining_stages(self):
        """Test GEMM with different pipeline stages."""
        import tilelang
        import tilelang.language as T

        for num_stages in [2, 3, 4]:
            @tilelang.jit(target="cuda", out_idx=[2])
            def matmul_kernel(M, N, K, block_M, block_N, block_K):
                @T.prim_func
                def main(
                    A: T.Buffer((M, K), "float16"),
                    B: T.Buffer((K, N), "float16"),
                    C: T.Buffer((M, N), "float16")
                ):
                    with T.Kernel(
                        T.ceildiv(N, block_N),
                        T.ceildiv(M, block_M),
                        threads=128
                    ) as (bx, by):
                        A_shared = T.alloc_shared((block_M, block_K), "float16")
                        B_shared = T.alloc_shared((block_K, block_N), "float16")
                        C_local = T.alloc_fragment((block_M, block_N), "float32")

                        T.annotate_layout({
                            A_shared: T.make_swizzled_layout(A_shared),
                            B_shared: T.make_swizzled_layout(B_shared)
                        })

                        T.clear(C_local)

                        num_k_tiles = T.ceildiv(K, block_K)
                        for k in T.Pipelined(num_k_tiles, num_stages=num_stages):
                            T.copy(A[by * block_M, k * block_K], A_shared)
                            T.copy(B[k * block_K, bx * block_N], B_shared)
                            T.gemm(A_shared, B_shared, C_local, transpose_B=False)

                        T.copy(C_local, C[by * block_M, bx * block_N])

                return main

            func = matmul_kernel(
                M=512, N=512, K=512,
                block_M=64, block_N=64, block_K=32
            )
            assert func is not None, f"Failed with {num_stages} pipeline stages"


class TestGemmIntegration:
    """Integration tests for GEMM kernel with actual computation."""

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_correctness_against_pytorch(self):
        """Test GEMM correctness against PyTorch reference."""
        import tilelang
        import tilelang.language as T
        import torch

        @tilelang.jit(target="cuda", out_idx=[2])
        def matmul_kernel(M, N, K, block_M, block_N, block_K):
            @T.prim_func
            def main(
                A: T.Buffer((M, K), "float16"),
                B: T.Buffer((K, N), "float16"),
                C: T.Buffer((M, N), "float16")
            ):
                with T.Kernel(
                    T.ceildiv(N, block_N),
                    T.ceildiv(M, block_M),
                    threads=128
                ) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), "float16")
                    B_shared = T.alloc_shared((block_K, block_N), "float16")
                    C_local = T.alloc_fragment((block_M, block_N), "float32")

                    T.annotate_layout({
                        A_shared: T.make_swizzled_layout(A_shared),
                        B_shared: T.make_swizzled_layout(B_shared)
                    })

                    T.clear(C_local)

                    num_k_tiles = T.ceildiv(K, block_K)
                    for k in T.Pipelined(num_k_tiles, num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=False)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return main

        # Create test inputs
        M, N, K = 512, 512, 512
        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda')
        C = torch.empty(M, N, dtype=torch.float16, device='cuda')

        # Compute with TileLang
        func = matmul_kernel(M=M, N=N, K=K, block_M=128, block_N=128, block_K=32)
        func(A, B, C)

        # Compute reference
        C_ref = torch.matmul(A, B)

        # Check correctness (with tolerance for float16)
        assert torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2)


def _cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
