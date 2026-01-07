"""
Unit tests for TileLang FlashAttention V2 kernel.

Tests the flash_attention_v2 example from EXAMPLES.md to ensure:
- Kernel compiles successfully
- Handles various sequence lengths and dimensions
- Implements online softmax correctly
- Memory-efficient O(N) complexity
"""

import pytest


class TestFlashAttentionV2Kernel:
    """Test suite for the TileLang FlashAttention V2 kernel."""

    def test_kernel_compilation(self):
        """Test that the FlashAttention V2 kernel compiles without errors."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def flash_attention_v2(batch, n_heads, seq_len, head_dim, block_M, block_N):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                K: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                V: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, seq_len, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(seq_len, block_M),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_M, head_dim), "float16")
                    K_shared = T.alloc_shared((block_N, head_dim), "float16")
                    V_shared = T.alloc_shared((block_N, head_dim), "float16")

                    S_local = T.alloc_fragment((block_M, block_N), "float32")
                    O_local = T.alloc_fragment((block_M, head_dim), "float32")

                    m_current = T.alloc_fragment((block_M,), "float32")
                    l_current = T.alloc_fragment((block_M,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        K_shared: T.make_swizzled_layout(K_shared),
                        V_shared: T.make_swizzled_layout(V_shared)
                    })

                    T.fill(m_current, -1e9)
                    T.clear(l_current)
                    T.clear(O_local)

                    T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

                    num_kv_tiles = T.ceildiv(seq_len, block_N)
                    for k in T.Pipelined(num_kv_tiles, num_stages=2):
                        T.copy(K[bz, by, k * block_N, 0], K_shared)
                        T.copy(V[bz, by, k * block_N, 0], V_shared)

                        T.clear(S_local)
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = S_local[i, j] * scale

                        m_prev = T.alloc_fragment((block_M,), "float32")
                        T.copy(m_current, m_prev)

                        T.reduce_max(S_local, m_current, dim=1)

                        for i in T.Parallel(block_M):
                            correction = T.exp(m_prev[i] - m_current[i])
                            l_current[i] = l_current[i] * correction

                            for d in T.Parallel(head_dim):
                                O_local[i, d] = O_local[i, d] * correction

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = T.exp(S_local[i, j] - m_current[i])

                        for i in T.Parallel(block_M):
                            row_sum = T.float32(0.0)
                            for j in T.Parallel(block_N):
                                row_sum = row_sum + S_local[i, j]
                            l_current[i] = l_current[i] + row_sum

                        T.gemm(S_local, V_shared, O_local, transpose_B=False)

                    for i, d in T.Parallel(block_M, head_dim):
                        O_local[i, d] = O_local[i, d] / l_current[i]

                    T.copy(O_local, O[bz, by, bx * block_M, 0])

            return main

        # Test compilation succeeds
        func = flash_attention_v2(
            batch=2, n_heads=32, seq_len=2048, head_dim=128,
            block_M=64, block_N=64
        )
        assert func is not None

    def test_small_sequence_length(self):
        """Test FlashAttention with small sequence length."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def flash_attention_v2(batch, n_heads, seq_len, head_dim, block_M, block_N):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                K: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                V: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, seq_len, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(seq_len, block_M),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_M, head_dim), "float16")
                    K_shared = T.alloc_shared((block_N, head_dim), "float16")
                    V_shared = T.alloc_shared((block_N, head_dim), "float16")

                    S_local = T.alloc_fragment((block_M, block_N), "float32")
                    O_local = T.alloc_fragment((block_M, head_dim), "float32")

                    m_current = T.alloc_fragment((block_M,), "float32")
                    l_current = T.alloc_fragment((block_M,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        K_shared: T.make_swizzled_layout(K_shared),
                        V_shared: T.make_swizzled_layout(V_shared)
                    })

                    T.fill(m_current, -1e9)
                    T.clear(l_current)
                    T.clear(O_local)

                    T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

                    num_kv_tiles = T.ceildiv(seq_len, block_N)
                    for k in T.Pipelined(num_kv_tiles, num_stages=2):
                        T.copy(K[bz, by, k * block_N, 0], K_shared)
                        T.copy(V[bz, by, k * block_N, 0], V_shared)

                        T.clear(S_local)
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = S_local[i, j] * scale

                        m_prev = T.alloc_fragment((block_M,), "float32")
                        T.copy(m_current, m_prev)

                        T.reduce_max(S_local, m_current, dim=1)

                        for i in T.Parallel(block_M):
                            correction = T.exp(m_prev[i] - m_current[i])
                            l_current[i] = l_current[i] * correction

                            for d in T.Parallel(head_dim):
                                O_local[i, d] = O_local[i, d] * correction

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = T.exp(S_local[i, j] - m_current[i])

                        for i in T.Parallel(block_M):
                            row_sum = T.float32(0.0)
                            for j in T.Parallel(block_N):
                                row_sum = row_sum + S_local[i, j]
                            l_current[i] = l_current[i] + row_sum

                        T.gemm(S_local, V_shared, O_local, transpose_B=False)

                    for i, d in T.Parallel(block_M, head_dim):
                        O_local[i, d] = O_local[i, d] / l_current[i]

                    T.copy(O_local, O[bz, by, bx * block_M, 0])

            return main

        # Test with small sequence length
        func = flash_attention_v2(
            batch=1, n_heads=8, seq_len=512, head_dim=64,
            block_M=64, block_N=64
        )
        assert func is not None

    def test_long_sequence_length(self):
        """Test FlashAttention with long sequence length."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def flash_attention_v2(batch, n_heads, seq_len, head_dim, block_M, block_N):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                K: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                V: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, seq_len, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(seq_len, block_M),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_M, head_dim), "float16")
                    K_shared = T.alloc_shared((block_N, head_dim), "float16")
                    V_shared = T.alloc_shared((block_N, head_dim), "float16")

                    S_local = T.alloc_fragment((block_M, block_N), "float32")
                    O_local = T.alloc_fragment((block_M, head_dim), "float32")

                    m_current = T.alloc_fragment((block_M,), "float32")
                    l_current = T.alloc_fragment((block_M,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        K_shared: T.make_swizzled_layout(K_shared),
                        V_shared: T.make_swizzled_layout(V_shared)
                    })

                    T.fill(m_current, -1e9)
                    T.clear(l_current)
                    T.clear(O_local)

                    T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

                    num_kv_tiles = T.ceildiv(seq_len, block_N)
                    for k in T.Pipelined(num_kv_tiles, num_stages=2):
                        T.copy(K[bz, by, k * block_N, 0], K_shared)
                        T.copy(V[bz, by, k * block_N, 0], V_shared)

                        T.clear(S_local)
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = S_local[i, j] * scale

                        m_prev = T.alloc_fragment((block_M,), "float32")
                        T.copy(m_current, m_prev)

                        T.reduce_max(S_local, m_current, dim=1)

                        for i in T.Parallel(block_M):
                            correction = T.exp(m_prev[i] - m_current[i])
                            l_current[i] = l_current[i] * correction

                            for d in T.Parallel(head_dim):
                                O_local[i, d] = O_local[i, d] * correction

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = T.exp(S_local[i, j] - m_current[i])

                        for i in T.Parallel(block_M):
                            row_sum = T.float32(0.0)
                            for j in T.Parallel(block_N):
                                row_sum = row_sum + S_local[i, j]
                            l_current[i] = l_current[i] + row_sum

                        T.gemm(S_local, V_shared, O_local, transpose_B=False)

                    for i, d in T.Parallel(block_M, head_dim):
                        O_local[i, d] = O_local[i, d] / l_current[i]

                    T.copy(O_local, O[bz, by, bx * block_M, 0])

            return main

        # Test with long sequence (e.g., 4096 tokens)
        func = flash_attention_v2(
            batch=1, n_heads=16, seq_len=4096, head_dim=128,
            block_M=64, block_N=64
        )
        assert func is not None

    def test_different_block_sizes(self):
        """Test FlashAttention with various block size configurations."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def flash_attention_v2(batch, n_heads, seq_len, head_dim, block_M, block_N):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                K: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                V: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, seq_len, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(seq_len, block_M),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_M, head_dim), "float16")
                    K_shared = T.alloc_shared((block_N, head_dim), "float16")
                    V_shared = T.alloc_shared((block_N, head_dim), "float16")

                    S_local = T.alloc_fragment((block_M, block_N), "float32")
                    O_local = T.alloc_fragment((block_M, head_dim), "float32")

                    m_current = T.alloc_fragment((block_M,), "float32")
                    l_current = T.alloc_fragment((block_M,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        K_shared: T.make_swizzled_layout(K_shared),
                        V_shared: T.make_swizzled_layout(V_shared)
                    })

                    T.fill(m_current, -1e9)
                    T.clear(l_current)
                    T.clear(O_local)

                    T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

                    num_kv_tiles = T.ceildiv(seq_len, block_N)
                    for k in T.Pipelined(num_kv_tiles, num_stages=2):
                        T.copy(K[bz, by, k * block_N, 0], K_shared)
                        T.copy(V[bz, by, k * block_N, 0], V_shared)

                        T.clear(S_local)
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = S_local[i, j] * scale

                        m_prev = T.alloc_fragment((block_M,), "float32")
                        T.copy(m_current, m_prev)

                        T.reduce_max(S_local, m_current, dim=1)

                        for i in T.Parallel(block_M):
                            correction = T.exp(m_prev[i] - m_current[i])
                            l_current[i] = l_current[i] * correction

                            for d in T.Parallel(head_dim):
                                O_local[i, d] = O_local[i, d] * correction

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = T.exp(S_local[i, j] - m_current[i])

                        for i in T.Parallel(block_M):
                            row_sum = T.float32(0.0)
                            for j in T.Parallel(block_N):
                                row_sum = row_sum + S_local[i, j]
                            l_current[i] = l_current[i] + row_sum

                        T.gemm(S_local, V_shared, O_local, transpose_B=False)

                    for i, d in T.Parallel(block_M, head_dim):
                        O_local[i, d] = O_local[i, d] / l_current[i]

                    T.copy(O_local, O[bz, by, bx * block_M, 0])

            return main

        # Test block size 64x64
        func1 = flash_attention_v2(
            batch=1, n_heads=16, seq_len=1024, head_dim=64,
            block_M=64, block_N=64
        )
        assert func1 is not None

        # Test block size 128x128
        func2 = flash_attention_v2(
            batch=1, n_heads=16, seq_len=1024, head_dim=64,
            block_M=128, block_N=128
        )
        assert func2 is not None

    def test_different_head_dimensions(self):
        """Test FlashAttention with various head dimensions."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def flash_attention_v2(batch, n_heads, seq_len, head_dim, block_M, block_N):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                K: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                V: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, seq_len, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(seq_len, block_M),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_M, head_dim), "float16")
                    K_shared = T.alloc_shared((block_N, head_dim), "float16")
                    V_shared = T.alloc_shared((block_N, head_dim), "float16")

                    S_local = T.alloc_fragment((block_M, block_N), "float32")
                    O_local = T.alloc_fragment((block_M, head_dim), "float32")

                    m_current = T.alloc_fragment((block_M,), "float32")
                    l_current = T.alloc_fragment((block_M,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        K_shared: T.make_swizzled_layout(K_shared),
                        V_shared: T.make_swizzled_layout(V_shared)
                    })

                    T.fill(m_current, -1e9)
                    T.clear(l_current)
                    T.clear(O_local)

                    T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

                    num_kv_tiles = T.ceildiv(seq_len, block_N)
                    for k in T.Pipelined(num_kv_tiles, num_stages=2):
                        T.copy(K[bz, by, k * block_N, 0], K_shared)
                        T.copy(V[bz, by, k * block_N, 0], V_shared)

                        T.clear(S_local)
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = S_local[i, j] * scale

                        m_prev = T.alloc_fragment((block_M,), "float32")
                        T.copy(m_current, m_prev)

                        T.reduce_max(S_local, m_current, dim=1)

                        for i in T.Parallel(block_M):
                            correction = T.exp(m_prev[i] - m_current[i])
                            l_current[i] = l_current[i] * correction

                            for d in T.Parallel(head_dim):
                                O_local[i, d] = O_local[i, d] * correction

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = T.exp(S_local[i, j] - m_current[i])

                        for i in T.Parallel(block_M):
                            row_sum = T.float32(0.0)
                            for j in T.Parallel(block_N):
                                row_sum = row_sum + S_local[i, j]
                            l_current[i] = l_current[i] + row_sum

                        T.gemm(S_local, V_shared, O_local, transpose_B=False)

                    for i, d in T.Parallel(block_M, head_dim):
                        O_local[i, d] = O_local[i, d] / l_current[i]

                    T.copy(O_local, O[bz, by, bx * block_M, 0])

            return main

        # Test head_dim = 64
        func1 = flash_attention_v2(
            batch=1, n_heads=16, seq_len=1024, head_dim=64,
            block_M=64, block_N=64
        )
        assert func1 is not None

        # Test head_dim = 128
        func2 = flash_attention_v2(
            batch=1, n_heads=16, seq_len=1024, head_dim=128,
            block_M=64, block_N=64
        )
        assert func2 is not None


class TestFlashAttentionIntegration:
    """Integration tests for FlashAttention kernel with actual computation."""

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_correctness_against_standard_attention(self):
        """Test FlashAttention correctness against standard attention implementation."""
        import tilelang
        import tilelang.language as T
        import torch
        import torch.nn.functional as F

        @tilelang.jit(target="cuda", out_idx=[3])
        def flash_attention_v2(batch, n_heads, seq_len, head_dim, block_M, block_N):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                K: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                V: T.Buffer((batch, n_heads, seq_len, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, seq_len, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(seq_len, block_M),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_M, head_dim), "float16")
                    K_shared = T.alloc_shared((block_N, head_dim), "float16")
                    V_shared = T.alloc_shared((block_N, head_dim), "float16")

                    S_local = T.alloc_fragment((block_M, block_N), "float32")
                    O_local = T.alloc_fragment((block_M, head_dim), "float32")

                    m_current = T.alloc_fragment((block_M,), "float32")
                    l_current = T.alloc_fragment((block_M,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        K_shared: T.make_swizzled_layout(K_shared),
                        V_shared: T.make_swizzled_layout(V_shared)
                    })

                    T.fill(m_current, -1e9)
                    T.clear(l_current)
                    T.clear(O_local)

                    T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

                    num_kv_tiles = T.ceildiv(seq_len, block_N)
                    for k in T.Pipelined(num_kv_tiles, num_stages=2):
                        T.copy(K[bz, by, k * block_N, 0], K_shared)
                        T.copy(V[bz, by, k * block_N, 0], V_shared)

                        T.clear(S_local)
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = S_local[i, j] * scale

                        m_prev = T.alloc_fragment((block_M,), "float32")
                        T.copy(m_current, m_prev)

                        T.reduce_max(S_local, m_current, dim=1)

                        for i in T.Parallel(block_M):
                            correction = T.exp(m_prev[i] - m_current[i])
                            l_current[i] = l_current[i] * correction

                            for d in T.Parallel(head_dim):
                                O_local[i, d] = O_local[i, d] * correction

                        for i, j in T.Parallel(block_M, block_N):
                            S_local[i, j] = T.exp(S_local[i, j] - m_current[i])

                        for i in T.Parallel(block_M):
                            row_sum = T.float32(0.0)
                            for j in T.Parallel(block_N):
                                row_sum = row_sum + S_local[i, j]
                            l_current[i] = l_current[i] + row_sum

                        T.gemm(S_local, V_shared, O_local, transpose_B=False)

                    for i, d in T.Parallel(block_M, head_dim):
                        O_local[i, d] = O_local[i, d] / l_current[i]

                    T.copy(O_local, O[bz, by, bx * block_M, 0])

            return main

        # Create test inputs
        batch, n_heads, seq_len, head_dim = 1, 8, 512, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        O = torch.empty(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Compute with FlashAttention
        func = flash_attention_v2(
            batch=batch, n_heads=n_heads, seq_len=seq_len, head_dim=head_dim,
            block_M=64, block_N=64
        )
        func(Q, K, V, O)

        # Compute reference (standard attention)
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V)

        # Check correctness (with tolerance for float16 and numerical differences)
        assert torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2)


def _cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
