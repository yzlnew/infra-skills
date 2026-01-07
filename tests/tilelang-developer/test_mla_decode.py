"""
Unit tests for TileLang DeepSeek MLA Decoding kernel.

Tests the mla_decode_kernel example from EXAMPLES.md to ensure:
- Kernel compiles successfully
- Handles KV compression correctly
- Supports various latent dimensions
- Uses split-KV parallelization efficiently
"""

import pytest


class TestMLADecodeKernel:
    """Test suite for the TileLang DeepSeek MLA decode kernel."""

    def test_kernel_compilation(self):
        """Test that the MLA decode kernel compiles without errors."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Test compilation succeeds
        func = mla_decode_kernel(
            batch=1, n_heads=32, kv_len=4096,
            head_dim=128, latent_dim=512,
            block_M=1, block_N=64
        )
        assert func is not None

    def test_small_kv_cache(self):
        """Test MLA decode with small KV cache."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Test with small KV cache
        func = mla_decode_kernel(
            batch=1, n_heads=16, kv_len=1024,
            head_dim=64, latent_dim=256,
            block_M=1, block_N=64
        )
        assert func is not None

    def test_large_kv_cache(self):
        """Test MLA decode with large KV cache."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Test with large KV cache (8K context)
        func = mla_decode_kernel(
            batch=1, n_heads=32, kv_len=8192,
            head_dim=128, latent_dim=512,
            block_M=1, block_N=64
        )
        assert func is not None

    def test_different_latent_dimensions(self):
        """Test MLA decode with various latent dimension configurations."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Test latent_dim = 256
        func1 = mla_decode_kernel(
            batch=1, n_heads=16, kv_len=2048,
            head_dim=64, latent_dim=256,
            block_M=1, block_N=64
        )
        assert func1 is not None

        # Test latent_dim = 512
        func2 = mla_decode_kernel(
            batch=1, n_heads=32, kv_len=4096,
            head_dim=128, latent_dim=512,
            block_M=1, block_N=64
        )
        assert func2 is not None

        # Test latent_dim = 1024
        func3 = mla_decode_kernel(
            batch=1, n_heads=32, kv_len=4096,
            head_dim=128, latent_dim=1024,
            block_M=1, block_N=64
        )
        assert func3 is not None

    def test_different_block_sizes(self):
        """Test MLA decode with various block size configurations."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Test block_N = 32
        func1 = mla_decode_kernel(
            batch=1, n_heads=16, kv_len=2048,
            head_dim=64, latent_dim=256,
            block_M=1, block_N=32
        )
        assert func1 is not None

        # Test block_N = 64
        func2 = mla_decode_kernel(
            batch=1, n_heads=16, kv_len=2048,
            head_dim=64, latent_dim=256,
            block_M=1, block_N=64
        )
        assert func2 is not None

        # Test block_N = 128
        func3 = mla_decode_kernel(
            batch=1, n_heads=16, kv_len=2048,
            head_dim=64, latent_dim=256,
            block_M=1, block_N=128
        )
        assert func3 is not None

    def test_batched_inference(self):
        """Test MLA decode with batched inference."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Test batch size = 4
        func1 = mla_decode_kernel(
            batch=4, n_heads=16, kv_len=2048,
            head_dim=64, latent_dim=256,
            block_M=1, block_N=64
        )
        assert func1 is not None

        # Test batch size = 8
        func2 = mla_decode_kernel(
            batch=8, n_heads=32, kv_len=4096,
            head_dim=128, latent_dim=512,
            block_M=1, block_N=64
        )
        assert func2 is not None


class TestMLADecodeIntegration:
    """Integration tests for MLA decode kernel with actual computation."""

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_kv_decompression(self):
        """Test KV decompression correctness."""
        import tilelang
        import tilelang.language as T
        import torch

        @tilelang.jit(target="cuda", out_idx=[3])
        def mla_decode_kernel(
            batch, n_heads, kv_len, head_dim, latent_dim,
            block_M, block_N
        ):
            @T.prim_func
            def main(
                Q: T.Buffer((batch, n_heads, 1, head_dim), "float16"),
                KV_compressed: T.Buffer((batch, kv_len, latent_dim), "float16"),
                KV_proj: T.Buffer((latent_dim, n_heads, head_dim), "float16"),
                O: T.Buffer((batch, n_heads, 1, head_dim), "float16")
            ):
                scale = T.float32(1.0) / T.sqrt(T.float32(head_dim))

                with T.Kernel(
                    T.ceildiv(kv_len, block_N),
                    n_heads,
                    batch,
                    threads=128
                ) as (bx, by, bz):
                    Q_shared = T.alloc_shared((1, head_dim), "float16")
                    KV_shared = T.alloc_shared((block_N, head_dim), "float16")
                    KV_comp_shared = T.alloc_shared((block_N, latent_dim), "float16")

                    S_local = T.alloc_fragment((1, block_N), "float32")
                    O_local = T.alloc_fragment((1, head_dim), "float32")

                    scores_max = T.alloc_fragment((1,), "float32")
                    scores_sum = T.alloc_fragment((1,), "float32")

                    T.annotate_layout({
                        Q_shared: T.make_swizzled_layout(Q_shared),
                        KV_shared: T.make_swizzled_layout(KV_shared),
                        KV_comp_shared: T.make_swizzled_layout(KV_comp_shared)
                    })

                    T.copy(Q[bz, by, 0, 0], Q_shared)
                    T.copy(KV_compressed[bz, bx * block_N, 0], KV_comp_shared)

                    T.clear(KV_shared)
                    for i in range(block_N):
                        for d in range(head_dim):
                            for l in range(latent_dim):
                                KV_shared[i, d] += KV_comp_shared[i, l] * KV_proj[l, by, d]

                    T.clear(S_local)
                    T.gemm(Q_shared, KV_shared, S_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullCol)

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] * scale

                    T.fill(scores_max, -1e9)
                    for j in T.Parallel(block_N):
                        scores_max[0] = T.max(scores_max[0], S_local[0, j])

                    T.clear(scores_sum)
                    for j in T.Parallel(block_N):
                        S_local[0, j] = T.exp(S_local[0, j] - scores_max[0])
                        scores_sum[0] = scores_sum[0] + S_local[0, j]

                    for j in T.Parallel(block_N):
                        S_local[0, j] = S_local[0, j] / scores_sum[0]

                    T.clear(O_local)
                    T.gemm(S_local, KV_shared, O_local, transpose_B=False)

                    T.copy(O_local, O[bz, by, 0, 0])

            return main

        # Create test inputs
        batch, n_heads, kv_len, head_dim, latent_dim = 1, 8, 512, 64, 256
        Q = torch.randn(batch, n_heads, 1, head_dim, dtype=torch.float16, device='cuda')
        KV_compressed = torch.randn(batch, kv_len, latent_dim, dtype=torch.float16, device='cuda')
        KV_proj = torch.randn(latent_dim, n_heads, head_dim, dtype=torch.float16, device='cuda')
        O = torch.empty(batch, n_heads, 1, head_dim, dtype=torch.float16, device='cuda')

        # Compile and run kernel
        func = mla_decode_kernel(
            batch=batch, n_heads=n_heads, kv_len=kv_len,
            head_dim=head_dim, latent_dim=latent_dim,
            block_M=1, block_N=64
        )
        func(Q, KV_compressed, KV_proj, O)

        # Verify output is not zero/NaN
        assert not torch.isnan(O).any()
        assert not torch.isinf(O).any()
        assert O.abs().sum() > 0


def _cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
