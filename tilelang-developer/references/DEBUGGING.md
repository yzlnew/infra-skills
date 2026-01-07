# TileLang Debugging & Troubleshooting Guide

Common errors, performance issues, and their solutions.

## Table of Contents

1. [Compilation Errors](#compilation-errors)
2. [Runtime Errors](#runtime-errors)
3. [Performance Issues](#performance-issues)
4. [Correctness Issues](#correctness-issues)
5. [Hardware-Specific Issues](#hardware-specific-issues)

---

## Compilation Errors

### Error: "Shared memory size exceeded"

**Symptom:**
```
RuntimeError: Shared memory allocation exceeded hardware limit
Required: 196608 bytes, Available: 163840 bytes (on A100)
```

**Cause:** Total shared memory allocation exceeds GPU limits.

**Hardware Limits:**
- NVIDIA A100: 164 KB per block
- NVIDIA H100: 228 KB per block
- AMD MI300X: 64 KB per block

**Solutions:**

1. **Reduce block dimensions:**
   ```python
   # Before: 128x128 blocks with stage-3 pipeline = 3 * 2 * 128 * 128 * 2 bytes = 196 KB
   block_M, block_N = 128, 128

   # After: Use 64x64 blocks
   block_M, block_N = 64, 64  # Reduces to 49 KB
   ```

2. **Reduce pipeline stages:**
   ```python
   # Before: 3-stage pipeline
   for k in T.Pipelined(num_blocks, num_stages=3):

   # After: 2-stage pipeline
   for k in T.Pipelined(num_blocks, num_stages=2):
   ```

3. **Use mixed precision strategically:**
   ```python
   # Store less critical data in lower precision
   A_shared = T.alloc_shared((block_M, block_K), "float16")  # 2 bytes
   # Instead of float32 (4 bytes) where not needed
   ```

**SMEM calculation formula:**
```
Total SMEM = (A_shared_size + B_shared_size + ...) * num_stages
```

---

### Error: "Shape mismatch in T.gemm"

**Symptom:**
```
ValueError: Shape mismatch: cannot multiply (128, 32) by (64, 128)
```

**Cause:** Incompatible dimensions in matrix multiplication.

**Solution:** Verify dimension alignment:
```python
# For C += A @ B:
# A must be (M, K)
# B must be (K, N)
# C must be (M, N)

# If transposing:
T.gemm(A, B, C, transpose_A=False, transpose_B=True)
# Then B shape should be (N, K) before transpose
```

**Debug approach:**
```python
# Add shape verification
print(f"A_shared: {A_shared.shape}")  # Should be (block_M, block_K)
print(f"B_shared: {B_shared.shape}")  # Should be (block_K, block_N)
print(f"C_local: {C_local.shape}")    # Should be (block_M, block_N)
```

---

### Error: "Invalid buffer access / Index out of bounds"

**Symptom:**
```
IndexError: Buffer access out of bounds
Accessing A[2048, 1024] but buffer shape is (2048, 1024)
```

**Cause:** Zero-based indexing or incorrect boundary handling.

**Solution:** Use proper boundary checking:
```python
# Wrong: May access beyond bounds
T.copy(A[by * block_M, k * block_K], A_shared)

# Right: Add boundary checks for non-divisible dimensions
if by * block_M < M and k * block_K < K:
    T.copy(A[by * block_M, k * block_K], A_shared)

# Or use T.ceildiv to ensure proper grid coverage
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128):
```

**TileLang often handles this automatically**, but verify edge cases.

---

### Error: "Unknown attribute 'policy' in T.gemm"

**Symptom:**
```
AttributeError: T.gemm() got an unexpected keyword argument 'policy'
```

**Cause:** Outdated TileLang version or incorrect import.

**Solution:**
```python
# Ensure correct import
import tilelang.language as T

# Verify TileLang version
import tilelang
print(tilelang.__version__)  # Should be >= 0.1.6

# Update if needed
# pip install --upgrade tilelang
```

---

## Runtime Errors

### CUDA Error: Illegal memory access

**Symptom:**
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**Common Causes:**

1. **Race condition in shared memory:**
   ```python
   # Missing synchronization
   T.copy(A[...], A_shared)
   T.gemm(A_shared, B_shared, C_local)  # May read before copy completes

   # Fix: T.Pipelined handles this automatically
   for k in T.Pipelined(...):
       T.copy(A[...], A_shared)
       T.gemm(A_shared, B_shared, C_local)
   ```

2. **Incorrect indexing:**
   ```python
   # Wrong: Column-major indexing for row-major layout
   T.copy(A[k * block_K, by * block_M], A_shared)  # Transposed by mistake

   # Correct:
   T.copy(A[by * block_M, k * block_K], A_shared)
   ```

3. **Bank conflict causing unexpected behavior:**
   - Ensure `T.make_swizzled_layout` is applied to all shared memory buffers

---

### Error: Results are all zeros

**Symptom:** Output tensor contains only zeros after kernel execution.

**Common Causes:**

1. **Forgot to initialize accumulator:**
   ```python
   # Missing initialization
   C_local = T.alloc_fragment((block_M, block_N), "float32")
   T.clear(C_local)  # ← MUST add this
   ```

2. **Wrong output index in @tilelang.jit:**
   ```python
   # Wrong: out_idx not specified
   @tilelang.jit(target="cuda")

   # Correct: Specify which parameter is output
   @tilelang.jit(target="cuda", out_idx=[2])  # C is the 3rd parameter (index 2)
   def func(A, B, C):
   ```

3. **Forgot to copy results back:**
   ```python
   # Missing final copy
   T.copy(C_local, C[by * block_M, bx * block_N])  # ← Must add this
   ```

---

### Error: Results contain NaN or Inf

**Symptom:** Output contains NaN (Not a Number) or Inf (Infinity).

**Common Causes:**

1. **Division by zero in softmax:**
   ```python
   # Wrong: Can divide by zero
   for i, j in T.Parallel(M, N):
       output[i, j] = scores[i, j] / sum_scores[i]

   # Correct: Add epsilon
   for i, j in T.Parallel(M, N):
       output[i, j] = scores[i, j] / (sum_scores[i] + 1e-8)
   ```

2. **Overflow in accumulation:**
   ```python
   # Wrong: Using float16 for accumulator
   C_local = T.alloc_fragment((block_M, block_N), "float16")

   # Correct: Use float32 for accumulation
   C_local = T.alloc_fragment((block_M, block_N), "float32")
   ```

3. **Numerical instability in softmax:**
   ```python
   # Wrong: Direct exp without max subtraction
   for i, j in T.Parallel(M, N):
       output[i, j] = T.exp(scores[i, j])

   # Correct: Subtract max for stability
   max_val = T.reduce_max(scores, dim=1)
   for i, j in T.Parallel(M, N):
       output[i, j] = T.exp(scores[i, j] - max_val[i])
   ```

---

## Performance Issues

### Issue: Performance much lower than expected

**Symptoms:**
- Achieving only 30-50% of theoretical peak
- Much slower than cuBLAS/vendor libraries

**Diagnostic Steps:**

1. **Check if swizzle layout is applied:**
   ```python
   # MUST have this for good performance
   T.annotate_layout({
       A_shared: T.make_swizzled_layout(A_shared),
       B_shared: T.make_swizzled_layout(B_shared)
   })
   ```

2. **Verify pipelining is enabled:**
   ```python
   # Without pipelining - BAD
   for k in range(num_blocks):
       T.copy(...)
       T.gemm(...)

   # With pipelining - GOOD
   for k in T.Pipelined(num_blocks, num_stages=3):
       T.copy(...)
       T.gemm(...)
   ```

3. **Check block dimensions are optimal:**
   ```python
   # Too small - low occupancy
   block_M, block_N, block_K = 32, 32, 16  # BAD

   # Good balance
   block_M, block_N, block_K = 128, 128, 32  # GOOD for A100
   ```

4. **Verify block_K is aligned:**
   ```python
   # Misaligned - can't use Tensor Core efficiently
   block_K = 24  # BAD

   # Aligned to 16 or 32
   block_K = 32  # GOOD
   ```

---

### Issue: Low GPU occupancy

**Symptom:** GPU utilization < 60% in nvidia-smi.

**Causes & Solutions:**

1. **Too few thread blocks:**
   ```python
   # Calculate expected blocks
   num_blocks = (N // block_N) * (M // block_M)
   num_sms = 108  # A100 has 108 SMs

   # Need at least 2-4 blocks per SM
   if num_blocks < num_sms * 2:
       # Reduce block size to increase block count
       block_M = block_M // 2
       block_N = block_N // 2
   ```

2. **Too much shared memory per block:**
   - Limits number of concurrent blocks per SM
   - Solution: Reduce block size or pipeline stages

3. **Too many registers per thread:**
   - Compiler may allocate too many registers
   - Solution: Reduce fragment sizes or simplify computation

**Profiling command:**
```bash
# Use NVIDIA Nsight Compute
ncu --set full --launch-count 1 python your_script.py

# Key metrics to check:
# - SM Efficiency (should be > 70%)
# - Memory Throughput (should be > 60% of peak)
# - Warp Execution Efficiency (should be > 80%)
```

---

### Issue: Bank conflicts detected

**Symptom:** Low shared memory bandwidth (< 1 TB/s on A100).

**Detection:**
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum python script.py
```

**Solution:** ALWAYS use swizzle layout:
```python
# This is NOT optional for performance
T.annotate_layout({
    A_shared: T.make_swizzled_layout(A_shared),
    B_shared: T.make_swizzled_layout(B_shared)
})
```

If conflicts persist, verify access patterns are coalesced.

---

### Issue: Pipelining not effective

**Symptom:** No performance gain from increasing num_stages.

**Possible Causes:**

1. **Computation too fast (memory bound):**
   - Pipeline stages don't help if compute is instant
   - Try larger block sizes to increase compute density

2. **Pipeline not properly formed:**
   ```python
   # Wrong: Dependencies prevent pipelining
   for k in T.Pipelined(num_blocks, num_stages=3):
       T.copy(A[...], A_shared)
       T.gemm(A_shared, B_shared, C_local)
       # Synchronization here breaks pipeline
       if k == 0:
           T.clear(temp)

   # Correct: Keep loop body uniform
   for k in T.Pipelined(num_blocks, num_stages=3):
       T.copy(A[...], A_shared)
       T.gemm(A_shared, B_shared, C_local)
   ```

3. **Not enough work to hide latency:**
   - Need K dimension large enough (K > 1024 typically)
   - For small K, pipelining overhead may dominate

---

## Correctness Issues

### Issue: Results differ slightly from reference

**Symptom:** Numerical differences in output (e.g., 1e-3 error).

**Likely Causes:**

1. **Expected FP16 precision loss:**
   - FP16 has ~3 decimal digits of precision
   - Differences up to 1e-3 are normal for large reductions

2. **Different accumulation order:**
   - TileLang may accumulate in different order than reference
   - Solution: Use higher precision accumulator (float32)

3. **Fused operations:**
   - Fused multiply-add has better precision than separate ops
   - This is actually MORE accurate than reference

**Validation approach:**
```python
import torch

# Use appropriate tolerance for FP16
torch.testing.assert_close(
    tilelang_output,
    reference_output,
    rtol=1e-3,  # Relative tolerance
    atol=1e-3   # Absolute tolerance
)
```

---

### Issue: Results match for small inputs but fail for large

**Symptom:** Works for 128×128 but fails for 4096×4096.

**Likely Causes:**

1. **Integer overflow in index calculation:**
   ```python
   # May overflow for large matrices
   offset = by * block_M * K + k * block_K  # If using int32

   # Solution: Use int64 or split calculation
   row_offset = by * block_M
   col_offset = k * block_K
   ```

2. **Accumulation overflow:**
   ```python
   # FP16 accumulator overflows for large K
   C_local = T.alloc_fragment((block_M, block_N), "float16")  # BAD

   # Use FP32 accumulator
   C_local = T.alloc_fragment((block_M, block_N), "float32")  # GOOD
   ```

3. **Shared memory reuse bugs:**
   - Verify synchronization in pipelines
   - Check that buffers are properly cleared between iterations

---

## Hardware-Specific Issues

### NVIDIA Hopper (H100) Issues

**Issue:** Code runs slower on H100 than A100.

**Possible Cause:** Not leveraging Hopper-specific features.

**Solution:** Use larger blocks and ensure TMA is triggered:
```python
# H100 can handle larger tiles
block_M, block_N = 256, 128  # vs 128, 128 on A100

# Ensure pipeline depth leverages async features
for k in T.Pipelined(num_blocks, num_stages=4):  # vs 3 on A100
```

---

### AMD MI300X Issues

**Issue:** Code works on NVIDIA but fails on AMD.

**Solutions:**

1. **Verify target is set correctly:**
   ```python
   @tilelang.jit(target="hip")  # Not "cuda"
   ```

2. **Check shared memory limits:**
   - MI300X has only 64KB per block vs 164KB on A100
   - May need smaller blocks or fewer pipeline stages

3. **Test with both backends:**
   ```python
   # TileLang can compile to HIP via TVM
   # Verify correctness on both targets
   ```

---

### Huawei Ascend NPU Issues

**Issue:** Performance not as expected on Ascend.

**Notes:**
- Ascend backend is more experimental
- May require specific block sizes
- Check TileLang Ascend documentation for NPU-specific tuning

---

## Debugging Tools & Techniques

### Print Debugging

```python
# Add debug prints in kernel
for i, j in T.Parallel(block_M, block_N):
    if i == 0 and j == 0:  # Only print from one thread
        T.print("Debug:", C_local[i, j])
```

### Verify Intermediate Results

```python
# Copy intermediate results to global memory for inspection
debug_buffer = T.Buffer((M, N), "float32")
T.copy(A_shared, debug_buffer[by * block_M, 0])
# Check debug_buffer from host
```

### Simplify to Isolate Issue

```python
# Remove pipelining
for k in range(num_blocks):  # Instead of T.Pipelined

# Remove swizzle
# T.annotate_layout(...)  # Comment out

# Use simpler dimensions
M = N = K = 128  # Instead of 4096
```

### Profile with NVIDIA Tools

```bash
# Nsight Compute for detailed metrics
ncu --set full python script.py

# Nsight Systems for timeline view
nsys profile --stats=true python script.py
```

---

## Best Practices to Avoid Issues

1. **Always use swizzle layout** for shared memory in GEMM-like kernels
2. **Always use FP32 accumulators** for matrix multiplication
3. **Always use T.Pipelined** for multi-iteration memory transfers
4. **Always initialize accumulators** with T.clear()
5. **Always align block_K** to 16 or 32 for Tensor Core usage
6. **Test with small dimensions first** before scaling up
7. **Validate correctness** before optimizing performance
8. **Profile to identify bottlenecks** rather than guessing

---

## Getting Help

If issues persist after trying these solutions:

1. **Check TileLang GitHub issues:** https://github.com/tile-ai/tilelang/issues
2. **Consult examples:** Look for similar kernels in tilelang/examples/
3. **Enable verbose compilation:** May reveal optimization issues
4. **Compare with reference:** Diff against known-working CUDA code
5. **Ask community:** TVM/TileLang forums and Discord
