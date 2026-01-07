# TileLang API Reference

Complete API reference for TileLang primitives and functions.

## Table of Contents

1. [Decorators & Context Managers](#decorators--context-managers)
2. [Memory Management](#memory-management)
3. [Data Movement & Computation](#data-movement--computation)
4. [Scheduling & Optimization](#scheduling--optimization)

---

## Decorators & Context Managers

### @tilelang.jit

JIT compilation decorator for TileLang kernels.

```python
@tilelang.jit(target="cuda", out_idx=[2])
def kernel_func(...):
    ...
```

**Parameters:**
- `target`: Compilation target - `"cuda"` (NVIDIA), `"hip"` (AMD), `"cpu"`
- `out_idx`: List of output tensor indices (optional)

### @T.prim_func

Marks a function as a primitive TileLang function (kernel entry point).

```python
@T.prim_func
def main(A: T.Buffer(...), B: T.Buffer(...)):
    ...
```

### T.Kernel

Context manager defining kernel launch configuration and grid dimensions.

```python
with T.Kernel(grid_x, grid_y, threads=N) as (bx, by):
    # bx, by correspond to blockIdx.x, blockIdx.y
    # threads=N specifies threads per block
    ...
```

**Parameters:**
- `grid_x`, `grid_y`: Grid dimensions (number of thread blocks)
- `threads`: Number of threads per block (default: 128)
- Returns block indices `(bx, by)` for indexing

**Example:**
```python
# Process matrix in 128x128 tiles with 128 threads per block
with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
    ...
```

---

## Memory Management

### T.Buffer

Type annotation for tensor parameters with shape and dtype.

```python
A: T.Buffer((M, K), "float16")
```

**Parameters:**
- Shape tuple (can include symbolic dimensions)
- Data type: `"float16"`, `"float32"`, `"int8"`, `"bfloat16"`, etc.

### T.alloc_shared

Allocate shared memory (on-chip L1 cache, ~164KB on A100).

```python
A_shared = T.alloc_shared((block_M, block_K), "float16")
```

**Key Points:**
- Shared across threads in a thread block
- Much faster than global memory but limited in size
- **Critical:** Apply swizzle layout to avoid bank conflicts

### T.alloc_fragment

Allocate register fragment (highest bandwidth, per-thread private storage).

```python
C_local = T.alloc_fragment((block_M, block_N), "float32")
```

**Key Points:**
- Stored in register file (fastest memory)
- Used for accumulators and small working sets
- Compiler automatically distributes across threads

### T.make_swizzled_layout

Create optimized memory layout to eliminate bank conflicts.

```python
swizzle_layout = T.make_swizzled_layout(A_shared)
```

**Purpose:** Transforms memory access pattern to avoid concurrent accesses to the same memory bank.

### T.annotate_layout

Apply memory layout to buffers.

```python
T.annotate_layout({
    A_shared: T.make_swizzled_layout(A_shared),
    B_shared: T.make_swizzled_layout(B_shared)
})
```

**Critical for performance:** Always apply swizzle layouts to shared memory buffers used in matrix operations.

---

## Data Movement & Computation

### T.copy

Intelligent data transfer between memory hierarchies.

```python
T.copy(source, destination)
```

**Automatically selects optimal instruction based on memory levels:**
- Global → Shared: `cp.async` (Ampere) or `TMA` (Hopper)
- Shared → Register: `ldmatrix` (for Tensor Core)
- Register → Global: Vectorized stores

**Examples:**
```python
# Load tile from global memory to shared memory
T.copy(A[by * block_M, k * block_K], A_shared)

# Load from shared to register fragment
T.copy(A_shared, A_local)

# Write results back to global memory
T.copy(C_local, C[by * block_M, bx * block_N])
```

### T.gemm

Matrix multiplication using Tensor Cores or WMMA instructions.

```python
T.gemm(A, B, C, transpose_A=False, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
```

**Parameters:**
- `A`, `B`: Input matrices (shared memory or register fragments)
- `C`: Accumulator (register fragment) - performs `C += A @ B`
- `transpose_A`, `transpose_B`: Whether to transpose inputs
- `policy`: Warp parallelization strategy

**GemmWarpPolicy Options:**
- `FullRow`: One warp processes full row (good for attention mechanisms)
- `FullCol`: One warp processes full column (good for MLA structures)
- `Square`: Balanced 2D distribution

**Hardware mapping:**
- Volta/Turing: `mma.sync` instructions
- Ampere: `mma.sync` with optimized schedules
- Hopper: `wgmma.mma_async` with asynchronous execution

### T.gemm_sp

Sparse matrix multiplication using 2:4 structured sparsity (Sparse Tensor Core).

```python
T.gemm_sp(A_sparse, B, C, ...)
```

**Requirements:**
- A must have 2:4 sparsity pattern (2 zeros per 4 elements)
- Doubles theoretical compute throughput

### T.fill

Fill buffer with scalar value.

```python
T.fill(buffer, value)
```

### T.clear

Zero-initialize buffer (equivalent to `T.fill(buffer, 0)`).

```python
T.clear(C_local)
```

**Common use:** Initialize accumulators before gemm operations.

### T.Parallel

Parallel loop for element-wise operations.

```python
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)  # ReLU
```

**Automatically maps to thread-level parallelism** and applies vectorization where possible.

### T.reduce_max / T.reduce_sum

Reduction operations along specified dimension.

```python
T.reduce_max(input_tensor, output_tensor, dim=1)
T.reduce_sum(input_tensor, output_tensor, dim=0)
```

**Used in:** Softmax, LayerNorm, attention score normalization

---

## Scheduling & Optimization

### T.Pipelined

Software pipelining for overlapping computation and memory transfers.

```python
for k in T.Pipelined(num_iterations, num_stages=3):
    T.copy(A[...], A_shared)  # Memory transfer
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)  # Computation
```

**Parameters:**
- `num_iterations`: Loop iteration count
- `num_stages`: Pipeline depth (typically 2-4)
  - Stage 2: Double buffering
  - Stage 3: Triple buffering (optimal for many workloads)
  - Stage 4+: Diminishing returns, increases shared memory usage

**How it works:**
1. Compiler hoists `T.copy` operations to earlier iterations
2. Inserts async wait groups to synchronize correctly
3. Creates prologue (warmup), steady state, and epilogue (drain) code

**Shared memory requirement:** `SMEM_usage ≈ base_usage × num_stages`

### T.use_swizzle

Enable block-level swizzle scheduling for better L2 cache utilization.

```python
T.use_swizzle(panel_size=10)
```

**Parameters:**
- `panel_size`: Controls the swizzling pattern (typically 8-16)

**Effect:** Reorders thread block execution to improve L2 cache hit rate.

---

## Helper Functions

### T.ceildiv

Ceiling division for grid dimension calculation.

```python
num_blocks = T.ceildiv(N, block_size)
# Equivalent to: (N + block_size - 1) // block_size
```

### T.min / T.max

Element-wise min/max operations.

```python
result = T.max(a, b)
result = T.min(a, b)
```

### T.exp / T.log / T.sqrt

Mathematical functions (automatically vectorized when possible).

```python
exp_val = T.exp(x)
log_val = T.log(x)
sqrt_val = T.sqrt(x)
```

---

## Memory Layout Shapes

Use `"?"` for dimensions that should be auto-inferred:

```python
B_shared = T.alloc_shared((block_K, "?"), "float16")
```

The compiler uses Z3 solver to infer optimal dimension size based on usage context.

---

## Type System

Supported dtypes:
- Float: `"float16"`, `"float32"`, `"float64"`, `"bfloat16"`
- Integer: `"int8"`, `"int16"`, `"int32"`, `"int64"`
- Unsigned: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`

**Best practices:**
- Use `float16` or `bfloat16` for inputs to leverage Tensor Cores
- Use `float32` for accumulators to avoid overflow
- Use `int8` for quantized models
