# Sketching Algorithms

This page provides detailed information about the sketching algorithm implemented in SketchySVD.jl, based on [TYUC2019](@cite).

## Algorithm Overview

The SketchySVD algorithm processes a data matrix ``A = [a_1, a_2, \ldots, a_n]`` column by column, maintaining compressed representations (sketches) that enable efficient SVD computation.

## Main Algorithm (Algorithm 4.1 from [TYUC2019](@cite))

### Initialization Phase

**Input**: Dimensions ``m, n``, target rank ``r``, sketching dimensions ``k, s, q``

**Initialize**:
1. **Random test matrices**:
   - ``\Xi \in \mathbb{R}^{k \times m}`` (range test matrix)
   - ``\Omega \in \mathbb{R}^{k \times n}`` (corange test matrix)
   - ``\Phi \in \mathbb{R}^{s \times m}`` (core test matrix 1)
   - ``\Psi \in \mathbb{R}^{s \times n}`` (core test matrix 2)
   - ``\Theta \in \mathbb{R}^{q \times m}`` (error test matrix, Gaussian)

2. **Sketch matrices** (initialized to zero):
   - ``X \in \mathbb{R}^{k \times n}`` (corange sketch)
   - ``Y \in \mathbb{R}^{m \times k}`` (range sketch)
   - ``Z \in \mathbb{R}^{s \times s}`` (core sketch)
   - ``E \in \mathbb{R}^{q \times n}`` (error sketch, optional)

### Streaming Phase

**For each column** ``a_j`` (``j = 1, 2, \ldots, n``):

```julia
# Update corange sketch
X[:, j] = Ξ * a_j

# Update range sketch (rank-1 update)
Y = Y + a_j * Ω[j, :]'

# Update core sketch
z = Φ * a_j
Z = Z + z * Ψ[j, :]'

# Update error sketch (optional)
E[:, j] = Θ * a_j
```

**Computational cost per column**:
- Corange: ``O(km)``
- Range: ``O(mk_{\text{nnz}})`` where ``k_{\text{nnz}}`` is number of non-zeros in ``\Omega[j,:]``
- Core: ``O(sm + sk_{\text{nnz}})``
- Error: ``O(qm)``

### Finalization Phase

**After all columns processed**:

1. **Orthonormalize range sketch**:
   ```julia
   Q_Y = qr(Y).Q
   ```

2. **Orthonormalize corange sketch**:
   ```julia
   Q_X = qr(X').Q
   ```

3. **Form intermediate matrix**:
   ```julia
   # Solve (Φ * Q_Y) * C = Z
   temp = (Φ * Q_Y) \ Z
   # Solve C * (Ψ * Q_X)' = temp
   C = temp / (Ψ * Q_X)'
   ```

4. **Compute SVD of core matrix**:
   ```julia
   U_c, Σ_c, V_c = svd(C)
   ```

5. **Extract rank-r approximation**:
   ```julia
   V = Q_Y * U_c[:, 1:r]
   Σ = Σ_c[1:r]
   W = Q_X * V_c[:, 1:r]
   ```

**Result**: ``A \approx V \Sigma W^T``

## Optimized Implementation Details

### Memory Layout

SketchySVD uses column-major storage (Julia's default) for optimal performance:
- Sequential column access in streaming phase
- Efficient BLAS operations for matrix multiplications
- Cache-friendly memory access patterns

### Sparse Matrix Optimizations

For sparse test matrices (e.g., ``\Omega, \Psi``), rank-1 updates are optimized:

```julia
# Instead of: Y = Y + a_j * Ω[j, :]'
# Use sparse iteration:
for (idx, val) in pairs(Ω[j, :])
    Y[:, idx] += val * a_j
end
```

This avoids materializing dense intermediate results.

### BLAS Level 3 Optimizations

For batch updates (`dump!` operation):

```julia
# Optimized batch update using mul!
mul!(X, Ξ, A, ν, η)  # X = η*X + ν*(Ξ*A)
mul!(Y, A, Ω', ν, η)  # Y = η*Y + ν*(A*Ω')
```

This leverages highly optimized BLAS routines.

## Batch vs Incremental Processing

### Incremental Mode (`dump_all=false`)

**Advantages**:
- Constant memory footprint
- True streaming capability
- Can handle data larger than RAM

**Use when**:
- Data arrives sequentially
- Memory is limited
- Real-time processing needed

### Batch Mode (`dump_all=true`)

**Advantages**:
- Faster due to BLAS Level 3 operations
- Better cache utilization
- Fewer function calls

**Use when**:
- All data available at once
- Memory is sufficient
- Maximum speed required

## Algorithm Variants

### Standard Incremental Update

```julia
increment!(sketchy, a_j)  # η=1, ν=1
```

### Exponential Forgetting

```julia
increment!(sketchy, a_j, η, ν)  # η < 1
```

Useful for tracking time-varying subspaces.

### Fixed-Memory Sketching

Using storage budget ``T``:
```julia
sketchy = init_sketchy(m=m, n=n, r=r, T=1000)
```

Automatically determines optimal ``k, s`` for given memory constraint.

## Parallelization Opportunities

While the current implementation is sequential, several operations can be parallelized:

1. **Independent sketch updates**: ``X`` and ``E`` updates are independent
2. **Batch processing**: Multiple columns can be processed in parallel
3. **Finalization**: QR decompositions can use parallel BLAS

## Numerical Stability

The algorithm maintains numerical stability through:

1. **Orthonormalization**: QR decompositions ensure orthonormal bases
2. **Scaling**: Forgetting factors prevent overflow/underflow
3. **Stable linear solves**: Uses backslash operator (LU/QR based)

For ill-conditioned problems, consider:
- Increasing sketching dimensions ``k, s``
- Using SVD instead of QR for orthonormalization
- Regularization techniques

## Error Analysis

**Sources of error**:

1. **Rank truncation**: Approximating rank-``\rho`` matrix with rank-``r`` (``r < \rho``)
2. **Sketching**: Random projection introduces additional error
3. **Numerical**: Finite precision arithmetic

**Total error bound** [TYUC2019](@cite):
```math
\|A - V\Sigma W^T\|_F^2 \leq \sum_{i=r+1}^{\rho} \sigma_i^2(A) + \epsilon_{\text{sketch}} + \epsilon_{\text{num}}
```

The sketching error ``\epsilon_{\text{sketch}}`` decreases with ``k, s``. Note that this a crude generalization and more precise bounds are available in the literature. For the SketchySVD algorithm, refer to [theory.md](theory.md).

## Comparison with Standard SVD

| Aspect | Standard SVD | SketchySVD |
|--------|--------------|------------|
| Memory | ``O(mn)`` | ``O(k(m+n) + s^2)`` |
| Time (batch) | ``O(mn \min(m,n))`` | ``O((k+s)mn)`` |
| Time (stream) | Not applicable | ``O(km + sn)`` per column |
| Accuracy | Exact | Approximate |
| Online | No | Yes |

SketchySVD is advantageous when:
- ``k, s \ll \min(m,n)``
- Streaming/online processing required
- Memory constrained
- Low-rank structure exists
