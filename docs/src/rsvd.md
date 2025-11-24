# Randomized SVD

The Randomized SVD (rSVD) is a fast algorithm for computing approximate singular value decompositions using randomized sampling techniques [HMT2011](@cite).

## Basic Algorithm

Given a matrix ``A \in \mathbb{R}^{m \times n}`` and target rank ``k``, randomized SVD computes an approximation ``A \approx U \Sigma V^T`` using the following steps:

### Algorithm Steps

**1. Random Sampling**

Generate a random test matrix ``G \in \mathbb{R}^{n \times \ell}`` where ``\ell = k + p`` (``p`` is oversampling parameter):

```math
G = \text{randn}(n, \ell)
```

**2. Range Approximation**

Form the sample matrix and orthonormalize:

```math
Y = AG, \quad Q = \text{orth}(Y)
```

where ``Q \in \mathbb{R}^{m \times \ell}`` has orthonormal columns that approximate the range of ``A``.

**3. Dimensionality Reduction**

Project ``A`` onto the low-dimensional subspace:

```math
B = Q^T A \in \mathbb{R}^{\ell \times n}
```

**4. SVD of Small Matrix**

Compute the SVD of the small matrix ``B``:

```math
B = \tilde{U} \tilde{\Sigma} \tilde{V}^T
```

**5. Recover Approximate SVD**

Construct the approximate SVD of ``A``:

```math
U = Q\tilde{U}_{:,1:k}, \quad \Sigma = \tilde{\Sigma}_{1:k,1:k}, \quad V = \tilde{V}_{:,1:k}
```

## Power Iterations

For improved accuracy, especially when the singular values decay slowly, power iterations can be applied:

```julia
function rsvd_with_power(A, k; p=5, q=2)
    m, n = size(A)
    ℓ = k + p
    
    # Random matrix
    G = randn(n, ℓ)
    
    # Form sample and orthonormalize
    Y = A * G
    Q = Matrix(qr!(Y).Q)
    
    # Power iterations
    for j in 1:q
        # Z = orth(A' * Q)
        Z = A' * Q
        Z = Matrix(qr!(Z).Q)
        
        # Q = orth(A * Z)
        Q = A * Z
        Q = Matrix(qr!(Q).Q)
    end
    
    # Compute SVD
    B = Q' * A
    F = svd!(B)
    
    # Extract rank-k approximation
    U = Q * F.U[:, 1:k]
    S = F.S[1:k]
    V = F.V[:, 1:k]
    
    return U, S, V
end
```

**Effect**: Each power iteration amplifies the dominant singular values by ``\sigma_i^2``, improving the approximation.

**Recommendation**: Use ``q=1`` or ``q=2`` power iterations for most applications.

## Transpose Trick

For tall-thin matrices (``m \gg n``), it's more efficient to work with ``A^T``:

```julia
function rsvd_transpose(A, k; p=5, q=0)
    m, n = size(A)
    @assert m > 5*n "Use standard rSVD for non-tall matrices"
    
    ℓ = k + p
    
    # Work with transpose
    G = randn(m, ℓ)
    Y = A' * G
    Q = Matrix(qr!(Y).Q)
    
    # Power iterations (if q > 0)
    for j in 1:q
        Z = A * Q
        Z = Matrix(qr!(Z).Q)
        Q = A' * Z
        Q = Matrix(qr!(Q).Q)
    end
    
    # Compute SVD
    B = (A * Q)'
    F = svd!(B)
    
    # U and V are swapped
    V = Q * F.U[:, 1:k]
    S = F.S[1:k]
    U = F.V[:, 1:k]
    
    return U, S, V
end
```

**Speedup**: Reduces complexity from ``O(mn^2)`` to ``O(m n \ell)`` when ``\ell \ll n``.

## Adaptive Rank Selection

The `rsvd_adaptive` function automatically determines the effective rank based on singular value decay:

```julia
function rsvd_adaptive(A, k_max; tol=1e-10, p=5, q=0)
    # Compute rSVD with k_max
    U, S, V = rsvd(A, k_max, p=p, q=q)
    
    # Find cutoff where S[i] < tol * S[1]
    cutoff = findfirst(s -> s < tol * S[1], S)
    
    if isnothing(cutoff)
        return U, S, V
    else
        k_actual = cutoff - 1
        return U[:, 1:k_actual], S[1:k_actual], V[:, 1:k_actual]
    end
end
```

**Use case**: When the effective rank is unknown but you have a tolerance threshold.

## Random Matrix Types

SketchySVD supports multiple random matrix types for different trade-offs:

### 1. Gaussian (Standard)

```julia
G = randn(n, ℓ)
```

**Properties**:
- Dense matrix
- Strong theoretical guarantees
- Standard choice for general matrices

**Complexity**: ``O(mn\ell)`` per multiplication

### 2. Rademacher (±1 entries)

```julia
G = Float64.(rand([-1, 1], n, ℓ))
```

**Properties**:
- Faster generation than Gaussian
- Similar performance to Gaussian
- Smaller storage (can use Int8)

**Complexity**: Same as Gaussian, but faster generation

### 3. Sparse Gaussian

```julia
using SparseArrays
G = sprandn(n, ℓ, density)
```

**Properties**:
- Much faster multiplication for large ``n``
- Requires higher oversampling ``p``
- Good for very large matrices

**Complexity**: ``O(mn\cdot \text{density} \cdot \ell)``

**Recommendation**: Use ``density \approx \min(1, \frac{\log \ell}{n})``

### 4. Subsampled Randomized Fourier Transform (SRFT)

```julia
G = srft_matrix(n, ℓ)
```

**Properties**:
- Implicitly formed (not stored)
- Very fast via FFT
- Excellent for structured matrices

**Complexity**: ``O(mn \log n)`` using FFT

**Structure**:
```math
G = \sqrt{\frac{n}{\ell}} \cdot R \cdot F \cdot D
```

where:
- ``D``: diagonal with random ±1 entries
- ``F``: DFT matrix (applied via FFT)
- ``R``: row sampling (selects ``\ell`` rows)

## Error Bounds

For a rank-``\rho`` matrix approximated by rank-``k`` rSVD:

**Expected error** [HMT2011](@cite):
```math
\mathbb{E}[\|A - U\Sigma V^T\|_F] \leq \left(1 + \frac{k}{p-1}\right)^{1/2} \left(\sum_{i=k+1}^{\rho} \sigma_i^2\right)^{1/2}
```

**With ``q`` power iterations** [M2019](@cite):
```math
\mathbb{E}[\|A - U\Sigma V^T\|_2] \leq \left[ (1 + \sqrt{\frac{k}{p-1}})\sigma_{k+1}^{2q+1} + \frac{e\sqrt{k+p}}{p}\left( \sum_{j=k+1}^{\min(m,n)}\sigma_{j}^{2(2q+1)} \right)^{1/2} \right]^{1/(2q+1)}
```

**Key insights**:
- Error decreases with oversampling ``p``
- Power iterations drastically reduce error
- Error bounded by tail singular values

## Parameter Selection Guidelines

### Oversampling Parameter ``p``

| Value | Use Case |
|-------|----------|
| ``p=5`` | Default, good balance |
| ``p=10`` | High accuracy required |
| ``p=0\text{-}2`` | Speed priority, good spectrum |

### Power Iterations ``q``

| Value | Use Case |
|-------|----------|
| ``q=0`` | Fast approximation, well-conditioned |
| ``q=1`` | Standard accuracy |
| ``q=2`` | High accuracy |
| ``q\geq 3`` | Ill-conditioned matrices |

### Transpose Trick

**Use when**: ``m > 5n`` and ``k + p < n``

**Speedup**: Approximately ``\frac{m}{5n}`` times faster

## Computational Complexity

| Operation | Standard SVD | rSVD (no power) | rSVD (q power) |
|-----------|--------------|-----------------|----------------|
| Time | ``O(mn\min(m,n))`` | ``O(mn\ell)`` | ``O(qmn\ell)`` |
| Space | ``O(mn)`` | ``O(m\ell + n\ell)`` | Same |

Where ``\ell = k + p \ll \min(m,n)``.

**Example**: For ``m=10000, n=5000, k=50, p=10``:
- Standard SVD: ``\sim 10^{12}`` operations
- rSVD: ``\sim 3 \times 10^9`` operations
- **Speedup**: ~300×

## Numerical Stability

### Orthonormalization

Use QR factorization for numerical stability:

```julia
Q = Matrix(qr!(Y).Q)  # Stable orthonormalization
```

**Avoid**: Gram-Schmidt without reorthogonalization (unstable)

### Precision Considerations

For very ill-conditioned matrices:
1. Use more power iterations
2. Increase oversampling
3. Consider double-double precision
4. Apply regularization

## Implementation in SketchySVD.jl

The package provides three main functions:

### `rsvd` - Standard randomized SVD

```julia
U, S, V = rsvd(A, k; p=5, q=0, rng=randn, transpose_trick=true)
```

### `rsvd_adaptive` - Adaptive rank selection

```julia
U, S, V = rsvd_adaptive(A, k_max; tol=1e-10, p=5, q=0, rng=randn)
```

### Random Matrix Generators

```julia
# Gaussian (default)
U, S, V = rsvd(A, k, rng=gaussian_rng)

# Rademacher
U, S, V = rsvd(A, k, rng=rademacher_rng)

# Sparse Gaussian
U, S, V = rsvd(A, k, rng=sparse_gaussian_rng(0.1))

# SRFT
U, S, V = rsvd(A, k, rng=srft_rng)

# Sparse redux
U, S, V = rsvd(A, k, rng=sparse_rng)
```

## Comparison: Batch rSVD vs Streaming SketchySVD

| Aspect | rSVD | SketchySVD |
|--------|------|------------|
| **Input** | Full matrix required | Streaming columns |
| **Memory** | ``O(mn + m\ell + n\ell)`` | ``O(k(m+n) + s^2)`` |
| **Speed** | Fast (BLAS Level 3) | Moderate (streaming) |
| **Online** | No | Yes |
| **Forgetting** | No | Yes (``\eta, \nu``) |
| **Use case** | Batch processing | Streaming/online |

**When to use rSVD**: All data available, maximum speed, one-time computation

**When to use SketchySVD**: Streaming data, memory constrained, time-varying subspaces
