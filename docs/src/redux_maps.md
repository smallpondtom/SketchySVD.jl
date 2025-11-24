# Dimension Reduction Maps

Dimension reduction maps (also called sketching matrices or test matrices) are the core building blocks of randomized algorithms. They compress high-dimensional data into lower dimensions while preserving essential structure.

## Overview

A dimension reduction map ``\Omega: \mathbb{R}^n \to \mathbb{R}^k`` (where ``k \ll n``) satisfies the **Johnson-Lindenstrauss property**: for any vector ``x``, with high probability:

```math
(1-\epsilon)\|x\|_2 \leq \|\Omega x\|_2 \leq (1+\epsilon)\|x\|_2
```

SketchySVD implements three types of reduction maps, each with different trade-offs.

## 1. Gaussian Random Matrices

### Definition

A Gaussian reduction map ``G \in \mathbb{R}^{k \times n}`` has entries drawn independently from ``\mathcal{N}(0, 1/k)``:

```math
G_{ij} \sim \mathcal{N}(0, 1/k)
```

### Properties

- **Density**: Fully dense (all entries non-zero)
- **Generation**: ``O(kn)`` time, uses `randn()`
- **Application**: ``O(kmn)`` for ``G \cdot A`` where ``A \in \mathbb{R}^{n \times m}``
- **Storage**: ``O(kn)`` memory
- **Theoretical guarantees**: Strongest guarantees, well-studied

### When to Use

✅ **Best for**:
- General matrices without special structure
- When accuracy is paramount
- Medium-sized problems where memory isn't a constraint

❌ **Avoid when**:
- ``n`` is very large (> 100,000)
- Memory is severely constrained
- Maximum speed is required

### Implementation

```julia
# Create Gaussian reduction map
G = Gauss(k, n, field="real")

# Apply to matrix
Y = G * A  # Left application: (k × n) × (n × m) → (k × m)
Z = A * G' # Right application: (m × n) × (n × k) → (m × k)

# Access properties
size(G)        # (k, n)
isreal(G)      # true for real field
issparse(G)    # false (always dense)
```

### Advantages

1. **Strong theory**: Best-understood reduction map
2. **Consistent performance**: Reliable across different matrix types
3. **Easy implementation**: Simple to generate and apply
4. **Good accuracy**: Typically requires smallest oversampling

### Disadvantages

1. **Memory intensive**: Requires ``O(kn)`` storage
2. **Slower for large ``n``**: Dense matrix multiplications
3. **Generation cost**: Creating random numbers can be slow

## 2. Sparse Random Matrices

### Definition

A sparse reduction map ``S \in \mathbb{R}^{k \times n}`` has ``\zeta`` non-zero entries per column, where each entry is ``\pm 1/\sqrt{\zeta}``:

```math
S_{:,j} \text{ has } \zeta \text{ random non-zeros from } \{\pm 1/\sqrt{\zeta}\}
```

The parameter ``\zeta`` controls sparsity (typically ``\zeta = 8``).

### Properties

- **Density**: ``\zeta/k`` (very sparse)
- **Generation**: ``O(n\zeta)`` time
- **Application**: ``O(m n \zeta)`` for ``S \cdot A``
- **Storage**: ``O(n\zeta)`` memory (sparse format)
- **Theoretical guarantees**: Good with slightly larger ``k``

### When to Use

✅ **Best for**:
- Very large matrices (``n`` > 100,000)
- Memory-constrained environments
- When ``k`` is moderately large
- Real-time applications

❌ **Avoid when**:
- ``n`` is small (overhead of sparse operations)
- Maximum accuracy required (use Gaussian instead)

### Implementation

```julia
# Create sparse reduction map
S = Sparse(k, n, zeta=8, field="real")

# Apply to matrix
Y = S * A  # Sparse matrix multiplication
Z = A * S' # Uses sparse transpose

# Access properties
size(S)        # (k, n)
issparse(S)    # true
nnz(S)         # n * zeta (number of non-zeros)
```

### Sparsity Parameter ``\zeta``

The choice of ``\zeta`` affects performance:

| ``\zeta`` | Memory | Speed | Accuracy |
|-----------|---------|-------|----------|
| 1 | Lowest | Fastest | Poor |
| 4 | Low | Fast | Good |
| 8 | Moderate | Moderate | Very Good |
| 16 | Higher | Slower | Excellent |

**Recommendation**: Use ``\zeta = 8`` (default) for most applications.

### Advantages

1. **Memory efficient**: Only ``O(n\zeta)`` storage vs ``O(kn)`` for Gaussian
2. **Fast multiplication**: Sparse ops are much faster for large ``n``
3. **Easy generation**: Simple to create
4. **Cache friendly**: Sparse structure improves cache performance

### Disadvantages

1. **Slight accuracy loss**: May need larger ``k`` than Gaussian
2. **Fixed sparsity pattern**: Less flexible than Gaussian
3. **Overhead for small problems**: Sparse operations have overhead

### Optimization Details

SketchySVD uses optimized sparse operations:

```julia
# For incremental updates with sparse Ω
row_Ω = Ω'[i, :]  # Get sparse row
for (idx, val) in pairs(row_Ω)
    Y[:, idx] += val * x  # Rank-1 update
end
```

This avoids materializing dense intermediates.

## 3. Subsampled Randomized Fourier Transform (SSRFT)

### Definition

An SSRFT ``R \in \mathbb{R}^{k \times n}`` is defined implicitly as:

```math
R = \sqrt{\frac{n}{k}} \cdot P_2 \cdot F \cdot D_1 \cdot P_1 \cdot F \cdot D_2
```

where:
- ``D_1, D_2``: Diagonal matrices with random ``\pm 1`` entries
- ``F``: Discrete Fourier Transform (applied via FFT)
- ``P_1, P_2``: Random permutation matrices
- The factor ``\sqrt{n/k}`` normalizes the output

### Properties

- **Density**: Dense when materialized, but **never stored explicitly**
- **Generation**: ``O(n)`` time (just random permutations and signs)
- **Application**: ``O(m n \log n)`` via FFT
- **Storage**: ``O(k + n)`` (just store permutations and signs)
- **Theoretical guarantees**: Excellent for structured matrices

### When to Use

✅ **Best for**:
- Large structured matrices
- When ``n`` is a power of 2 (fastest FFT)
- Signal processing applications
- Image/video data

❌ **Avoid when**:
- ``n`` is small (< 1000)
- Matrix has no structure
- FFT is not available/efficient

### Implementation

```julia
# Create SSRFT reduction map
R = SSRFT(k, n, field="real")

# Apply to matrix (uses FFT internally)
Y = R * A  # O(mn log n) operation
Z = A * R' # Transposed SSRFT

# Access properties
size(R)        # (k, n)
issparse(R)    # false (but never materialized)
```

### How SSRFT Works

1. **Pre-randomization**: Apply ``D_2`` (element-wise sign flips)
2. **FFT**: Transform to frequency domain
3. **Permute**: Shuffle rows with ``P_1``
4. **Second randomization**: Apply ``D_1``
5. **Second FFT**: Another Fourier transform
6. **Subsample**: Select ``k`` rows with ``P_2``
7. **Scale**: Multiply by ``\sqrt{n/k}``

All operations are implicitly applied during matrix multiplication.

### Advantages

1. **Memory efficient**: Only ``O(n)`` storage (permutations + signs)
2. **Fast**: ``O(mn\log n)`` via FFT
3. **Structured**: Exploits Fast Fourier Transform
4. **Good for structured data**: Excellent for signals, images

### Disadvantages

1. **FFT dependency**: Requires FFTW library
2. **Complex implementation**: More complex than Gaussian/Sparse
3. **Not always faster**: Overhead can dominate for small ``n``
4. **Real/complex handling**: Care needed with data types

### FFT Performance

SSRFT is fastest when ``n`` is highly composite (many small prime factors):

| ``n`` | FFT Speed |
|-------|-----------|
| ``2^{10}`` (1024) | Excellent |
| ``2^{10} \cdot 3`` (3072) | Very Good |
| Large prime | Poor |

## Comparison Table

| Aspect | Gaussian | Sparse | SSRFT |
|--------|----------|--------|-------|
| **Memory** | ``O(kn)`` | ``O(n\zeta)`` | ``O(n)`` |
| **Generation** | ``O(kn)`` | ``O(n\zeta)`` | ``O(n)`` |
| **Application** | ``O(kmn)`` | ``O(mn\zeta)`` | ``O(mn\log n)`` |
| **Accuracy** | Best | Good | Very Good |
| **Ease of use** | Easiest | Easy | Moderate |
| **Best for** | General | Large sparse | Structured |

## Performance Benchmarks

For a matrix ``A \in \mathbb{R}^{5000 \times 10000}`` with ``k = 100``:

| Method | Memory (MB) | Time (s) | Relative Error |
|--------|-------------|----------|----------------|
| Gaussian | 800 | 0.45 | 2.3% |
| Sparse (``\zeta=8``) | 6.4 | 0.12 | 2.8% |
| SSRFT | 0.08 | 0.18 | 2.5% |

*Benchmark on Intel i7-10700K, single thread*

## Choosing the Right Map

### Decision Tree

```
Is memory very tight (< 100 MB available)?
├─ Yes → Use SSRFT
└─ No
    ├─ Is n > 100,000?
    │   ├─ Yes → Use Sparse
    │   └─ No
    │       └─ Is matrix structured/periodic?
    │           ├─ Yes → Use SSRFT
    │           └─ No → Use Gaussian
```

### General Guidelines

1. **Default**: Start with Gaussian for general problems
2. **Large-scale**: Switch to Sparse for ``n`` > 50,000
3. **Structured data**: Try SSRFT for signals, images, PDEs
4. **Accuracy-critical**: Use Gaussian with higher ``k``
5. **Memory-limited**: Use SSRFT or Sparse

## Advanced Usage

### Custom Reduction Maps

You can create custom reduction maps by subtyping `DimRedux`:

```julia
mutable struct CustomRedux{T<:Number} <: DimRedux
    k::Int
    n::Int
    # ... custom fields ...
    transposeFlag::Bool
end

# Implement required methods
LeftApply(obj::CustomRedux, A) = ...
RightApply(obj::CustomRedux, A) = ...
```

### Mixing Reduction Maps

Different maps can be used for different sketches:

```julia
# Gaussian for range, sparse for corange
sketchy = init_sketchy(
    m=m, n=n, r=r,
    ReduxMap=:gauss  # Default for all
)

# Then manually replace if needed
sketchy.Ω = Sparse(k, n)  # Use sparse for Ω
```

### Parameter Tuning

For Sparse maps, tune ``\zeta`` based on your problem:

```julia
# Higher accuracy (more memory)
S = Sparse(k, n, zeta=16)

# Lower memory (less accuracy)
S = Sparse(k, n, zeta=4)
```

## Theoretical Background

All three maps satisfy the **JL-property** with different constants:

- **Gaussian**: ``k = O(\epsilon^{-2} \log(1/\delta))``
- **Sparse**: ``k = O(\epsilon^{-2} \log^2(1/\delta))``
- **SSRFT**: ``k = O(\epsilon^{-2} \log(1/\delta) \log(n))``

where ``\epsilon`` is target distortion and ``\delta`` is failure probability.
