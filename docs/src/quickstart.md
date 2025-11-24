# Getting Started

This guide will help you get up and running with SketchySVD.jl quickly.

## Installation

Install SketchySVD from the Julia REPL:

```julia
using Pkg
Pkg.add("SketchySVD")
```

Or in package mode (press `]`):

```julia-repl
pkg> add SketchySVD
```

### Development Version

For the latest features:

```julia
using Pkg
Pkg.add(url="https://github.com/smallpondtom/SketchySVD.jl")
```

## Quick Example

Here's a minimal example to compute a low-rank approximation:

```julia
using SketchySVD
using LinearAlgebra

# Generate random matrix
m, n = 1000, 500
A = randn(m, n)

# Compute rank-20 approximation
r = 20
U, S, V = rsvd(A, r)

# Check approximation quality
A_approx = U * Diagonal(S) * V'
relative_error = norm(A - A_approx) / norm(A)
println("Relative error: ", relative_error)  # Should be ~0.5-0.8 for random matrix
```

## Basic Workflows

### 1. Randomized SVD (Batch Processing)

Use `rsvd()` when you have the entire matrix at once:

```julia
using SketchySVD

# Your matrix
A = randn(5000, 2000)

# Compute top 50 singular triplets
U, S, V = rsvd(A, 50)

# Optional: use oversampling for better accuracy
U, S, V = rsvd(A, 50; p=10)  # p=10 extra dimensions

# Optional: use power iterations for better accuracy
U, S, V = rsvd(A, 50; q=2)   # q=2 power iterations
```

**When to use**: Static matrices, batch data, when full matrix fits in memory.

### 2. Sketchy SVD (Streaming Processing)

Use Sketchy when data arrives incrementally:

```julia
using SketchySVD

# Initialize sketchy structure
m, n = 5000, 2000
r = 50  # Target rank
sketchy = init_sketchy(m=m, n=n, r=r)

# Process data column by column
for j in 1:n
    x = get_data_column(j)  # Your data source
    increment!(sketchy, x)
end

# Finalize to get SVD
finalize!(sketchy)
```

**When to use**: Streaming data, online learning, memory constraints, temporal data.

## Choosing Parameters

### Target Rank ``r``

The most important parameter is the target rank:

```julia
# Low-rank (fast, less accurate)
sketchy = init_sketchy(m=m, n=n, r=10)

# Medium-rank (balanced)
sketchy = init_sketchy(m=m, n=n, r=50)

# High-rank (slower, more accurate)
sketchy = init_sketchy(m=m, n=n, r=200)
```

**Guidelines**:
- Start with ``r = 50`` for most applications
- Use ``r \geq 2 \times`` true rank for good accuracy
- Increase ``r`` if approximation error is too high

### Sketch Dimension ``k``

The sketch dimension controls accuracy vs speed trade-off:

```julia
# Default: k = 2r + 1
sketchy = init_sketchy(m=m, n=n, r=r)

# Higher accuracy: k = 3r
sketchy = init_sketchy(m=m, n=n, r=r, T=3*r)

# Lower memory: k = r + 10
sketchy = init_sketchy(m=m, n=n, r=r, T=r+10)
```

**Guidelines**:
- Default ``k = 2r+1`` works well for most cases
- Use larger ``k`` for noisy data
- Minimum: ``k \geq r``

### Reduction Map Type

Choose based on matrix size and structure:

```julia
# Gaussian (default): best accuracy
sketchy = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss)

# Sparse: large matrices (n > 50,000)
sketchy = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse)

# SSRFT: structured/periodic data
sketchy = init_sketchy(m=m, n=n, r=r, ReduxMap=:ssrft)
```

See [Dimension Reduction Maps](@ref) for detailed comparison.

## Common Use Cases

### Case 1: Large Dense Matrix

```julia
# Matrix too large for standard SVD
m, n = 100_000, 50_000
A = randn(m, n)  # 37 GB of memory!

# Use randomized SVD with sparse map
U, S, V = rsvd(A, 100; redux=:sparse)

# Or streaming if matrix doesn't fit in memory
sketchy = init_sketchy(m=m, n=n, r=100, ReduxMap=:sparse)
for j in 1:n
    # Load column from disk/network
    x = load_column(j)
    increment!(sketchy, x)
end
finalize!(sketchy)
```

### Case 2: Time Series Data

```julia
# Process temporal data with forgetting factor
m, n = 1000, 5000
r = 30
λ = 0.99  # Forget old data slowly

sketchy = init_sketchy(m=m, n=n, r=r)

for t in 1:n
    x = get_sensor_data(t)
    increment!(sketchy, x, 1.0, λ)
    
    # Optional: get current approximation
    if t % 100 == 0
        finalize!(sketchy)
        U = sketchy.V 
        S = sketchy.Σ
        V = sketchy.W
        analyze_current_state(U, S, V)
    end
end
```

### Case 3: Image/Video Processing

```julia
# Process video frames
n_frames = 1000
height, width = 480, 640
m = height * width  # Vectorized frames

sketchy = init_sketchy(m=m, n=n_frames, r=50, ReduxMap=:ssrft)

for t in 1:n_frames
    frame = read_video_frame(t)
    x = vec(frame)  # Flatten to vector
    increment!(sketchy, x)
end

finalize!(sketchy)

# U contains spatial patterns
# V contains temporal patterns
```

### Case 4: Sparse Data

```julia
using SparseArrays

# Sparse matrix (e.g., from sparse sensor data)
m, n = 10_000, 5_000
A = sprand(m, n, 0.01)  # 1% density

# Convert to dense for processing (sketches are dense)
r = 50
sketchy = init_sketchy(m=m, n=n, r=r)

for j in 1:n
    x = Vector(A[:, j])  # Convert sparse column to dense
    increment!(sketchy, x)
end

finalize!(sketchy)
```

## Error Estimation

Enable error estimation to monitor approximation quality:

```julia
# Initialize with error tracking
sketchy = init_sketchy(m=m, n=n, r=r, ErrorEstimate=true)

# Process data
for j in 1:n
    result = increment!(sketchy, x)
    
    # Check estimated error (optional)
    if !isnothing(result.est_err)
        println("Estimated error at column $j: ", result.est_err)
    end
end

# Final error estimate
est_err, _ = finalize(sketchy)
println("Final estimated relative error: ", est_err)
```

## Performance Tips

### 1. Use BLAS Threads

Julia automatically uses multi-threaded BLAS:

```julia
using LinearAlgebra
BLAS.set_num_threads(4)  # Use 4 threads

# Now sketching will use parallel BLAS operations
sketchy = init_sketchy(m=m, n=n, r=r)
```

### 2. Batch Processing

Process multiple columns at once when possible:

```julia
sketchy = init_sketchy(m=m, n=n, r=r)

# Instead of column-by-column
batch_size = 50
for start in 1:batch_size:n
    stop = min(start + batch_size - 1, n)
    B = A[:, start:stop]
    
    for j in start:stop
        increment!(sketchy, B[:, j-start+1])
    end
end
```

### 3. Choose Right Map

For large ``n``, sparse maps are much faster:

```julia
# For n > 50,000
sketchy = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse)
```

### 4. Preallocate Buffers

Avoid repeated allocations:

```julia
sketchy = init_sketchy(m=m, n=n, r=r
x_buffer = zeros(m)

for j in 1:n
    # Reuse buffer instead of allocating
    load_column!(x_buffer, j)
    increment!(sketchy, x_buffer)
end
```

## Troubleshooting

### Error: Dimension Mismatch

```julia
# ERROR: DimensionMismatch
sketchy = init_sketchy(m=1000, n=500, r=50)
x = randn(999)  # Wrong size!
increment!(sketchy, x)
```

**Solution**: Ensure `length(x) == m` where `m` is the row dimension.

### Poor Approximation Quality

```julia
# Relative error too high
U, S, V = rsvd(A, 20)
error = norm(A - U*Diagonal(S)*V') / norm(A)
println(error)  # Too large!
```

**Solutions**:
1. Increase rank `r`
2. Use oversampling: `rsvd(A, r; p=20)`
3. Use power iterations: `rsvd(A, r; q=2)`
4. Increase sketch dimension `T`

### Memory Issues

```julia
# OutOfMemoryError with Gaussian map
sketchy = init_sketchy(m=100_000, n=500_000, r=100)  # Too large!
```

**Solutions**:
1. Use sparse map: `ReduxMap=:sparse`
2. Use SSRFT: `ReduxMap=:ssrft`
3. Reduce rank `r`
4. Process in chunks

### Slow Performance

```julia
# Processing is too slow
@time for j in 1:n
    increment!(sketchy, x, j)
end
```

**Solutions**:
1. Use sparse map for large `n`
2. Enable BLAS threading
3. Reduce sketch dimension `T`
4. Use batch processing

## Need Help?

- **Issues**: Report bugs on [GitHub Issues](https://github.com/username/SketchySVD.jl/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/username/SketchySVD.jl/discussions)
- **Documentation**: Read the full [documentation](https://username.github.io/SketchySVD.jl/)
