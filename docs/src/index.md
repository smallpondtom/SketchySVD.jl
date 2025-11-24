# SketchySVD.jl

*Fast, memory-efficient streaming SVD for Julia*

[![Build Status](https://github.com/smallpondtom/SketchySVD.jl/workflows/CI/badge.svg?branch=main)](https://github.com/username/SketchySVD.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://username.github.io/SketchySVD.jl/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SketchySVD.jl provides state-of-the-art algorithms for computing low-rank approximations of large matrices using **randomized sketching** techniques. Based on the SIAM paper by Tropp et al. (2019), this package enables:

- **Streaming SVD**: Process data that doesn't fit in memory
- **Randomized SVD**: 10-50x faster than standard SVD with minimal accuracy loss
- **Multiple sketching methods**: Gaussian, Sparse, and SSRFT reduction maps
- **Online learning**: Track evolving low-rank structure in non-stationary data

## Why SketchySVD?

### 🚀 Fast
Randomized algorithms are **orders of magnitude faster** than standard SVD for large matrices:
```julia
# Standard SVD: ~30 seconds for 10000×5000 matrix
U, S, V = svd(A)

# Randomized SVD: ~2 seconds with comparable accuracy
U, S, V = rsvd(A, 50)
```

### 💾 Memory Efficient
Process matrices **larger than available RAM** by streaming:
```julia
# Matrix: 100GB, RAM: 16GB → No problem!
sketchy = init_sketchy(100_000, 50_000, 100; ReduxMap=:sparse)
for j in 1:n
    x = load_column_from_disk(j)
    increment!(sketchy, x, j)
end
U, S, V = finalize(sketchy)
```

### 🎯 Accurate
Theoretical guarantees ensure controlled approximation error:
```julia
# Error estimation built-in
sketchy = init_sketchy(m, n, r; estimate_error=true)
# ... process data ...
U, S, V, est_err = finalize(sketchy)
println("Estimated relative error: $est_err")
```

### ⚡ Flexible
Choose the right algorithm for your problem:
- **Gaussian**: Best accuracy, general matrices
- **Sparse**: Fastest for large-scale problems
- **SSRFT**: Optimal for structured/periodic data

## Quick Start

### Installation

Install from the Julia package registry:

```julia
using Pkg
Pkg.add("SketchySVD")
```

Or install the development version:

```julia
Pkg.add(url="https://github.com/username/SketchySVD.jl")
```

### Your First Computation

Compute a rank-50 approximation in three lines:

```julia
using SketchySVD, LinearAlgebra

A = randn(5000, 2000)  # Your matrix
U, S, V = rsvd(A, 50)  # Randomized SVD
println("Approximation error: ", norm(A - U*Diagonal(S)*V') / norm(A))
```

### Streaming Large Data

Process data that doesn't fit in memory:

```julia
sketchy = init_sketchy(m, n, r)  # Initialize
for j in 1:n
    x = load_column(j)            # Get data
    increment!(sketchy, x, j)     # Update sketch
end
U, S, V = finalize(sketchy)      # Extract SVD
```

## Documentation Overview

### For New Users

1. **[Getting Started](quickstart.md)**: Installation, basic usage, parameter selection
2. **[Examples](examples.md)**: Complete examples for common applications
3. **[API Reference](api.md)**: Function documentation

### For Advanced Users

4. **[Mathematical Theory](theory.md)**: Sketching framework and theoretical guarantees
5. **[Sketching Algorithms](sketching.md)**: Detailed algorithm walkthrough
6. **[Dimension Reduction Maps](redux_maps.md)**: Comparison of Gaussian, Sparse, SSRFT
7. **[Randomized SVD](rsvd.md)**: Batch processing algorithms

## Key Algorithms

### Randomized SVD (Batch)

For matrices that fit in memory, use `rsvd()`:

```julia
U, S, V = rsvd(A, r)           # Basic usage
U, S, V = rsvd(A, r; p=10)     # With oversampling
U, S, V = rsvd(A, r; q=2)      # With power iterations
U, S, V = rsvd(A, r; redux=:sparse)  # Sparse sketching
```

**Use when**: Full matrix available, batch processing, quick prototyping.

### Sketchy SVD (Streaming)

For streaming data, use `init_sketchy()` + `increment!()` + `finalize()`:

```julia
sketchy = init_sketchy(m, n, r; ReduxMap=:sparse, γ=0.99)
for j in 1:n
    increment!(sketchy, x_j, j)
end
U, S, V = finalize(sketchy)
```

**Use when**: Data arrives incrementally, memory constrained, online learning.

## Usage Examples

### Example 1: Random Matrix Decomposition

See `scripts/random_matrix.jl` for a complete example of decomposing a random matrix and comparing results with the exact SVD.

```julia
using SketchySVD
using LinearAlgebra
using Test

# Generate test matrix
m, n, r = 1000, 500, 50
A = randn(m, n)

# Initialize and compute
sketchy = init_sketchy(m=m, n=n, r=r; method=:ssrft)
full_increment!(sketchy, A)
finalize!(sketchy)

# Compare with exact SVD
U_exact, Σ_exact, V_exact = svd(A)
error = norm(Σ_exact[1:r] - sketchy.Σ) / norm(Σ_exact[1:r])
println("Relative error in singular values: ", error)
```

### Example 2: Viscous Burgers Equation

See `scripts/burgers.jl` for an application to fluid dynamics data.

```julia
using SketchySVD
using CSV, DataFrames

# Load data
data = CSV.read("scripts/data/viscous_burgers1d_states.csv", DataFrame)
A = Matrix(data)
m, n = size(A)

# Compute sketchy SVD
r = 10
sketchy = init_sketchy(m=m, n=n, r=r; method=:gauss)
full_increment!(sketchy, A)
finalize!(sketchy)

# Analyze dominant modes
println("Top 5 singular values: ", sketchy.Σ[1:5])
```

## Sketching Methods

### Gaussian Random Projection (`:gauss`)
Projects data using Gaussian random matrices. Provides good accuracy with moderate computational cost.

### Sparse Random Projection (`:sparse`)
Uses sparse random matrices for projection. Faster than Gaussian for very large matrices with similar accuracy.

### Subsampled Randomized Fourier Transform (`:ssrft`)
Leverages FFT for fast random projections. Most efficient for structured matrices.

## Performance Tips

1. **Choose appropriate rank**: Set `r` to capture desired accuracy while minimizing computation
2. **Adjust oversampling**: Increase for better accuracy, decrease for speed
3. **Select optimal method**: 
   - Use `:ssrft` for structured/large matrices
   - Use `:sparse` for very large data
   - Use `:gauss` for general matrices
4. **Batch processing**: Use incremental updates for streaming data

## Testing

Run the test suite:

```julia
using Pkg
Pkg.test("SketchySVD")
```

Or run individual examples:

```julia
include("scripts/random_matrix.jl")
include("scripts/burgers.jl")
```

## Project Structure

```
SketchySVD.jl/
├── src/
│   ├── SketchySVD.jl      # Main module
│   ├── sketchy.jl         # Core data structures
│   ├── increment.jl       # Incremental update functions
│   ├── finalize.jl        # Finalization and SVD computation
│   └── redux/
│       ├── dimredux.jl    # Dimension reduction interface
│       ├── gauss.jl       # Gaussian sketching
│       ├── sparse.jl      # Sparse sketching
│       └── ssrft.jl       # SSRFT sketching
├── scripts/
│   ├── random_matrix.jl   # Example: random matrix SVD
│   ├── burgers.jl         # Example: Burgers equation
│   └── data/
│       └── viscous_burgers1d_states.csv
└── Project.toml
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/smallpondtom/SketchySVD.jl/blob/main/LICENSE) file for details.

## References

- J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, “Streaming Low-Rank Matrix Approximation with an Application to Scientific Simulation,” SIAM J. Sci. Comput., vol. 41, no. 4, pp. A2430–A2463, Jan. 2019, doi: 10.1137/18M1201068.
- N. Halko, P. G. Martinsson, and J. A. Tropp, “Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions,” SIAM Rev., vol. 53, no. 2, pp. 217–288, Jan. 2011, doi: 10.1137/090771806.


## Contact

Tomoki Koike (tkoike@gatech.edu)
