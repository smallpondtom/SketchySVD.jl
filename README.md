# SketchySVD.jl

A Julia package for computing randomized Singular Value Decomposition (SVD) using sketching techniques. This package implements efficient algorithms for approximating the SVD of large matrices by using random projections to reduce dimensionality before performing the decomposition, specifically in the data-streaming setting.

[![Build Status](https://github.com/smallpondtom/SketchySVD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/smallpondtom/SketchySVD.jl/actions/workflows/CI.yml)
[![Contributors](https://img.shields.io/github/contributors/smallpondtom/SketchySVD.jl)](
https://github.com/smallpondtom/SketchySVD.jl/graphs/contributors)
[![License](https://img.shields.io/github/license/smallpondtom/SketchySVD.jl)](
https://github.com/smallpondtom/SketchySVD.jl/blob/main/license.txt)
[![Doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://smallpondtom.github.io/SketchySVD.jl/stable)
[![Doc](https://img.shields.io/badge/docs-dev-green.svg)](https://smallpondtom.github.io/SketchySVD.jl/dev)

## Features

- **Incremental SVD Updates**: Update SVD decompositions efficiently as new data arrives
- **Multiple Sketching Methods**: 
  - Gaussian random projection
  - Sparse random projection
  - Subsampled Randomized Fourier Transform (SSRFT)
- **Memory Efficient**: Processes large matrices without loading them entirely into memory
- **Progress Tracking**: Built-in progress meters for long computations
- **High Performance**: Leverages FFTW and MKL for optimized linear algebra operations

## Installation

```julia
julia> ]
(@v1.11) pkg> add SketchySVD
```

Or in the Julia REPL package mode:

```julia
] add https://github.com/yourusername/SketchySVD.jl
```

## Dependencies

- FFTW.jl
- LinearAlgebra
- MKLSparse
- ProgressMeter
- Random
- SparseArrays
- StatsBase

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, “Streaming Low-Rank Matrix Approximation with an Application to Scientific Simulation,” SIAM J. Sci. Comput., vol. 41, no. 4, pp. A2430–A2463, Jan. 2019, doi: 10.1137/18M1201068.
- N. Halko, P. G. Martinsson, and J. A. Tropp, “Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions,” SIAM Rev., vol. 53, no. 2, pp. 217–288, Jan. 2011, doi: 10.1137/090771806.


## Contact

Tomoki Koike (tkoike@gatech.edu)
