module SketchySVD

# Packages
using FFTW
using LinearAlgebra
using ProgressMeter: Progress, next!
using Random
using SparseArrays
using StatsBase: sample

# Load functions
include("redux/dimredux.jl")
include("redux/gauss.jl")
include("redux/sparse.jl")
include("redux/ssrft.jl")
include("sketchy.jl")
include("increment.jl")
include("finalize.jl")
include("rsvd.jl")

# Conditionally load MKLSparse on non-macOS systems
if !Sys.isapple()
    try
        using MKLSparse
        const HAS_MKL = true
    catch
        const HAS_MKL = false
        @warn "MKLSparse not available, using standard sparse operations"
    end
else
    const HAS_MKL = false
end

export Sketchy 
export init_sketchy, increment!, dump!, finalize!, full_increment!
export rsvd, rsvd_adaptive
export gaussian_rng, uniform_rng, sparse_gaussian_rng
export sparse_rng, rademacher_rng, srft_rng

end # module SketchySVD
