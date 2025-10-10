module SketchySVD

# Packages
using FFTW
using LinearAlgebra
using MKLSparse
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

export Sketchy 
export init_sketchy, increment!, dump!, finalize!, full_increment! 

end # module SketchySVD
