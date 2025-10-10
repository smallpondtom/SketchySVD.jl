"""
Test SketchySVD algorithm with 1D viscous Burgers' equation data.
"""

#=================#
## Load Packages #3
#=================#
using CairoMakie
using CSV
using LinearAlgebra
using DataFrames
using Revise
using Test
using SketchySVD

#=============#
## Constants ##
#=============#
TOL = 1e-10

#=================#
## Generate Data ##
#=================#
# Load data 
X = CSV.read("scripts/data/viscous_burgers1d_states.csv", DataFrame) |> Matrix

# Dims
m, n = size(X)
r = 12

#===================#
## Compute the SVD ##
#===================#
sketchy = SketchySVD.init_sketchy(
    m=m, n=n, r=r, ReduxMap=:Sparse, ErrorEstimate=true, SpectralDecay=true,
    verbose=true
)
ees, scree = SketchySVD.full_increment!(sketchy, X, terminate=true)

#================#
## Standard SVD ##
#================#
V, Σ, W = svd(X)
V = V[:, 1:r]  # Left singular vectors
Σ = Σ[1:r]     # Singular values
W = W[:, 1:r]  # Right singular vectors

#=======#
## Test
#=======#
# Singular values
error = norm(Σ - sketchy.Σ) / norm(Σ)
println("Relative error in singular values: ", error)
@test error < TOL

## Left singular vectors (subspace angle)
left_subspace_angle_error = norm(V*V' - sketchy.V*sketchy.V') / sqrt(2)
println("Subspace angle error in left singular vectors: ", left_subspace_angle_error)
@test left_subspace_angle_error < TOL

## Right singular vectors (subspace angle)
right_subspace_angle_error = norm(W*W' - sketchy.W*sketchy.W') / sqrt(2)
println("Subspace angle error in right singular vectors: ", right_subspace_angle_error)
@test right_subspace_angle_error < TOL

#========#
## Plot ##
#========#
# Singular values
fig1 = Figure()
ax = Axis(fig1[1,1], title="Singular Values", xlabel="Index", ylabel="Value", yscale=log10)
scatterlines!(ax, 1:r, Σ, color=:black, linewidth=3, label="SVD")
scatterlines!(ax, 1:r, sketchy.Σ, color=:red, linewidth=2, linestyle=:dash, label="iSVD")
axislegend(ax, labelsize=20, position=:rt)
display(fig1)
