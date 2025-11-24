using SketchySVD
using LinearAlgebra
using Random

Random.seed!(42)

# Create a low-rank matrix with noise
m, n, true_rank = 1000, 500, 20
U_true = randn(m, true_rank)
U_true = qr(U_true).Q  # Orthonormalize
V_true = randn(n, true_rank)
V_true = qr(V_true).Q  # Orthonormalize
S_true = sort(rand(true_rank), rev=true) .* 100
A = U_true * Diagonal(S_true) * V_true' + 0.1 * randn(m, n)

# Compute approximation
r = 30  # Oversample beyond true rank
U, S, V = rsvd(A, r)

# Evaluate quality
A_approx = U * Diagonal(S) * V'
relative_error = norm(A - A_approx) / norm(A)
println("Relative approximation error: $relative_error")

# Compare singular values
using Plots
plot(S_true, label="True", marker=:circle, ylabel="Singular Value", xlabel="Index")
plot!(S[1:true_rank], label="Estimated", marker=:square, legend=:topright)