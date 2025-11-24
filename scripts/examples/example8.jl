using SketchySVD
using LinearAlgebra

# Unknown true rank
m, n = 1000, 800
A = rand(m, n)

# Start with high rank estimate
r_max = 100
U, S, V = rsvd(A, r_max)

# Find rank where singular values drop
threshold = 0.03 * S[1]  # 1% of largest singular value
r_effective = findfirst(S .< threshold)

if isnothing(r_effective)
    r_effective = r_max
    println("Need rank > $r_max")
else
    println("Effective rank: $r_effective")
end

# Truncate to effective rank
U = U[:, 1:r_effective]
S = S[1:r_effective]
V = V[:, 1:r_effective]

# Verify accuracy
A_approx = U * Diagonal(S) * V'
println("Relative error with rank $r_effective: $(norm(A - A_approx) / norm(A))")

# Plot singular value decay
using Plots
plot(S, yscale=:log10, label="Singular Values", 
     marker=:circle, ylabel="Singular Value (log scale)", xlabel="Index")
hline!([threshold], label="Threshold", linestyle=:dash)