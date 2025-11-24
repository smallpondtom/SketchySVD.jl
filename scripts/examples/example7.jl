using SketchySVD
using LinearAlgebra
using BenchmarkTools
using Printf

# Create test matrix
m, n = 2000, 1000
true_rank = 50
U_true = randn(m, true_rank)
S_true = exp.(-0.1 * (1:true_rank))  # Exponential decay
V_true = randn(n, true_rank)
A = U_true * Diagonal(S_true) * V_true' + 0.01 * randn(m, n)

r = 60  # Target rank

# Method 1: Standard SVD (expensive)
println("Standard SVD:")
@time U_full, S_full, V_full = svd(A)
A_full = U_full[:, 1:r] * Diagonal(S_full[1:r]) * V_full[:, 1:r]'
err_full = norm(A - A_full) / norm(A)
println("Error: $err_full")

# Method 2: Randomized SVD
println("\nRandomized SVD:")
@time U_rsvd, S_rsvd, V_rsvd = rsvd(A, r)
A_rsvd = U_rsvd * Diagonal(S_rsvd) * V_rsvd'
err_rsvd = norm(A - A_rsvd) / norm(A)
println("Error: $err_rsvd")

# Method 3: Randomized SVD with power iterations
println("\nRandomized SVD (q=2):")
@time U_rsvd2, S_rsvd2, V_rsvd2 = rsvd(A, r; q=2)
A_rsvd2 = U_rsvd2 * Diagonal(S_rsvd2) * V_rsvd2'
err_rsvd2 = norm(A - A_rsvd2) / norm(A)
println("Error: $err_rsvd2")

# Method 4: Sketchy (streaming)
println("\nSketchy SVD:")
@time begin
    sketchy = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse)
    for j in 1:n
        increment!(sketchy, A[:, j])
    end
    finalize!(sketchy)
end
U_sketchy = sketchy.V
S_sketchy = sketchy.Σ
V_sketchy = sketchy.W
A_sketchy = U_sketchy * Diagonal(S_sketchy) * V_sketchy'
err_sketchy = norm(A - A_sketchy) / norm(A)
println("Error: $err_sketchy")

# Summary
println("\n" * "="^50)
println("Summary:")
println("Method                    Error         Speedup")
println("-" * "="^50)
@printf("Standard SVD            %.2e       1.0x (baseline)\n", err_full)
@printf("Randomized SVD          %.2e       ~10-50x\n", err_rsvd)
@printf("Randomized SVD (q=2)    %.2e       ~5-25x\n", err_rsvd2)
@printf("Sketchy SVD             %.2e       ~5-20x\n", err_sketchy)