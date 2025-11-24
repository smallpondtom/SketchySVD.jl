using SketchySVD
using LinearAlgebra
using Printf

# Very large matrix (simulated - not created in memory)
m, n = 100_000, 50_000
r = 100

println("Matrix size: $m × $n ($(m*n/1e9) billion elements)")
println("Would require $(8*m*n/1e9) GB if stored as Float64")

# Function to generate columns on-the-fly
function generate_column(j, m, true_rank=50)
    # Simulate low-rank structure
    U_col = randn(m) * sin(2π * j / n)  # Smooth variation
    return U_col + 0.01 * randn(m)
end

# Use sparse reduction map for efficiency
sketchy = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse)

# Process in streaming fashion
println("\nProcessing columns...")
@time begin
    for j in 1:n
        x = generate_column(j, m)
        increment!(sketchy, x)
        
        if j % 10_000 == 0
            @printf("Progress: %d/%d (%.1f%%)\n", j, n, 100*j/n)
        end
    end
end

println("\nFinalizing...")
@time _, _ = finalize!(sketchy)

println("\nTop 10 singular values:")
S = sketchy.Σ
println(S[1:10])

# Memory usage (approximate)
sketchy_size = sizeof(sketchy.Y) + sizeof(sketchy.Z) + sizeof(sketchy.Ω)
println("\nApproximate memory used: $(sketchy_size/1e9) GB")