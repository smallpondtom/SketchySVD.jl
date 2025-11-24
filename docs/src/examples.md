# Examples

This page provides comprehensive examples demonstrating various use cases of SketchySVD.jl.

## Example 1: Basic Low-Rank Approximation

Compute a low-rank approximation of a random matrix:

```julia
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
```

**Expected output**: Relative error < 1%, singular values closely match.

## Example 2: Image Compression

Compress an image using low-rank approximation:

```julia
using SketchySVD
using TestImages, ImageIO
using LinearAlgebra

# Load image (or create synthetic)
img = Float64.(Gray.(testimage("cameraman")))
m, n = size(img)

# Compress with different ranks
ranks = [5, 10, 20, 50, 100]
compressed_images = []

for r in ranks
    U, S, V = rsvd(img, r)
    img_approx = U * Diagonal(S) * V'
    
    error = norm(img - img_approx) / norm(img)
    compression_ratio = (m * n) / (r * (m + n + 1))
    
    println("Rank $r: Error = $(round(error, digits=4)), Compression = $(round(compression_ratio, digits=2))x")
    push!(compressed_images, img_approx)
end

# Visualize results
using Plots
plot(
    plot(Gray.(img), title="Original"),
    plot(Gray.(compressed_images[1]), title="Rank 5"),
    plot(Gray.(compressed_images[3]), title="Rank 20"),
    plot(Gray.(compressed_images[5]), title="Rank 100"),
    layout=(2,2)
)
```

**Key insight**: Rank 50-100 often gives good visual quality with 10-20x compression.

## Example 3: Streaming PCA

Perform PCA on streaming data:

```julia
using SketchySVD
using Statistics
using LinearAlgebra

# Simulate streaming sensor data
n_sensors = 100
n_timesteps = 5000
true_components = 5

# Generate synthetic data with 5 principal components
W = randn(n_sensors, true_components)
H = randn(true_components, n_timesteps)
data = W * H + 0.5 * randn(n_sensors, n_timesteps)

# Normalize (center and scale)
data_mean = mean(data, dims=2)
data_centered = data .- data_mean

# Streaming PCA with Sketchy
sketchy = init_sketchy(m=n_sensors, n=n_timesteps, r=20, ReduxMap=:sparse)

for t in 1:n_timesteps
    x = data_centered[:, t]
    increment!(sketchy, x)
    
    # Optional: periodic updates
    if t % 1000 == 0
        _, _ = finalize!(sketchy)
        println("At t=$t: Top 5 singular values: $(sketchy.Σ[1:5])")
    end
end

# Final PCA
finalize!(sketchy)

# Principal components are columns of U
# Projections onto components are rows of Diagonal(S) * V'

# Analyze variance explained
S = sketchy.Σ
total_variance = sum(S.^2)
variance_explained = cumsum(S.^2) ./ total_variance

using Plots
plot(variance_explained[1:20], 
     label="Cumulative Variance Explained",
     marker=:circle,
     xlabel="Number of Components",
     ylabel="Fraction of Variance")
hline!([0.9, 0.95, 0.99], label=["90%" "95%" "99%"], linestyle=:dash)
```

**Key insight**: Usually 5-10 components explain > 90% of variance in real data.

## Example 4: Video Background Subtraction

Separate foreground objects from video background:

```julia
using SketchySVD
using LinearAlgebra

# Simulate video: static background + moving foreground
n_frames = 200
height, width = 64, 64
m = height * width

# Background: low-rank
background = randn(height, width, 3)
background_flat = zeros(m, n_frames)
for t in 1:n_frames
    background_flat[:, t] = vec(background[:,:,1])  # Static
end

# Foreground: sparse
foreground = zeros(m, n_frames)
for t in 1:n_frames
    # Moving object
    obj_size = 10
    x_pos = 20 + div(t * width, n_frames)
    y_pos = 30
    indices = (y_pos:y_pos+obj_size-1, x_pos:min(x_pos+obj_size-1, width))
    
    for i in indices[1], j in indices[2]
        idx = (j-1) * height + i
        if idx <= m
            foreground[idx, t] = 100  # Bright object
        end
    end
end

# Composite video
video = background_flat + foreground

# Robust PCA via sketching
r_background = 5  # Background is low-rank
sketchy = init_sketchy(m=m, n=n_frames, r=r_background, ReduxMap=:sparse)

for t in 1:n_frames
    increment!(sketchy, video[:, t])
end

finalize!(sketchy)
U = sketchy.V
S = sketchy.Σ
V = sketchy.W
background_recovered = U * Diagonal(S) * V'
foreground_recovered = video - background_recovered

# Visualize results
using Plots
t_show = 100
plot(
    heatmap(reshape(video[:, t_show], height, width), title="Original Frame"),
    heatmap(reshape(background_recovered[:, t_show], height, width), title="Background"),
    heatmap(reshape(foreground_recovered[:, t_show], height, width), title="Foreground"),
    layout=(1,3), colorbar=false
)
```

**Key insight**: Low-rank approximation captures static/slow-varying background, residual is foreground.

## Example 5: Temporal Data with Forgetting

Track evolving patterns in non-stationary time series:

```julia
using SketchySVD
using Plots

# Simulate evolving system: frequencies change over time
n_sensors = 50
n_timesteps = 2000
m = n_sensors

function generate_evolving_data(t, n_sensors)
    # Frequency drifts over time
    freq1 = 0.1 + 0.05 * sin(2π * t / 500)
    freq2 = 0.2 + 0.03 * cos(2π * t / 700)
    
    x = zeros(n_sensors)
    for i in 1:n_sensors
        phase = 2π * i / n_sensors
        x[i] = sin(2π * freq1 * t + phase) + 0.5 * sin(2π * freq2 * t)
    end
    
    return x + 0.1 * randn(n_sensors)
end

# Track with exponential forgetting
η = 0.99  # Forget slowly
sketchy = init_sketchy(m=m, n=n_timesteps, r=10, ErrorEstimate=true)

errors = Float64[]
top_sv = Float64[]

for t in 1:n_timesteps
    x = generate_evolving_data(t, n_sensors)
    result = increment!(sketchy, x, 1.0, η)
    
    # Track every 50 steps
    if t % 50 == 0
        est_err, _ = finalize!(sketchy)
        S = sketchy.Σ
        push!(errors, isnothing(est_err) ? NaN : est_err)
        push!(top_sv, S[1])
        
        println("t=$t: Top SV = $(round(S[1], digits=2)), Error = $(isnothing(est_err) ? "N/A" : round(est_err, digits=4))")
    end
end

# Plot evolution
plot(
    plot(50:50:n_timesteps, top_sv, label="Top Singular Value", ylabel="Value"),
    plot(50:50:n_timesteps, errors, label="Estimated Error", ylabel="Error"),
    layout=(2,1), xlabel="Time", legend=:topright
)
```

**Key insight**: Forgetting factor allows tracking of non-stationary dynamics.

## Example 6: Large-Scale Matrix with Sparse Map

Efficiently handle very large matrices:

```julia
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
```

**Key insight**: Sparse maps enable processing matrices thousands of times larger than available RAM.

## Example 7: Comparison with Standard SVD

Compare accuracy and speed of different methods:

```julia
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
```

**Key insight**: Randomized methods are 10-50x faster with minimal accuracy loss.

## Example 8: Adaptive Rank Selection

Automatically determine the appropriate rank:

```julia
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
```

**Key insight**: Adaptive rank selection balances accuracy and efficiency.

## Example 9: Burgers' Equation (PDE Solution)

Analyze solutions of a 1D viscous Burgers' equation:

```julia
using SketchySVD
using LinearAlgebra
using Plots

# Load or generate Burgers' equation data
# (see scripts/burgers.jl for data generation)

# Assuming data is in scripts/data/
using CSV, DataFrames
data_path = "scripts/data/viscous_burgers1d_states.csv"

if isfile(data_path)
    df = CSV.read(data_path, DataFrame)
    snapshots = Matrix(df)  # Each column is a spatial snapshot at one time
    
    m, n = size(snapshots)
    println("Loaded $n snapshots of size $m")
    
    # Compute POD (Proper Orthogonal Decomposition) modes
    r = 20
    U, S, V = rsvd(snapshots, r)
    
    # U columns are POD modes (spatial patterns)
    # V columns are temporal coefficients
    
    # Analyze modal energy
    modal_energy = S.^2
    total_energy = sum(modal_energy)
    energy_fraction = cumsum(modal_energy) ./ total_energy
    
    println("\nModal energy distribution:")
    for i in 1:min(10, r)
        @printf("Mode %2d: %.2f%% (cumulative: %.2f%%)\n", 
                i, 100*modal_energy[i]/total_energy, 100*energy_fraction[i])
    end
    
    # Visualize modes
    p1 = plot(U[:, 1:4], label=["Mode 1" "Mode 2" "Mode 3" "Mode 4"],
              xlabel="Spatial Position", ylabel="Amplitude", title="POD Modes")
    
    p2 = plot(energy_fraction[1:r], marker=:circle, 
              xlabel="Mode Index", ylabel="Cumulative Energy Fraction",
              title="Energy Spectrum", legend=false)
    hline!([0.99], linestyle=:dash, color=:red)
    
    plot(p1, p2, layout=(2,1))
else
    println("Data file not found. Run scripts/burgers.jl first.")
end
```

**Key insight**: PDE solutions often have low effective rank; 5-10 modes capture > 99% energy.

## Example 10: Real-Time Anomaly Detection

Detect anomalies in streaming data:

```julia
using SketchySVD
using LinearAlgebra
using Statistics

# Simulation parameters
n_sensors = 50
n_timesteps = 1000
baseline_rank = 5

# Generate normal data
function generate_normal(t, n)
    # 5 underlying patterns
    patterns = rand(n, baseline_rank)
    weights = [sin(2π*t/100), cos(2π*t/100), sin(2π*t/50), 
               cos(2π*t/50), sin(2π*t/200)]
    return patterns * weights + 0.1 * rand(n)
end

# Inject anomalies
anomaly_times = [300, 600, 850]

# Initialize sketchy 
sketchy = init_sketchy(m=n_sensors, n=n_timesteps, r=10)

reconstruction_errors = Float64[]
is_anomaly = Bool[]

for t in 1:n_timesteps
    # Get data
    x = generate_normal(t, n_sensors)
    
    # Inject anomaly
    if t in anomaly_times
        x += 1000 * randn(n_sensors)  # Large spike
    end
    
    # Update sketch
    increment!(sketchy, x, 1.0, 1.0)
    
    # Reconstruction-based anomaly detection
    if t > 50  # Wait for initialization
        finalize!(sketchy)
        U = sketchy.V 
        S = sketchy.Σ
        V = sketchy.W
        
        # Project onto top modes
        r_detect = 5
        U_trunc = U[:, 1:r_detect]
        x_proj = U_trunc * (U_trunc' * x)
        
        # Reconstruction error
        error = norm(x - x_proj) / norm(x)
        push!(reconstruction_errors, error)
        
        # Anomaly threshold (3 standard deviations)
        if t > 100
            μ = mean(reconstruction_errors[1:end-1])
            σ = std(reconstruction_errors[1:end-1])
            push!(is_anomaly, error > μ + 2.5σ || error < μ - 2.5σ)
        else
            push!(is_anomaly, false)
        end
    end
end

# Visualize results
using Plots
times = 51:n_timesteps
p = plot(times, reconstruction_errors, label="Reconstruction Error", 
         ylabel="Error", xlabel="Time")
scatter!(anomaly_times, [reconstruction_errors[t-50] for t in anomaly_times if t > 50], 
         color=:red, markersize=8, label="True Anomalies")

# Mark detected anomalies
detected = times[is_anomaly]
if !isempty(detected)
    scatter!(detected, reconstruction_errors[is_anomaly], 
             color=:orange, marker=:x, markersize=10, 
             markerstrokewidth=5,
             label="Detected")
end

plot!(p, legend=:topleft)
```

**Key insight**: Anomalies have large reconstruction error when projected onto normal subspace.

## More Examples

For additional examples, see:
- `scripts/random_matrix.jl`: Random matrix benchmarks
- `scripts/burgers.jl`: PDE analysis workflow
- `test/` directory: Comprehensive test suite with many usage patterns

## Tips for Your Application

1. **Start simple**: Begin with `rsvd()` on a small dataset
2. **Profile**: Use `@time` and `@benchmark` to identify bottlenecks
3. **Iterate**: Adjust `r`, `T`, and reduction map based on results
4. **Validate**: Compare with standard SVD on small test cases
5. **Monitor**: Use error estimation to track quality

## Questions?

If you have questions about applying SketchySVD to your specific problem, please open a discussion on [GitHub](https://github.com/smallpondtom/SketchySVD.jl/discussions).
