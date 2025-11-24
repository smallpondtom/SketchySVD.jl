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