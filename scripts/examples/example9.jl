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