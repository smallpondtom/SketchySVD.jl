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