"""
    tail_energy(σ::AbstractVector, r::Int)

Compute the tail energy τ²ᵣ₊₁(A) = Σⱼ₌ᵣ₊₁ σⱼ², where σ contains the singular values of A.

Returns τᵣ₊₁(A) (the square root of the sum).
"""
function tail_energy(σ::AbstractVector{T}, r::Int) where T<:Real
    if r >= length(σ)
        return zero(T)
    end
    return sqrt(sum(σ[j]^2 for j in (r+1):length(σ)))
end

"""
    tail_energy_squared(σ::AbstractVector, r::Int)

Compute τ²ᵣ₊₁(A) = Σⱼ₌ᵣ₊₁ σⱼ².
"""
function tail_energy_squared(σ::AbstractVector{T}, r::Int) where T<:Real
    if r >= length(σ)
        return zero(T)
    end
    return sum(σ[j]^2 for j in (r+1):length(σ))
end

"""
    truncated_approx_error_bound(A::AbstractMatrix, r::Int, k::Int, s::Int; 
                                  field::Symbol=:real)

Compute the error bound from Corollary 5.5 for the rank-r truncated approximation.

# Arguments
- `A`: The input matrix
- `r`: Target rank for the truncated approximation
- `k`: Range sketch size parameter (number of columns in the random test matrix Ω)
- `s`: Core sketch size parameter (total sketch dimension, s > k)
- `field`: Either `:real` (default) or `:complex`

# Returns
The expected spectral norm error bound: E‖A - [[Â]]ᵣ‖₂

# Mathematical formulation
The bound is:
    τᵣ₊₁(A) + 2[((s-α)/(s-k-α)) · min{ρ < k-α} ((k+ρ-α)/(k-ρ-α)) · τ²ᵨ₊₁(A)]^(1/2)

where α = 1 for real matrices, α = 0 for complex matrices.
"""
function truncated_approx_error_bound(
    A::AbstractMatrix, 
    r::Int, k::Int, s::Int;
    field::Symbol=:real
)

    # Validate inputs
    @assert r >= 0 "Target rank r must be non-negative"
    @assert k > r "Range sketch size k must be greater than target rank r"
    @assert s > k "Core sketch size parameter s must be greater than k"
    
    # Field-dependent parameters (equation 5.1)
    α = field == :real ? 1 : 0
    # β = field == :real ? 1 : 2  # Not used in this bound
    
    # Additional constraint from the bound
    @assert s > k + α "Need s > k + α for the bound to be valid"
    @assert k > α "Need k > α for the bound to be valid"
    
    # Compute singular values
    σ = svdvals(A)
    
    return _compute_bound(σ, r, k, s, α)
end

"""
    truncated_approx_error_bound(σ::AbstractVector, r::Int, k::Int, s::Int;
                                  field::Symbol=:real)

Compute the error bound directly from singular values.
"""
function truncated_approx_error_bound(
    σ::AbstractVector{T}, 
    r::Int, k::Int, s::Int;
    field::Symbol=:real
) where T<:Real

    # Validate inputs
    @assert r >= 0 "Target rank r must be non-negative"
    @assert k > r "Range sketch size k must be greater than target rank r"
    @assert s > k "Core sketch size parameter s must be greater than k"
    
    # Field-dependent parameter
    α = field == :real ? 1 : 0
    
    @assert s > k + α "Need s > k + α for the bound to be valid"
    @assert k > α "Need k > α for the bound to be valid"
    
    return _compute_bound(σ, r, k, s, α)
end

"""
Internal function to compute the bound given singular values and parameters.
"""
function _compute_bound(
    σ::AbstractVector{T}, 
    r::Int, k::Int, s::Int, α::Int
) where T<:Real

    # First term: τᵣ₊₁(A)
    τ_r1 = tail_energy(σ, r)
    
    # Coefficient: (s - α) / (s - k - α)
    coef = (s - α) / (s - k - α)
    
    # Minimize over ρ < k - α
    # ρ can range from 0 to k - α - 1 (since ρ < k - α and ρ must give valid indices)
    min_val = Inf
    ρ_max = k - α - 1
    
    for ρ in 0:ρ_max
        # Compute (k + ρ - α) / (k - ρ - α)
        numerator = k + ρ - α
        denominator = k - ρ - α
        
        if denominator <= 0
            continue  # Skip invalid denominators
        end
        
        ratio = numerator / denominator
        
        # Compute τ²ᵨ₊₁(A)
        τ_sq = tail_energy_squared(σ, ρ)
        
        val = ratio * τ_sq
        min_val = min(min_val, val)
    end
    
    # Handle edge case where no valid ρ found
    if isinf(min_val)
        min_val = zero(T)
    end
    
    # Second term: 2 * sqrt(coef * min_val)
    second_term = 2 * sqrt(coef * min_val)
    
    return τ_r1 + second_term
end

"""
    optimal_rho(σ::AbstractVector, k::Int; field::Symbol=:real)

Find the optimal ρ that minimizes the term in the error bound.

Returns (ρ_opt, min_value).
"""
function optimal_rho(σ::AbstractVector{T}, k::Int; field::Symbol=:real) where T<:Real
    α = field == :real ? 1 : 0
    
    ρ_opt = 0
    min_val = Inf
    ρ_max = k - α - 1
    
    for ρ in 0:ρ_max
        numerator = k + ρ - α
        denominator = k - ρ - α
        
        if denominator <= 0
            continue
        end
        
        ratio = numerator / denominator
        τ_sq = tail_energy_squared(σ, ρ)
        val = ratio * τ_sq
        
        if val < min_val
            min_val = val
            ρ_opt = ρ
        end
    end
    
    return ρ_opt, min_val
end


# Example usage and tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("Testing truncated approximation error bound...")
    
    # Create a test matrix with known singular value decay
    m, n = 100, 50
    # Create matrix with exponential singular value decay
    U, _ = qr(randn(m, m))
    V, _ = qr(randn(n, n))
    σ_true = [exp(-0.1 * i) for i in 1:n]
    A = U[:, 1:n] * Diagonal(σ_true) * V'
    
    # Parameters
    r = 5   # Target rank
    k = 10  # Sketch size
    s = 15  # Oversampling
    
    println("\nMatrix size: $m × $n")
    println("Target rank r = $r")
    println("Sketch size k = $k")
    println("Oversampling s = $s")
    
    # Compute bound
    bound = truncated_approx_error_bound(A, r, k, s; field=:real)
    println("\nError bound: $bound")
    
    # Compare with actual tail energy
    σ = svdvals(A)
    τ = tail_energy(σ, r)
    println("Tail energy τᵣ₊₁(A): $τ")
    println("Optimal rank-r approximation error (σᵣ₊₁): $(σ[r+1])")
    
    # Find optimal ρ
    ρ_opt, min_val = optimal_rho(σ, k; field=:real)
    println("\nOptimal ρ = $ρ_opt with minimized value = $min_val")
end