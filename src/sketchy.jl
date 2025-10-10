"""
    Sketchy

Sketchy is a randomized linear dimension reduction map that is used to reduce 
the dimension of the data matrix in the data-streaming setting. It is based on 
the sketching technique that is used to approximate the Singular Value 
Decomposition of the data matrix. The Sketchy algorithm is used to compute the 
SVD of the data matrix in an incremental fashion. 
For more details see [TYUC2019].

# Fields
- `V::AbstractArray{T}`: Left singular vector matrix.
- `Σ::AbstractArray{T}`: Singular value matrix.
- `W::AbstractArray{T}`: Right singular vector matrix.
- `Υ::AbstractArray{T}`: Test matrix for range (k x m).
- `Ω::AbstractArray{T}`: Test matrix for corange (k x n).
- `Φ::AbstractArray{T}`: Test matrix for core (s x m).
- `Ψ::AbstractArray{T}`: Test matrix for core (s x n).
- `Θ::AbstractArray{T}`: Gaussian test matrix for error (q x m).
- `X::AbstractArray{T}`: Corange sketch (k x n).
- `Y::AbstractArray{T}`: Range sketch (m x k).
- `Z::AbstractArray{T}`: Core sketch (s x s).
- `W::AbstractArray{T}`: Error sketch (q x n).
- `increment::Function`: Single step incremental update function.
- `full_increment!::Function`: All data incremental update function.

# References
- [TYUC2019] J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, “Streaming 
  Low-Rank Matrix Approximation with an Application to Scientific Simulation,” 
  SIAM J. Sci.  Comput., vol. 41, no. 4, pp. A2430-A2463, Jan. 2019, 
  doi: 10.1137/18M1201068.
- [TYUC2019SUP] J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, 
  “Suplementary Materials: Streaming Low-Rank Matrix Approximation with an 
  Application to Scientific Simulation,” SIAM J. Sci. Comput., vol. 41, no. 4, 
  pp. A2430-A2463, Jan. 2019, doi: 10.1137/18M1201068.
"""
mutable struct Sketchy{T<:Number}
    # SVD components
    V::AbstractArray{T}  # left singular vector matrix (m × r)
    Σ::AbstractArray{T}  # singular value matrix (r × r) but stored as a vector
    W::AbstractArray{T}  # right singular vector matrix (n × r)

    # Random matrices
    Ξ::DimRedux                  # test matrix for range (k × m)
    Ω::DimRedux                  # test matrix for corange (k × n)
    Φ::DimRedux                  # test matrix for core (s × m)
    Ψ::DimRedux                  # test matrix for core (s × n)
    Θ::Union{DimRedux, Nothing}  # Gaussian test matrix for error (q × m)

    # Sketches
    X::AbstractArray{T}                  # corange sketch (k × n)
    Y::AbstractArray{T}                  # range sketch (m × k)
    Z::AbstractArray{T}                  # core sketch (s × s)
    E::Union{AbstractArray{T}, Nothing}  # error sketch (q × n)

    # Index counter
    ct::Integer # counter for the number of data streams

    # Dimensions
    dims::Dict{<:Symbol,<:Integer}

    # Constant depending on data type
    α::Integer
    β::Integer

    # Flags 
    verbose::Bool
    ErrorEstimate::Bool
    SpectralDecay::Bool
end

function init_sketchy(;
    m::Integer,                # row size of data matrix
    n::Integer,                # column size of data matrix
    r::Integer,                # selected rank to truncate the SVD
    k::Integer=0,              # range dimension for sketching
    s::Integer=0,              # core dimension for sketching
    q::Integer=0,              # error dimension for sketching
    T::Real=0,                 # storage budget for sketching
    ReduxMap::Symbol=:sparse,  # type of reduction map for sketching
    dtype::String="real",      # data type for the data
    verbose::Bool=false,       # verbosity flag
    ErrorEstimate::Bool=false, # flag to estimate the error
    SpectralDecay::Bool=false  # flag to compute the scree plot
)
    # Initialize the iSVD algorithm
    V = zeros(m, r)
    Σ = zeros(r)
    W = zeros(n, r)

    # α, β parameters (equation 5.1 in [TYUC2019])
    α = dtype == "real" ? 1 : 0
    β = dtype == "real" ? 1 : 2

    # Determine the sketching dimensions
    if iszero(T)  # without using the budget (refer to equation 5.3 in [TYUC2019]) 
        k = iszero(k) ? 4*r + α : k
        s = iszero(s) ? 2*k + α : s
        q = iszero(q) ? m : q
    else  # using the budget T (refer to equations 5.5 and 5.6 in [TYUC2019])
        if iszero(k) || iszero(s)
            k = floor((sqrt((m + n + 4*α)^2 + 16*(T - α^2)) - (m + n + 4*α)) / 8)
            s = floor(sqrt(T - k*(m + n)))
        end
        q = iszero(q) ? m : q
    end
    # k ≤ s ≤ min{m, n}
    s = min(s, min(m, n))
    k = min(k, s)

    theta_flag = ErrorEstimate || SpectralDecay

    # Initialize the random matrices (algorithm 4.1 in [TYUC2019])
    if ReduxMap == :gauss
        Ξ = Gauss(k, m, field=dtype)
        Ω = Gauss(k, n, field=dtype)
        Φ = Gauss(s, m, field=dtype)
        Ψ = Gauss(s, n, field=dtype)
        Θ = theta_flag ? Gauss(q, m, field=dtype) : nothing
    elseif ReduxMap == :ssrft
        Ξ = SSRFT(k, m, field=dtype)
        Ω = SSRFT(k, n, field=dtype)
        Φ = SSRFT(s, m, field=dtype)
        Ψ = SSRFT(s, n, field=dtype)
        Θ = theta_flag ? Gauss(q, m, field=dtype) : nothing # always Gaussian
    elseif ReduxMap == :sparse
        Ξ = Sparse(k, m, field=dtype)
        Ω = Sparse(k, n, field=dtype)
        Φ = Sparse(s, m, field=dtype)
        Ψ = Sparse(s, n, field=dtype)
        Θ = theta_flag ? Gauss(q, m, field=dtype) : nothing # always Gaussian
    else
        error("Invalid reduction map. Available options are " *
              ":gauss, :ssrft, and :sparse.")
    end

    # Initialize the sketches
    X = zeros(k, n)
    Y = zeros(m, k)
    Z = zeros(s, s)
    E = theta_flag ? zeros(q, n) : nothing

    if verbose
        @info "Sketchy parameters: k = $k, s = $s, q = $q"
        @info "Reduction map: $ReduxMap"
        @info "Error estimation: $ErrorEstimate"
        @info "Spectral decay computation: $SpectralDecay"
    end

    return Sketchy(
        V, Σ, W, 
        Ξ, Ω, Φ, Ψ, Θ, 
        X, Y, Z, E, 0,
        Dict(:m=>m, :n=>n, :r=>r, :q=>q, :k=>k, :s=>s), 
        α, β, verbose, ErrorEstimate, SpectralDecay)
end

