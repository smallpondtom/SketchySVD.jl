"""
    rsvd(A::AbstractArray, k::Int; p::Int=5, q::Int=0, 
         rng::Union{DimRedux,Function}=randn, 
         transpose_trick::Bool=true)

Compute a rank-k randomized SVD approximation of matrix A using power iterations
and orthonormalization for improved accuracy.

# Arguments
- `A::AbstractArray`: Input matrix (m x n)
- `k::Int`: Target rank for the approximation

# Keyword Arguments
- `p::Int=5`: Oversampling parameter (samples k+p random vectors)
- `q::Int=0`: Number of power iterations (0 means no power iterations)
- `rng::Union{DimRedux,Function}=Gauss`: Random matrix generator function with 
  signature (nrows, ncols)
- `transpose_trick::Bool=true`: Use transpose trick when m >> n for efficiency

# Returns
- `V`: Left singular vectors (m x k)
- `Σ`: Singular values (vector of length k)
- `W`: Right singular vectors (n x k)

# Examples
```julia
# Basic usage with default Gaussian random matrix
V, Σ, W = rsvd(A, 50)

# With power iterations for better accuracy
V, Σ, W = rsvd(A, 50, q=2)

# With custom random matrix (e.g., sparse random)
using SparseArrays
sparse_rng(m, n) = sprandn(m, n, 0.1)  # 10% density
V, Σ, W = rsvd(A, 50, rng=sparse_rng)

# With Rademacher (±1) random matrix (faster generation)
V, Σ, W = rsvd(A, 50, rng=rademacher_rng)

# With SRFT (Subsampled Randomized Fourier Transform) - very efficient
V, Σ, W = rsvd(A, 50, rng=srft_rng, q=1)
```
"""
function rsvd(A::AbstractArray, k::Int; 
              p::Int=5, 
              q::Int=0, 
              rng::Union{DimRedux,Function}=randn,
              transpose_trick::Bool=true)
    
    m, n = size(A)
    
    # Validate inputs
    k > 0 || throw(ArgumentError("k must be positive"))
    p >= 0 || throw(ArgumentError("p must be non-negative"))
    q >= 0 || throw(ArgumentError("q must be non-negative"))
    k + p <= min(m, n) || throw(ArgumentError("k+p must be ≤ min(m,n)"))
    
    # Use transpose trick if A is tall and thin (m >> n)
    if transpose_trick && m > 5*n && k + p < n
        return rsvd_transpose(A, k, p, q, rng)
    end
    
    # Standard randomized SVD
    return rsvd_standard(A, k, p, q, rng)
end

"""
    rsvd_standard(A, k, p, q, rng)

Standard randomized SVD implementation (for m ≤ n or when transpose trick is disabled).
"""
function rsvd_standard(A::AbstractArray, k::Int, p::Int, q::Int, 
                       rng::Union{DimRedux,Function})
    m, n = size(A)
    ℓ = k + p  # Total number of random vectors
    
    # Step 1: Generate random test matrix G (n × ℓ)
    G = rng(n, ℓ)
    
    # Step 2: Form Y = A*G and orthonormalize to get Q
    Y = A * G
    Q = Matrix(qr!(Y).Q)  # Efficient QR factorization
    
    # Step 3: Power iterations (optional, for q > 0)
    if q > 0
        # Pre-allocate workspace for power iterations
        W = similar(Q, n, ℓ)
        for j in 1:q
            # W = orth(A'*Q)
            mul!(W, A', Q)  # In-place multiplication
            W_qr = qr!(W)
            W = Matrix(W_qr.Q)
            
            # Q = orth(A*W)
            mul!(Q, A, W)  # Reuse Q storage
            Q_qr = qr!(Q)
            Q = Matrix(Q_qr.Q)
        end
    end
    
    # Step 4: B = Q'*A (small matrix: ℓ × n)
    B = Q' * A
    
    # Step 5: Compute SVD of small matrix B
    F = svd!(B)  # In-place SVD
    
    # Step 6: Compute U = Q*Û (but only keep first k columns)
    U = Q * F.U[:, 1:k]
    S = F.S[1:k]
    V = F.V[:, 1:k]
    
    return U, S, V
end

"""
    rsvd_transpose(A, k, p, q, rng)

Transpose trick for tall-thin matrices (m >> n).
More efficient when the matrix has many more rows than columns.
"""
function rsvd_transpose(A::AbstractArray, k::Int, p::Int, q::Int, 
                        rng::Union{DimRedux,Function})
    m, n = size(A)
    ℓ = k + p
    
    # Generate random test matrix (m × ℓ) - note the dimension swap
    G = rng(m, ℓ)
    
    # Form Y = A'*G and orthonormalize
    Y = A' * G
    Q = Matrix(qr!(Y).Q)
    
    # Power iterations with transposed operations
    if q > 0
        W = similar(Q, m, ℓ)
        for j in 1:q
            # W = orth(A*Q)
            mul!(W, A, Q)
            W_qr = qr!(W)
            W = Matrix(W_qr.Q)
            
            # Q = orth(A'*W)
            mul!(Q, A', W)
            Q_qr = qr!(Q)
            Q = Matrix(Q_qr.Q)
        end
    end
    
    # B = Q'*A' = (A*Q)' (small matrix: ℓ × m)
    B = (A * Q)' |> Matrix
    
    # SVD of B
    F = svd!(B)
    
    # For transpose trick: U and V are swapped
    V = Q * F.U[:, 1:k]  # Right singular vectors
    S = F.S[1:k]
    U = F.V[:, 1:k]      # Left singular vectors
    
    return U, S, V
end

"""
    rsvd_adaptive(A, k; tol=1e-10, max_iter=10, p=5, q=0, rng=randn)

Adaptive randomized SVD that automatically determines the rank based on 
singular value decay.

# Arguments
- `A`: Input matrix
- `k`: Initial target rank estimate

# Keyword Arguments
- `tol::Float64=1e-10`: Tolerance for singular value cutoff
- `max_iter::Int=10`: Maximum number of adaptive iterations
- `p::Int=5`: Oversampling parameter
- `q::Int=0`: Number of power iterations
- `rng::Function=randn`: Random matrix generator

# Returns
- `U, S, V`: SVD factors where rank is automatically determined
"""
function rsvd_adaptive(A::AbstractArray, k::Int; 
                       tol::Float64=1e-10, 
                       max_iter::Int=10,
                       p::Int=5, 
                       q::Int=0, 
                       rng::Union{DimRedux,Function}=randn)
    
    U, S, V = rsvd(A, k, p=p, q=q, rng=rng)
    
    # Find cutoff based on tolerance
    cutoff_idx = findfirst(s -> s < tol * S[1], S)
    
    if isnothing(cutoff_idx)
        return U, S, V
    else
        idx = cutoff_idx - 1
        return U[:, 1:idx], S[1:idx], V[:, 1:idx]
    end
end

# ============================================================================
# Utility functions for different random matrix types
# ============================================================================

"""
    gaussian_rng(m, n)

Standard Gaussian random matrix generator (default).
"""
gaussian_rng(m::Int, n::Int) = randn(m, n)

"""
    uniform_rng(m, n)

Uniform random matrix generator on [-1, 1].
"""
uniform_rng(m::Int, n::Int) = 2 .* rand(m, n) .- 1

"""
    sparse_gaussian_rng(density)

Create a sparse Gaussian random matrix generator with specified density.
"""
function sparse_gaussian_rng(density::Float64)
    function generator(m::Int, n::Int)
        return sprandn(m, n, density)
    end
    return generator
end

"""
    rademacher_rng(m, n)

Rademacher random matrix (entries are ±1 with equal probability).
More efficient than Gaussian for some applications.
"""
rademacher_rng(m::Int, n::Int) = Float64.(rand([-1, 1], m, n))

"""
    srft_rng(m, n)

Subsampled Randomized Fourier Transform (SRFT) matrix generator.
More efficient than Gaussian random matrices, especially for large problems.

The SRFT is an implicit structured random matrix of the form:
    SRFT = √(m/k) · R · F · D

where:
- D is a diagonal matrix with random ±1 entries (size m × m)
- F is the DFT matrix (applied via FFT)
- R is a random row sampling matrix (selects n rows from m)

The resulting matrix has size (n, m).

This implementation doesn't form the matrix explicitly but returns a lazy
operator that applies the SRFT efficiently.
"""
function srft_rng(m::Int, n::Int)
    # We want to create a matrix of size (m, n)
    # This means: input vectors have length n, output vectors have length m
    
    # Random signs for diagonal matrix D (size n)
    d = rand([-1.0, 1.0], n)
    
    # Random row indices to sample (select m rows from n after FFT)
    if m >= n
        indices = collect(1:n)
        scale = sqrt(n / m)
    else
        indices = sort(sample(1:n, m, replace=false))
        scale = sqrt(n / m)
    end
    
    # Return a lazy matrix-like object with size (m, n)
    return SRFTMatrix(m, n, d, indices, scale)
end

"""
    SRFTMatrix

Lazy representation of a Subsampled Randomized Fourier Transform matrix.
The matrix has size (m_out, n_in): takes n_in-dimensional vectors and 
produces m_out-dimensional outputs.
"""
struct SRFTMatrix{T<:Real} <: AbstractMatrix{T}
    m_out::Int  # Output dimension (number of rows in matrix)
    n_in::Int   # Input dimension (number of columns in matrix)
    d::Vector{T}  # Diagonal signs (length n_in)
    indices::Vector{Int}  # Row indices to sample (length ≤ n_in)
    scale::T
end

# Matrix size is (m_out, n_in)
Base.size(S::SRFTMatrix) = (S.m_out, S.n_in)
Base.eltype(::SRFTMatrix{T}) where T = T

# Implement getindex for element-wise access (computed on demand)
function Base.getindex(S::SRFTMatrix, i::Int, j::Int)
    @boundscheck checkbounds(S, i, j)
    
    # Create unit vector e_j
    e_j = zeros(S.n_in)
    e_j[j] = 1.0
    
    # Apply SRFT and extract i-th element
    result = S * e_j
    return real(result[i])
end

# Matrix-vector multiplication: SRFT * v where v has length n_in
function Base.:*(S::SRFTMatrix, v::AbstractVector)
    @assert length(v) == S.n_in "Vector length $(length(v)) must match matrix columns $(S.n_in)"
    
    # Step 1: Apply diagonal scaling D
    dv = S.d .* v
    
    # Step 2: Apply FFT (produces vector of length n_in)
    fft_result = fft(dv)
    
    # Step 3: Subsample rows
    sampled = fft_result[S.indices]
    
    # Step 4: Scale and pad if necessary
    result = S.scale .* sampled
    
    # If m_out > length(indices), pad with zeros
    if S.m_out > length(S.indices)
        padded = zeros(ComplexF64, S.m_out)
        padded[1:length(result)] = result
        result = padded
    end
    
    # Convert to real if imaginary part is negligible
    if maximum(abs.(imag(result))) < 1e-10
        return real(result)
    end
    return result
end

# Matrix-matrix multiplication: SRFT * M where M has n_in rows
function Base.:*(S::SRFTMatrix, M::AbstractMatrix)
    @assert size(M, 1) == S.n_in "Matrix rows $(size(M, 1)) must match SRFT columns $(S.n_in)"
    
    n_cols = size(M, 2)
    
    # Process first column to determine if result should be real or complex
    first_col = S * M[:, 1]
    is_real = eltype(M) <: Real && maximum(abs.(imag(first_col))) < 1e-10
    
    if is_real
        result = zeros(Float64, S.m_out, n_cols)
        result[:, 1] = real(first_col)
        
        for j in 2:n_cols
            col_result = S * M[:, j]
            result[:, j] = real(col_result)
        end
    else
        result = zeros(ComplexF64, S.m_out, n_cols)
        result[:, 1] = first_col
        
        for j in 2:n_cols
            result[:, j] = S * M[:, j]
        end
    end
    
    return result
end

# For cases where we need explicit matrix (not recommended for large matrices)
function Base.Matrix(S::SRFTMatrix)
    result = zeros(ComplexF64, S.m_out, S.n_in)
    
    for j in 1:S.n_in
        e_j = zeros(S.n_in)
        e_j[j] = 1.0
        result[:, j] = S * e_j
    end
    
    if maximum(abs.(imag(result))) < 1e-10
        return real(result)
    end
    return result
end