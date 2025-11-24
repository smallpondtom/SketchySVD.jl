"""
SSRFT dimension reduction map. See Algorithm SM3.2 in Supplementary Material of [TYUC2019].
"""

# Define the SSRFT type as a mutable struct
mutable struct SSRFT{T<:Number} <: DimRedux
    k::Int
    n::Int
    field::String
    Pi1::SparseMatrixCSC{T,Int}
    Pi2::SparseMatrixCSC{T,Int}
    coords::Vector{Int}
    transposeFlag::Bool
end

# Constructor
function SSRFT(k::Int, n::Int; field::String="real")
    # if k > n
    #     error("k should be less than or equal to n.")
    # end
    field = lowercase(field)
    transposeFlag = false
    coords = randperm(n)[1:k]
    if field == "real"
        I1 = randperm(n)
        J1 = collect(1:n)
        V1 = sign.(randn(n))
        Pi1 = sparse(I1, J1, V1, n, n)

        I2 = randperm(n)
        J2 = collect(1:n)
        V2 = sign.(randn(n))
        Pi2 = sparse(I2, J2, V2, n, n)

        return SSRFT{Float64}(k, n, field, Pi1, Pi2, coords, transposeFlag)
    elseif field == "complex"
        I1 = randperm(n)
        J1 = collect(1:n)
        V1 = sign.(randn(n) .+ im * randn(n))
        Pi1 = sparse(I1, J1, V1, n, n)

        I2 = randperm(n)
        J2 = collect(1:n)
        V2 = sign.(randn(n) .+ im * randn(n))
        Pi2 = sparse(I2, J2, V2, n, n)

        return SSRFT{ComplexF64}(k, n, field, Pi1, Pi2, coords, transposeFlag)
    else
        error("Input 'field' should be 'real' or 'complex'.")
    end
end

# Redraw another SSRFT random matrix
function redraw(obj::SSRFT{T}, k, n) where {T<:Number}
    return SSRFT(k, n, field=obj.field)
end

function redraw(obj::SSRFT{T}) where {T<:Number}
    return SSRFT(obj.k, obj.n, field=obj.field)
end

# LeftApply method
# function LeftApply(obj::SSRFT{T}, M::AbstractArray{T2}) where {T<:Number, T2<:Number}
#     B = obj.Pi1 * M
#     if isreal(obj)
#         B = FFTW.dct(B, 2)
#     else
#         B = FFTW.fft(B, dims=1)
#     end
#     B = obj.Pi2 * B
#     if isreal(obj)
#         B = FFTW.dct(B, 2)
#     else
#         B = FFTW.fft(B, dims=1)
#     end
#     B = B[obj.coords, :]
#     return B
# end

# function LeftApply!(obj::SSRFT{T}, M::AbstractArray{T2}, B::AbstractArray{T}) where {T<:Number, T2<:Number}
#     # Preallocate temporary array
#     temp1 = similar(B, size(obj.Pi1, 1), size(M, 2))
    
#     # Compute temp1 = Pi1 * M
#     mul!(temp1, obj.Pi1, M)
    
#     # In-place DCT or FFT
#     if isreal(obj)
#         FFTW.dct!(temp1, 2)  # DCT-II in-place
#     else
#         temp1 .= FFTW.fft(temp1, dims=1)
#     end
    
#     # Compute temp1 = Pi2 * temp1
#     mul!(temp1, obj.Pi2, temp1)
    
#     # Second in-place DCT or FFT
#     if isreal(obj)
#         FFTW.dct!(temp1, 2)
#     else
#         temp1 .= FFTW.fft(temp1, dims=1)
#     end
    
#     # Extract rows corresponding to obj.coords
#     B .= temp1[obj.coords, :]
#     return B
# end

function LeftApply(obj::SSRFT{T}, M::AbstractMatrix{T2}) where {T<:Number, T2<:Number}
    # M is 2D: we assume size(M) == (n, something)
    # 1) Multiply
    B = obj.Pi1 * M  # still 2D
    # 2) DCT or FFT across dimension 1 (rows)
    if isreal(obj)
        # region=2 => Type-II DCT
        # dims=1   => transform along 1st dimension
        B = FFTW.dct(B, 1)
    else
        B = FFTW.fft(B, 1)
    end
    # 3) Multiply
    B = obj.Pi2 * B
    # 4) DCT or FFT across dimension 1 again
    if isreal(obj)
        B = FFTW.dct(B, 1)
    else
        B = FFTW.fft(B, 1)
    end
    # 5) Finally pick out the rows given by obj.coords
    B = B[obj.coords, :]  # result is (k, something)
    return B
end

# Overload for vector inputs: reshape to (n,1), call the main method, reshape back
function LeftApply(obj::SSRFT{T}, v::AbstractVector{T2}) where {T<:Number, T2<:Number}
    # We want a (n,1) matrix
    M = reshape(v, :, 1)                    # shape = (n,1)
    B = LeftApply(obj, M)                   # shape = (k,1)
    return dropdims(B, dims=2)              # convert back to a length-k vector
end

function LeftApply!(obj::SSRFT{T}, M::AbstractMatrix{T2}, B::AbstractMatrix{T}) where {T<:Number, T2<:Number}
    # M: (n, p)
    # B: (k, p) final result

    n = obj.n
    k = obj.k
    @assert size(M,1) == n "M must have n rows"
    @assert size(B,1) == k "B must have k rows"
    @assert size(M,2) == size(B,2) "Second dimension mismatch"

    # Temporary workspace: same shape as M except we only need n rows for Pi1
    temp1 = similar(M, n, size(M,2))

    # 1) temp1 = Pi1 * M
    mul!(temp1, obj.Pi1, M)    # shape (n, p)

    # 2) In-place DCT or FFT over dimension 1
    if isreal(obj)
        FFTW.dct!(temp1, 1)   # Type-II DCT in-place over rows
    else
        temp1 .= FFTW.fft(temp1, 1)
    end

    # 3) temp1 = Pi2 * temp1
    mul!(temp1, obj.Pi2, temp1) # shape (n, p)

    # 4) Second DCT/FFT in-place
    if isreal(obj)
        FFTW.dct!(temp1, 1)
    else
        temp1 .= FFTW.fft(temp1, 1)
    end

    # 5) B = temp1[obj.coords, :]
    B .= temp1[obj.coords, :]
    return B
end

function LeftApply!(obj::SSRFT{T}, v::AbstractVector{T2}, b::AbstractVector{T}) where {T<:Number, T2<:Number}
    n = obj.n
    k = obj.k
    @assert length(v) == n "Input vector must have length n"
    @assert length(b) == k "Output vector must have length k"

    # 1) Reshape v to (n,1)
    M = reshape(v, n, 1)

    # 2) We need a temporary (n,1) and Bfinal is (k,1)
    temp1 = similar(M, n, 1)
    Bfinal = similar(b, k, 1)

    # Re-use the 2D method for the heavy lifting
    LeftApply!(obj, M, Bfinal)

    # Bfinal is (k,1); copy that into b (k,)
    @inbounds @simd for i in 1:k
        b[i] = Bfinal[i,1]
    end
    return b
end

# RightApply method
# function RightApply(obj::SSRFT{T}, M::AbstractArray{T2}) where {T<:Number, T2<:Number}
#     k, n = obj.k, obj.n
#     B = zeros(T, k, n)
#     B[:, obj.coords] = M
#     if isreal(obj)
#         B = FFTW.dct(B', 3)'  # DCT-III
#     else
#         B = n * FFTW.ifft(B', dims=1)'
#     end
#     B = B * obj.Pi2
#     if isreal(obj)
#         B = FFTW.dct(B', 3)'  # DCT-III
#     else
#         B = n * FFTW.ifft(B', dims=1)'
#     end
#     B = B * obj.Pi1
#     return B
# end

# function RightApply!(obj::SSRFT{T}, M::AbstractArray{T2}, B::AbstractArray{T}) where {T<:Number, T2<:Number}
#     k, n = obj.k, obj.n
    
#     # Preallocate temporary array
#     temp1 = zeros(T, k, n)
#     temp1[:, obj.coords] = M
    
#     # In-place DCT-III or IFFT
#     if isreal(obj)
#         FFTW.dct!(temp1', 3)  # DCT-III in-place on transpose
#         temp1 = temp1'
#     else
#         temp1 .= n * FFTW.ifft(temp1', dims=1)'
#     end
    
#     # Multiply temp1 = temp1 * Pi2
#     mul!(temp1, temp1, obj.Pi2)
    
#     # Second in-place DCT-III or IFFT
#     if isreal(obj)
#         FFTW.dct!(temp1', 3)
#         temp1 = temp1'
#     else
#         temp1 .= n * FFTW.ifft(temp1', dims=1)'
#     end
    
#     # Compute B = temp1 * Pi1
#     mul!(B, temp1, obj.Pi1)
#     return B
# end

function RightApply(obj::SSRFT{T}, M::AbstractMatrix{T2}) where {T<:Number, T2<:Number}
    p = size(M,1)
    k, n = obj.k, obj.n
    # M is presumably (?), so that we produce something shaped (?,?,?)
    # Original code: B is (k, n), then B[:, obj.coords] = M
    # So if M is (k, someP), then obj.coords must have length `someP`.
    @assert size(M,2) == k "M must have k columns for RightApply"
    @assert length(obj.coords) == size(M,2) "Mismatch with obj.coords"

    B = zeros(T, p, n)
    B[:, obj.coords] = M

    if isreal(obj)
        # DCT-III over dimension 1
        # B = FFTW.dct(B', 3)'  
        B = FFTW.dct(B')'  
    else
        B = n * FFTW.ifft(B', 1)' 
    end

    B = B * obj.Pi2

    if isreal(obj)
        # B = FFTW.dct(B', 3)'
        B = FFTW.dct(B')'
    else
        B = n * FFTW.ifft(B', 1)'
    end

    B = B * obj.Pi1
    return B
end

function RightApply(obj::SSRFT{T}, v::AbstractVector{T2}) where {T<:Number, T2<:Number}
    k, n = obj.k, obj.n
    @assert length(v) == k  "Input vector must have length k"

    # We'll wrap 'v' as (k,1) so it can go through the same logic as the matrix version
    M = reshape(v, k, 1)
    B = RightApply(obj, M)  # B is (k, n)
    return B   # Depending on your usage, you might want to return B as (k,n),
               # or you might prefer to pick out just some sub-vector.
end

function RightApply!(obj::SSRFT{T}, M::AbstractMatrix{T2}, B::AbstractMatrix{T}) where {T<:Number, T2<:Number}
    k, n = obj.k, obj.n
    # B is final shape (k, n)
    @assert size(B,1) == k
    @assert size(B,2) == n
    @assert size(M,1) == k
    @assert size(M,2) == length(obj.coords)

    # 1) Fill B so that B[:, obj.coords] = M and 0 elsewhere
    fill!(B, zero(T))
    @inbounds for colIdx in eachindex(obj.coords)
        j = obj.coords[colIdx]
        @inbounds @simd for i in 1:k
            B[i, j] = M[i, colIdx]
        end
    end

    # 2) In-place DCT-III or IFFT
    if isreal(obj)
        # FFTW.dct!(B', 3; dims=1)  # transform the transpose, then transpose back
        FFTW.dct!(B', 1)  # transform the transpose, then transpose back
        transpose!(B, B')  # or do B .= (B')' 
    else
        B .= n .* FFTW.ifft(B', 1)'
    end

    # 3) B = B * Pi2
    mul!(B, B, obj.Pi2)

    # 4) second in-place DCT-III or IFFT
    if isreal(obj)
        # FFTW.dct!(B', 3; dims=1)
        FFTW.dct!(B', 1)
        transpose!(B, B')
    else
        B .= n .* FFTW.ifft(B', 1)'
    end

    # 5) B = B * Pi1
    mul!(B, B, obj.Pi1)
    return B
end

function RightApply!(obj::SSRFT{T}, v::AbstractVector{T2}, B::AbstractMatrix{T}) where {T<:Number, T2<:Number}
    k, n = obj.k, obj.n
    @assert length(v) == k
    @assert size(B,1) == k
    @assert size(B,2) == n

    # Reshape v to (k,1)
    M = reshape(v, k, 1)
    RightApply!(obj, M, B)
    return B
end

# isreal method
function Base.isreal(obj::SSRFT)
    return obj.field == "real"
end

# issparse method
function SparseArrays.issparse(obj::SSRFT)
    return false
end

# nnz method
function SparseArrays.nnz(obj::SSRFT)
    return obj.k * obj.n
end

# Display method
function Base.show(io::IO, obj::SSRFT)
    println(io, "SSRFT of size ($(obj.k), $(obj.n)) with field $(obj.field)")
end

# Slicing the SSRFT object with view
function Base.view(obj::SSRFT{T}, I::AbstractUnitRange{<:Integer}, J::UnitRange{<:Integer}) where {T<:Number}
    # Create a view of the underlying Pi1, Pi2, and coords array
    subPi1 = view(obj.Pi1, J, J)
    subPi2 = view(obj.Pi2, J, J)
    subcoords = view(obj.coords, I)
    # Construct a new SSRFT object with the updated dimensions and subarray reference
    return SSRFT{T}(length(I), length(J), obj.field, subPi1, subPi2, subcoords, obj.transposeFlag)
end

# Slicing the SSRFT object with view (single index dispatch)
function Base.view(obj::SSRFT{T}, i::Int, j::Int) where {T<:Number}
    return view(obj, i:i, j:j)
end

function Base.copy(obj::SSRFT{T}) where {T<:Number}
    return SSRFT{T}(obj.k, obj.n, obj.field, obj.Pi1, obj.Pi2, obj.coords, obj.transposeFlag)
end

function Base.copy!(obj::SSRFT{T}, src::SSRFT{T}) where {T<:Number}
    obj.k = src.k
    obj.n = src.n
    obj.field = src.field
    obj.Pi1 = src.Pi1
    obj.Pi2 = src.Pi2
    obj.coords = src.coords
    obj.transposeFlag = src.transposeFlag
    return obj
end

# function Base.copy!(mat::AbstractMatrix{T}, obj::SSRFT{T}) where {T<:Number}
#     return copy!(mat, LeftApply(obj, Matrix{T}(I, obj.n, obj.n)))
# end

function Base.getindex(obj::SSRFT{T}, i::Int, ::Colon) where {T<:Number}
    if !obj.transposeFlag
        @assert 1 ≤ i ≤ obj.k "Row index out of range"
        # Build the entire k x n matrix, then pick row i
        Id = Matrix{T}(I, obj.n, obj.n)
        fullmat = LeftApply(obj, Id)
        return fullmat[i, :]
    else
        @assert 1 ≤ i ≤ obj.n "Row index out of range in transposed sense"
        # i-th row of SSRFT' => i-th column of SSRFT
        e = zeros(T, obj.n)
        e[i] = one(T)
        col = LeftApply(obj, e)  # k x 1
        # return reshape(col, 1, :)
        return vec(col) # ensures shape is (k,) not (k,1)
    end
end

# function Base.getindex(obj::SSRFT{T}, i::Int, ::Colon) where {T<:Number}
#     if !obj.transposeFlag
#         @assert 1 ≤ i ≤ obj.k "Row index out of range"
#         # Build the entire k x n matrix, then pick row i
#         Id = Matrix{T}(I, obj.n, obj.n)
#         fullmat = LeftApply(obj, Id)
#         return fullmat[i, :]
#     else
#         @assert 1 ≤ i ≤ obj.n "Row index out of range in transposed sense"
#         # i-th row of SSRFT' => i-th column of SSRFT
#         e = zeros(T, obj.n)
#         e[i] = one(T)
#         col = LeftApply(obj, e)  # k x 1
#         return reshape(col, 1, :)
#     end
# end

function Base.getindex(obj::SSRFT{T}, ::Colon, j::Int) where {T<:Number}
    if !obj.transposeFlag
        @assert 1 ≤ j ≤ obj.n "Column index out of range"
        e = zeros(T, obj.n)
        e[j] = one(T)
        col_2d = LeftApply(obj, e)  # (k,1)
        # Return a 1D vector of length k
        return dropdims(col_2d, dims=2)  # shape => (k,)
    else
        @assert 1 ≤ j ≤ obj.k "Column index out of range in transposed sense"
        Id = Matrix{T}(I, obj.n, obj.n)
        fullmat = LeftApply(obj, Id)  # (k,n)
        rowvec_2d = fullmat[j, :]     # shape => (1,n)
        # Return a 1D vector of length n
        return rowvec_2d[1, :]       # or dropdims(rowvec_2d, dims=1)
    end
end

# function Base.getindex(obj::SSRFT{T}, ::Colon, j::Int) where {T<:Number}
#     if !obj.transposeFlag
#         @assert 1 ≤ j ≤ obj.n "Column index out of range"
#         # j-th column of SSRFT => SSRFT * e_j
#         e = zeros(T, obj.n)
#         e[j] = one(T)
#         col = LeftApply(obj, e)  # k x 1
#         return col
#     else
#         @assert 1 ≤ j ≤ obj.k "Column index out of range in transposed sense"
#         # j-th column of SSRFT' => j-th row of SSRFT
#         Id = Matrix{T}(I, obj.n, obj.n)
#         fullmat = LeftApply(obj, Id)
#         rowvec = fullmat[j, :]
#         return reshape(rowvec, :, 1)
#     end
# end

# Dispatch for mul! with SSRFT on the left.
function LinearAlgebra.mul!(C::AbstractMatrix,
                            A::SSRFT,
                            B::AbstractMatrix,
                            α::Number=true,
                            β::Number=false)
    # Use SSRFT's left application: C = α * LeftApply(A,B) + β * C
    temp = LeftApply(A, B)
    C .= β * C .+ α * temp
    return C
end

# Dispatch for mul! with SSRFT on the left and SubArray arguments
function LinearAlgebra.mul!(C::SubArray,
                            A::SSRFT,
                            B::SubArray,
                            α::Number=true,
                            β::Number=false)
    # Use SSRFT's left application: C = α * LeftApply(A,B) + β * C
    temp = LeftApply(A, B)
    C .= β * C .+ α * temp
    return C
end

# Dispatch for mul! with SSRFT on the right.
function LinearAlgebra.mul!(C::AbstractMatrix,
                            A::AbstractMatrix,
                            B::SSRFT,
                            α::Number=true,
                            β::Number=false)
    # Use SSRFT's right application: C = α * RightApply(B,A) + β * C
    temp = RightApply(B, A)
    C .= β * C .+ α * temp
    return C
end

# Dispatch for mul! with SSRFT on the right and SubArray arguments
function LinearAlgebra.mul!(C::SubArray,
                            A::SubArray,
                            B::SSRFT,
                            α::Number=true,
                            β::Number=false)
    # Use SSRFT's right application: C = α * RightApply(B,A) + β * C
    temp = RightApply(B, A)
    C .= β * C .+ α * temp
    return C
end