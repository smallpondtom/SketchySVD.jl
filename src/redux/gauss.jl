"""
Gaussian dimension reduction map. See Algorithm SM3.1 in Supplementary Material of [TYUC2019].
"""

# Define the Gauss type as a mutable struct
mutable struct Gauss{T<:Number} <: DimRedux
    k::Int
    n::Int
    field::String
    Xi::Array{T,2}
    transposeFlag::Bool
end

# Constructor
function Gauss(k::Int, n::Int; field::String="real")
    # if k > n
    #     error("k should be less than or equal to n.")
    # end
    field = lowercase(field)
    transposeFlag = false
    if field == "real"
        Xi = randn(k, n)
        return Gauss{Float64}(k, n, field, Xi, transposeFlag)
    elseif field == "complex"
        Xi = randn(k, n) .+ im * randn(k, n)
        return Gauss{ComplexF64}(k, n, field, Xi, transposeFlag)
    else
        error("Input 'field' should be 'real' or 'complex'.")
    end
end

# Redraw another Gaussian random matrix
function redraw(obj::Gauss{T}, k, n) where {T<:Number}
    return Gauss(k, n, field=obj.field)
end

function redraw(obj::Gauss{T}) where {T<:Number}
    return Gauss(obj.k, obj.n, field=obj.field)
end

# LeftApply method
function LeftApply(obj::Gauss{T}, A::AbstractArray{T2}) where {T<:Number, T2<:Number}
    return obj.Xi * A
end

function LeftApply!(obj::Gauss{T}, A::AbstractArray{T2}, C::AbstractArray{T}) where {T<:Number, T2<:Number}
    mul!(C, obj.Xi, A)
    return C
end

# RightApply method
function RightApply(obj::Gauss{T}, A::AbstractArray{T2}) where {T<:Number, T2<:Number}
    return A * obj.Xi
end

function RightApply!(obj::Gauss{T}, A::AbstractArray{T2}, C::AbstractArray{T}) where {T<:Number, T2<:Number}
    mul!(C, A, obj.Xi)
    return C
end

# isreal method
function Base.isreal(obj::Gauss)
    return obj.field == "real"
end

# issparse method (always returns false)
function SparseArrays.issparse(obj::Gauss)
    return false
end

# nnz method
function SparseArrays.nnz(obj::Gauss)
    return nnz(obj.Xi)
end

# Display method
function Base.show(io::IO, obj::Gauss)
    if !obj.transposeFlag
        show(io, obj.Xi)
    else
        show(io, obj.Xi')
    end
end

# Slicing the Gauss object with view
function Base.view(obj::Gauss{T}, I::AbstractUnitRange{<:Integer}, J::UnitRange{<:Integer}) where {T<:Number}
    # Create a view of the underlying Xi array
    subXi = view(obj.Xi, I, J)
    # Construct a new Gauss object with the updated dimensions and subarray reference
    return Gauss{T}(length(I), length(J), obj.field, subXi, obj.transposeFlag)
end

# Slicing the Gauss object with view (single index dispatch)
function Base.view(obj::Gauss{T}, i::Int, j::Int) where {T<:Number}
    return view(obj, i:i, j:j)
end

function Base.copy(obj::Gauss{T}) where {T<:Number}
    return Gauss{T}(obj.k, obj.n, obj.field, obj.Xi, obj.transposeFlag)
end

function Base.copy!(obj::Gauss{T}, src::Gauss{T}) where {T<:Number}
    obj.k = src.k
    obj.n = src.n
    obj.field = src.field
    obj.Xi = src.Xi
    obj.transposeFlag = src.transposeFlag
end

function Base.copy!(mat::AbstractArray{T}, obj::Gauss{T}) where {T<:Number}
    copyto!(mat, obj.Xi)
end

function Base.getindex(obj::Gauss{T}, i::Int, ::Colon) where {T<:Number}
    # decide what 'row i' means depending on transposeFlag
    if obj.transposeFlag
        # If transposeFlag is true, then "row i" of obj actually
        # corresponds to "column i" of the underlying Xi.
        return obj.Xi[:, i]
    else
        # If transposeFlag is false, "row i" means the i-th row of Xi.
        return obj.Xi[i, :]
    end
end

function Base.getindex(obj::Gauss{T}, ::Colon, j::Int) where {T<:Number}
    if obj.transposeFlag
        # "column j" of the transposed matrix = "row j" of Xi
        return obj.Xi[j, :]
    else
        # "column j" of the normal matrix
        return obj.Xi[:, j]
    end
end

function Base.getindex(obj::Gauss{T}, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}) where {T<:Number}
    # For a non-transposed case:
    if !obj.transposeFlag
        return obj.Xi[I, J]
    else
        # For a transposed case, treat it as Xi'[I, J] = Xi[J, I]
        return obj.Xi[J, I]
    end
end

# Additional dispatch for when A is a Gauss object
function LinearAlgebra.mul!(C::AbstractVecOrMat,
                            A::Gauss,
                            B::AbstractVecOrMat,
                            α::Number=true,
                            β::Number=false)
    # Use the appropriate interpretation of Xi based on transposeFlag
    Xi = A.transposeFlag ? A.Xi' : A.Xi
    T = eltype(C)
    mul!(C, Xi, B, T(α), T(β))
    return C
end

# Additional dispatch for when B is a Gauss object
function LinearAlgebra.mul!(C::AbstractVecOrMat,
                            A::AbstractVecOrMat,
                            B::Gauss,
                            α::Number=true,
                            β::Number=false)
    # Use the appropriate interpretation of Xi based on transposeFlag
    Xi = B.transposeFlag ? B.Xi' : B.Xi
    T = eltype(C)
    mul!(C, A, Xi, T(α), T(β))
    return C
end