"""
Sparse dimension reduction map. See Algorithm SM3.3 in Supplementary Material of [TYUC2019].
"""

mutable struct Sparse{T<:Number} <: DimRedux
    k::Int
    n::Int
    field::String
    Xi::SparseMatrixCSC{T, Int}
    transposeFlag::Bool
end

# Constructor
function Sparse(k::Int, n::Int; zeta::Int=min(k,8), field::String="real")
    # if k > n
    #     error("k should be less than or equal to n.")
    # end
    if zeta < 1 || zeta > k
        error("zeta should be between 1 and k.")
    end
    field = lowercase(field)
    transposeFlag = false

    # Create indCol: repeat each column index zeta times
    indCol = repeat(1:n, inner=zeta)

    # Initialize indRow
    indRow = Vector{Int}(undef, n * zeta)
    idx = 1
    for tt in 1:n
        rows = sort(sample(1:k, zeta; replace=false))
        indRow[idx:idx+zeta-1] = rows
        idx += zeta
    end

    # Generate values
    if field == "real"
        vals = sign.(randn(n * zeta))
        Xi = sparse(indRow, indCol, vals, k, n)
        return Sparse{Float64}(k, n, field, Xi, transposeFlag)
    elseif field == "complex"
        vals = sign.(randn(n * zeta) .+ im * randn(n * zeta))
        Xi = sparse(indRow, indCol, vals, k, n)
        return Sparse{ComplexF64}(k, n, field, Xi, transposeFlag)
    else
        error("Input 'field' should be 'real' or 'complex'.")
    end
end

# Redraw another Sparse random matrix
function redraw(obj::Sparse{T}, k, n) where {T<:Number}
    return Sparse(k, n, field=obj.field)
end

function redraw(obj::Sparse{T}) where {T<:Number}
    return Sparse(obj.k, obj.n, field=obj.field)
end

# LeftApply method
function LeftApply(obj::Sparse{T}, A::AbstractArray{T2}) where {T<:Number, T2<:Number}
    return obj.Xi * A
end

function LeftApply!(obj::Sparse{T}, A::AbstractArray{T2}, C::AbstractArray{T}) where {T<:Number, T2<:Number}
    mul!(C, obj.Xi, A)
    return C
end

# RightApply method
function RightApply(obj::Sparse{T}, A::AbstractArray{T2}) where {T<:Number, T2<:Number}
    return A * obj.Xi
end

function RightApply!(obj::Sparse{T}, A::AbstractArray{T2}, C::AbstractArray{T}) where {T<:Number, T2<:Number}
    mul!(C, A, obj.Xi)
    return C
end

# isreal method
function Base.isreal(obj::Sparse)
    return obj.field == "real"
end

# issparse method
function SparseArrays.issparse(obj::Sparse)
    return true
end

# nnz method
function SparseArrays.nnz(obj::Sparse)
    return nnz(obj.Xi)
end

# Display method
function Base.show(io::IO, obj::Sparse)
    if !obj.transposeFlag
        show(io, obj.Xi)
    else
        show(io, obj.Xi')
    end
end

# Slicing the Sparse object with view
function Base.view(obj::Sparse{T}, I::AbstractUnitRange{<:Integer}, J::UnitRange{<:Integer}) where {T<:Number}
    # Create a view of the underlying Xi array
    subXi = view(obj.Xi, I, J)
    # Construct a new Sparse object with the updated dimensions and subarray reference
    return Sparse{T}(length(I), length(J), obj.field, subXi, obj.transposeFlag)
end

# Slicing the Sparse object with view (single index dispatch)
function Base.view(obj::Sparse{T}, i::Int, j::Int) where {T<:Number}
    return view(obj, i:i, j:j)
end

function Base.copy(obj::Sparse{T}) where {T<:Number}
    return Sparse{T}(obj.k, obj.n, obj.field, obj.Xi, obj.transposeFlag)
end

function Base.copy!(obj::Sparse{T}, src::Sparse{T}) where {T<:Number}
    obj.k = src.k
    obj.n = src.n
    obj.field = src.field
    obj.Xi = src.Xi
    obj.transposeFlag = src.transposeFlag
end

function Base.copy!(mat::AbstractArray{T}, obj::Sparse{T}) where {T<:Number}
    copyto!(mat, obj.Xi)
end

function Base.getindex(obj::Sparse{T}, i::Int, ::Colon) where {T<:Number}
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

function Base.getindex(obj::Sparse{T}, ::Colon, j::Int) where {T<:Number}
    if obj.transposeFlag
        # "column j" of the transposed matrix = "row j" of Xi
        return obj.Xi[j, :]
    else
        # "column j" of the normal matrix
        return obj.Xi[:, j]
    end
end

function Base.getindex(obj::Sparse{T}, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}) where {T<:Number}
    # For a non-transposed case:
    if !obj.transposeFlag
        return obj.Xi[I, J]
    else
        # For a transposed case, treat it as Xi'[I, J] = Xi[J, I]
        return obj.Xi[J, I]
    end
end

# # Dispatch for mul! with Sparse objects
# function LinearAlgebra.mul!(C::AbstractVecOrMat{T},
#                           A::Sparse{T},
#                           B::AbstractVecOrMat{T},
#                           α::Number=true,
#                           β::Number=false) where {T<:Number}
#     # Handle transposition if needed
#     Xi = A.transposeFlag ? A.Xi' : A.Xi
    
#     # Use sparse matrix multiplication
#     mul!(C, Xi, B, α, β)
#     return C
# end

# # Additional dispatch for when B is the Sparse object
# function LinearAlgebra.mul!(C::AbstractVecOrMat{T},
#                           A::AbstractVecOrMat{T},
#                           B::Sparse{T},
#                           α::Number=true,
#                           β::Number=false) where {T<:Number}
#     # Handle transposition if needed
#     Xi = B.transposeFlag ? B.Xi' : B.Xi
    
#     # Use sparse matrix multiplication
#     mul!(C, A, Xi, α, β)
#     return C
# end

# Dispatch for mul! with Sparse objects
function LinearAlgebra.mul!(C::AbstractVecOrMat,
                          A::Sparse,
                          B::AbstractVecOrMat,
                          α::Number=true,
                          β::Number=false)
    # Handle transposition if needed
    Xi = A.transposeFlag ? A.Xi' : A.Xi
    
    # Use sparse matrix multiplication
    T = eltype(C)
    mul!(C, Xi, B, T(α), T(β)) 
    return C
end

# Additional dispatch for when B is the Sparse object
function LinearAlgebra.mul!(C::AbstractVecOrMat,
                          A::AbstractVecOrMat,
                          B::Sparse,
                          α::Number=true,
                          β::Number=false)
    # Handle transposition if needed
    Xi = B.transposeFlag ? B.Xi' : B.Xi
    
    # Use sparse matrix multiplication
    T = eltype(C)
    mul!(C, A, Xi, T(α), T(β))
    return C
end