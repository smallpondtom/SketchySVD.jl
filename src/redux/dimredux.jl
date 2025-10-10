# Define an abstract type for DimRedux
abstract type DimRedux end

function Base.adjoint(obj::DimRedux)
    new_obj = copy(obj)  # Create a shallow copy
    new_obj.transposeFlag = !obj.transposeFlag
    return new_obj
end

function Base.size(obj::DimRedux)
    if !obj.transposeFlag
        return (obj.k, obj.n)
    else
        return (obj.n, obj.k)
    end
end

function Base.size(obj::DimRedux, dim::Integer)
    s = size(obj)
    if dim == 1
        return s[1]
    elseif dim == 2
        return s[2]
    else
        error("Dimension must be 1 or 2.")
    end
end

function Base.length(obj::DimRedux)
    return max(obj.k, obj.n)
end

function numel(obj::DimRedux)
    return obj.k * obj.n
end

function istransposed(obj::DimRedux)
    return obj.transposeFlag
end

import Base: *

function *(obj1::DimRedux, obj2::AbstractArray)
    if !obj1.transposeFlag
        return LeftApply(obj1, obj2)
    else
        return (RightApply(adjoint(obj1), adjoint(obj2)))'
    end
end

function *(obj1::AbstractArray, obj2::DimRedux)
    if !obj2.transposeFlag
        return RightApply(obj2, obj1)
    else
        return (LeftApply(adjoint(obj2), adjoint(obj1)))'
    end
end

# Multiplication when DimRedux is on the left
# function *(obj1::DimRedux, obj2::AbstractArray)
#     if !obj1.transposeFlag
#         # obj1 is not transposed
#         C = similar(obj1.Xi, obj1.k, size(obj2, 2))
#         LeftApply!(obj1, obj2, C)
#         return C
#     else
#         # obj1 is transposed
#         C = similar(obj2, obj1.n, size(obj2, 2))
#         # Compute C = adjoint( RightApply!( adjoint(obj1), adjoint(obj2), temp ) )
#         temp = similar(C, size(C, 2), size(C, 1))  # Note the swapped dimensions
#         RightApply!(adjoint(obj1), adjoint(obj2), temp)
#         C .= adjoint(temp)
#         return C
#     end
# end

# # Multiplication when DimRedux is on the right
# function *(obj1::AbstractArray, obj2::DimRedux)
#     if !obj2.transposeFlag
#         # obj2 is not transposed
#         C = similar(obj1, size(obj1, 1), obj2.n)
#         RightApply!(obj2, obj1, C)
#         return C
#     else
#         # obj2 is transposed
#         C = similar(obj1, size(obj1, 1), obj2.k)
#         # Compute C = adjoint( LeftApply!( adjoint(obj2), adjoint(obj1), temp ) )
#         temp = similar(C, size(C, 2), size(C, 1))  # Note the swapped dimensions
#         LeftApply!(adjoint(obj2), adjoint(obj1), temp)
#         C .= adjoint(temp)
#         return C
#     end
# end
