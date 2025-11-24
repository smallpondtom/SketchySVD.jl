function update_sketchy!(sketchy::Sketchy, x::AbstractVector, i::Int)
    # 1) X update
    # [Approach 1: Slowest] 
    # @views sketchy.X[:, i] = η .* sketchy.X[:, i] .+ ν .* (sketchy.Ξ * x)
    # [Approach 2: Somewhat faster]
    # rmul!(sketchy.X, η)
    # sketchy.X .*= η
    # @views sketchy.X[:, i] +=  ν .* (sketchy.Ξ * x)
    # [Approach 3: Fastest]
    mul!(view(sketchy.X, :, i), sketchy.Ξ, x)
    
    # 2) Y update
    row_Ω = sketchy.Ω'[i, :]   # picks out i-th row

    # row_Ω is a sparse vector of length M
    # x is a (N x 1) dense vector, presumably
    # NOTE: sketchy.Y .+= ν .* (x * row_Ω)  # rank-1 update (this gives error since row_Ω is sparse)
    @inbounds @fastmath for (idx, val) in pairs(row_Ω)  # or for idx in nonzeroinds(row_Ω)
        @views sketchy.Y[:, idx] .+= val .* x  # row_Ω[idx] = val
    end

    # 3) Z update
    z = sketchy.Φ * x
    row_Ψ = sketchy.Ψ'[i, :]

    # Same as Y sketch matrix
    # NOTE: sketchy.Z .+= ν .* (z * row_Ψ)  # rank-1 update (this gives error since row_Ψ is sparse)
    @inbounds @fastmath for (idx, val) in pairs(row_Ψ)  # or for idx in nonzeroinds(row_Ψ)
        @views sketchy.Z[:, idx] .+= val .* z  # row_Ψ[idx] = val
    end

    # 4) E update
    # Same approaches as the X sketch matrix can be considered
    if sketchy.ErrorEstimate || sketchy.SpectralDecay
        mul!(view(sketchy.E, :, i), sketchy.Θ, x) 
    end

    return 
end

function update_sketchy!(sketchy::Sketchy, x::AbstractVector, i::Int, 
                         η::Real, ν::Real)
    # 1) X update
    # [Approach 1: Slowest] 
    # @views sketchy.X[:, i] = η .* sketchy.X[:, i] .+ ν .* (sketchy.Ξ * x)
    # [Approach 2: Somewhat faster]
    # rmul!(sketchy.X, η)
    # sketchy.X .*= η
    # @views sketchy.X[:, i] +=  ν .* (sketchy.Ξ * x)
    # [Approach 3: Fastest]
    mul!(view(sketchy.X, :, i), sketchy.Ξ, x, ν, η)
    
    # 2) Y update
    rmul!(sketchy.Y, η)
    row_Ω = sketchy.Ω'[i, :]   # picks out i-th row

    # row_Ω is a sparse vector of length M
    # x is a (N x 1) dense vector, presumably
    # NOTE: sketchy.Y .+= ν .* (x * row_Ω)  # rank-1 update (this gives error since row_Ω is sparse)
    @inbounds @fastmath for (idx, val) in pairs(row_Ω)  # or for idx in nonzeroinds(row_Ω)
        @views sketchy.Y[:, idx] .+= ν * val .* x  # row_Ω[idx] = val
    end

    # 3) Z update
    rmul!(sketchy.Z, η)
    z = sketchy.Φ * x
    row_Ψ = sketchy.Ψ'[i, :]

    # Same as Y sketch matrix
    # NOTE: sketchy.Z .+= ν .* (z * row_Ψ)  # rank-1 update (this gives error since row_Ψ is sparse)
    @inbounds @fastmath for (idx, val) in pairs(row_Ψ)  # or for idx in nonzeroinds(row_Ψ)
        @views sketchy.Z[:, idx] .+= ν * val .* z  # row_Ψ[idx] = val
    end

    # 4) E update
    # Same approaches as the X sketch matrix can be considered
    if sketchy.ErrorEstimate || sketchy.SpectralDecay
        mul!(view(sketchy.E, :, i), sketchy.Θ, x, ν, η) 
    end

    return 
end


function update_sketchy_sparse!(sketchy::Sketchy, x::AbstractVector, i::Int)
    # X update - simplified since ν=1, η=1
    mul!(view(sketchy.X, :, i), sketchy.Ξ, x)
    
    # Get the transpose row more efficiently
    Ω_T = sketchy.Ω'
    
    # Use sparse matrix operations directly instead of iterating
    if isa(Ω_T, SparseMatrixCSC)
        # Direct sparse vector operations
        row_vals = view(Ω_T.nzval, Ω_T.colptr[i]:Ω_T.colptr[i+1]-1)
        row_indices = view(Ω_T.rowval, Ω_T.colptr[i]:Ω_T.colptr[i+1]-1)
        
        # Simplified update since ν=1
        @inbounds for (idx_pos, col_idx) in enumerate(row_indices)
            val = row_vals[idx_pos]
            # Use axpy! for better performance: y = a*x + y
            BLAS.axpy!(val, x, view(sketchy.Y, :, col_idx))
        end
    else
        # Fallback for other sparse types
        row_Ω = Ω_T[i, :]
        @inbounds for (idx, val) in pairs(row_Ω)
            BLAS.axpy!(val, x, view(sketchy.Y, :, idx))
        end
    end
    
    # Pre-compute Φ * x once
    z = sketchy.Φ * x
    
    # Same optimization for Ψ
    Ψ_T = sketchy.Ψ'
    
    if isa(Ψ_T, SparseMatrixCSC)
        # Direct sparse vector operations
        row_vals = view(Ψ_T.nzval, Ψ_T.colptr[i]:Ψ_T.colptr[i+1]-1)
        row_indices = view(Ψ_T.rowval, Ψ_T.colptr[i]:Ψ_T.colptr[i+1]-1)
        
        # Simplified update since ν=1
        @inbounds for (idx_pos, col_idx) in enumerate(row_indices)
            val = row_vals[idx_pos]
            BLAS.axpy!(val, z, view(sketchy.Z, :, col_idx))
        end
    else
        # Fallback for other sparse types
        row_Ψ = Ψ_T[i, :]
        @inbounds for (idx, val) in pairs(row_Ψ)
            BLAS.axpy!(val, z, view(sketchy.Z, :, idx))
        end
    end
    
    # E update - simplified since ν=1, η=1
    if sketchy.ErrorEstimate || sketchy.SpectralDecay
        mul!(view(sketchy.E, :, i), sketchy.Θ, x)
    end

    return 
end

function update_sketchy_sparse!(sketchy::Sketchy, x::AbstractVector, i::Int, 
                                η::Real, ν::Real)
    # 1) X update - already optimized
    mul!(view(sketchy.X, :, i), sketchy.Ξ, x, ν, η)
    
    # 2) Y update - optimized for sparse Ω
    rmul!(sketchy.Y, η)
    
    # Pre-compute ν * x to avoid repeated scaling
    νx = ν .* x
    
    # Get the transpose row more efficiently
    Ω_T = sketchy.Ω'
    
    # Use sparse matrix operations directly instead of iterating
    if isa(Ω_T, SparseMatrixCSC)
        # Direct sparse vector operations
        row_vals = view(Ω_T.nzval, Ω_T.colptr[i]:Ω_T.colptr[i+1]-1)
        row_indices = view(Ω_T.rowval, Ω_T.colptr[i]:Ω_T.colptr[i+1]-1)
        
        # Vectorized update using BLAS operations
        for (idx_pos, col_idx) in enumerate(row_indices)
            val = row_vals[idx_pos]
            # Use axpy! for better performance: y = a*x + y
            BLAS.axpy!(val, νx, view(sketchy.Y, :, col_idx))
        end
    else
        # Fallback for other sparse types
        row_Ω = Ω_T[i, :]
        @inbounds for (idx, val) in pairs(row_Ω)
            BLAS.axpy!(val, νx, view(sketchy.Y, :, idx))
        end
    end
    
    # 3) Z update - optimized for sparse Φ and Ψ
    rmul!(sketchy.Z, η)
    
    # Pre-compute Φ * x once
    z = sketchy.Φ * x
    νz = ν .* z
    
    # Same optimization for Ψ
    Ψ_T = sketchy.Ψ'
    
    if isa(Ψ_T, SparseMatrixCSC)
        # Direct sparse vector operations
        row_vals = view(Ψ_T.nzval, Ψ_T.colptr[i]:Ψ_T.colptr[i+1]-1)
        row_indices = view(Ψ_T.rowval, Ψ_T.colptr[i]:Ψ_T.colptr[i+1]-1)
        
        # Vectorized update
        for (idx_pos, col_idx) in enumerate(row_indices)
            val = row_vals[idx_pos]
            BLAS.axpy!(val, νz, view(sketchy.Z, :, col_idx))
        end
    else
        # Fallback for other sparse types
        row_Ψ = Ψ_T[i, :]
        @inbounds for (idx, val) in pairs(row_Ψ)
            BLAS.axpy!(val, νz, view(sketchy.Z, :, idx))
        end
    end
    
    # 4) E update - already optimized
    if sketchy.ErrorEstimate || sketchy.SpectralDecay
        mul!(view(sketchy.E, :, i), sketchy.Θ, x, ν, η)
    end

    return 
end


function increment!(sketchy::Sketchy{T}, x::AbstractArray{T1}, η::T=one(T), 
                    ν::T=one(T)) where {T<:Number, T1<:Number}
    n = sketchy.dims[:n]
    m = sketchy.dims[:m]

    # Ensure x is a column vector of length m
    @assert length(x) == m "Input vector x must have length m."

    # Update the sketches (optimized implementation)
    if ν == one(T) && η == one(T)
        if isa(sketchy.Φ, Gauss)
            update_sketchy!(sketchy, x, sketchy.ct+1)
        else  # Sparse or SSRFT
            update_sketchy_sparse!(sketchy, x, sketchy.ct+1)
        end
    else
        if isa(sketchy.Φ, Gauss)
            update_sketchy!(sketchy, x, sketchy.ct+1, η, ν)
        else  # Sparse or SSRFT
            update_sketchy_sparse!(sketchy, x, sketchy.ct+1, η, ν)
        end
    end

    # Reset the counter
    if sketchy.ct > n
        @error string(
            "Number of data streams exceeded the preset column size."
        )
    else
        # Increment the counter (which is also the column index)
        sketchy.ct += 1
    end

    return 
end

function dump!(sketchy::Sketchy{T}, X::AbstractArray{T1}, η::T=one(T), 
               ν::T=one(T)) where {T<:Number, T1<:Number}
    # 1) X update: Use mul! with BLAS for better performance
    if η == one(T)
        # X = X + ν * (Ξ * X_new)
        mul!(sketchy.X, sketchy.Ξ, X, ν, one(T))
    else
        # X = η * X + ν * (Ξ * X_new)
        mul!(sketchy.X, sketchy.Ξ, X, ν, η)
    end
    
    # 2) Y update: Use mul! with BLAS
    if η == one(T)
        # Y = Y + ν * (X_new * Ω')
        mul!(sketchy.Y, X, sketchy.Ω', ν, one(T))
    else
        # Y = η * Y + ν * (X_new * Ω')
        mul!(sketchy.Y, X, sketchy.Ω', ν, η)
    end
    
    # 3) Z update: More complex but can be optimized
    # Z = η * Z + ν * (Φ * X * Ψ')
    # Break this into two steps to use BLAS operations
    temp_matrix = sketchy.Φ * X  # Temporary matrix
    if η == one(T)
        mul!(sketchy.Z, temp_matrix, sketchy.Ψ', ν, one(T))
    else
        mul!(sketchy.Z, temp_matrix, sketchy.Ψ', ν, η)
    end
    
    # 4) E update: Use mul! with BLAS
    if sketchy.ErrorEstimate
        if η == one(T)
            mul!(sketchy.E, sketchy.Θ, X, ν, one(T))
        else
            mul!(sketchy.E, sketchy.Θ, X, ν, η)
        end
    end
    
    return 
end

# function dump!(sketchy::Sketchy{T}, X::AbstractArray{T1}, η::T=one(T), 
#                ν::T=one(T)) where {T<:Number, T1<:Number}
#     sketchy.X = η .* sketchy.X .+ ν .* (sketchy.Ξ * X)
#     sketchy.Y = η .* sketchy.Y .+ ν .* (X * sketchy.Ω')
#     sketchy.Z = η .* sketchy.Z .+ ν .* (sketchy.Φ * X * sketchy.Ψ')
#     if !isnothing(sketchy.E)
#         sketchy.E = η .* sketchy.E .+ ν .* (sketchy.Θ * X)
#     end
#     return 
# end


function full_increment!(sketchy::Sketchy{T}, X::AbstractArray{T1}, 
                         η::T=one(T), ν::T=one(T); 
                         runtime::Bool=false, dump_all::Bool=false, 
                         terminate::Bool=false) where {T<:Number, T1<:Number}
    # Incremental POD
    K = size(X, 2)
    if sketchy.verbose
        p = Progress(K; desc="Incrementing iSVD...")
    end

    if dump_all  # dump all data at once
        t = runtime ? Float32 : nothing # Preallocate time
        if K == sketchy.dims[:n]
            if runtime
                t = @elapsed dump!(sketchy, X, η, ν)
            else
                dump!(sketchy, X, η, ν)
            end
        elseif K < sketchy.dims[:n]
            H = spzeros(sketchy.dims[:m], sketchy.dims[:n])
            copyto!(view(H, :, sketchy.ct+1:sketchy.ct+K), X)
            if runtime
                t = @elapsed dump!(sketchy, H, η, ν)
            else
                dump!(sketchy, H, η, ν)
            end
        else
            @error string(
                "Number of data streams must be less than or equal ", 
                "to the column size of the data matrix to dump all at once."
            )
        end
        sketchy.ct += K
    else         # Incremental update for all data
        t = runtime ? Vector{Float32}(undef,K) : nothing # Preallocate time
        i = 0
        for x in eachcol(X)
            if runtime
                t[i+=1] = @elapsed increment!(sketchy, x, η, ν)
            else
                increment!(sketchy, x, η, ν)
            end
            if sketchy.verbose
                next!(p)
            end
        end
    end

    # Terminate the sketchy algorithm
    if terminate
        est_err_squared, scree = finalize!(sketchy)
        return (est_err_squared=est_err_squared, scree=scree, runtime=t)
    else
        return (runtime=t,)  # Note the comma to make it a NamedTuple
    end
end