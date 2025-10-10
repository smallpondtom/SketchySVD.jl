function finalize!(sketchy::Sketchy{T}) where T<:Number
    if sketchy.verbose
        @info "Finalizing the sketch..."
        @info "Computing the low-rank approximation..."
    end
    # initial approximation with QR decompositions
    Qs = Matrix(qr(sketchy.Y).Q)
    Ps = Matrix(qr(sketchy.X').Q)
    tmp = (sketchy.Φ * Qs) \ sketchy.Z  # Solve (Φ * Q) * X = Z
    C = tmp / (sketchy.Ψ * Ps)'      # Solve X = X / (Ψ * P)'

    # Truncate the approximation and compute the singular vectors
    r = sketchy.dims[:r]
    Vc, Σc, Wc = svd(C)
    mul!(sketchy.V, Qs, Vc[:,1:r])
    copy!(sketchy.Σ, Σc[1:r])
    mul!(sketchy.W, Ps, Wc[:,1:r])

    # Compute the error estimation
    if sketchy.ErrorEstimate 
        if sketchy.verbose
            @info "Computing error estimate..."
        end
        Ahat = Qs * C * Ps'
        est_err_squared = norm(
            sketchy.E - sketchy.Θ * Ahat, 2) / sketchy.β / sketchy.dims[:q]
    else
        if sketchy.verbose
            @info "Skipping error estimate..."
        end
        est_err_squared = nothing
    end

    if sketchy.SpectralDecay
        if sketchy.verbose
            @info "Computing spectral decay..."
        end
        # Reconstruction of the data 
        if !(@isdefined Ahat)
            Ahat =  Qs * C * Ps'
        end
        # Compute the spectral decay for scree plot
        err2_0 = norm(sketchy.E, 2) / sketchy.β / sketchy.dims[:q]
        err2 = norm(
            sketchy.E - sketchy.Θ * Ahat, 2) / sketchy.β / sketchy.dims[:q]
        scree = map(
            (r) -> ((sum(svdvals(Ahat)[r+1:end]) + err2) / err2_0)^2,
            1:sketchy.dims[:s]
        )
    else
        if sketchy.verbose
            @info "Skipping spectral decay..."
        end
        scree = nothing
    end

    # Return the results
    if sketchy.verbose
        @info "Finalization complete."
    end
    return est_err_squared, scree
end
