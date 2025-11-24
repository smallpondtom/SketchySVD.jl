@testset "Integration Tests" begin
    Random.seed!(999)
    
    @testset "End-to-End: Small Scale Problem" begin
        # Problem dimensions
        m, n, true_rank = 100, 80, 10
        r = 10  # Target rank
        
        # Generate low-rank data
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 50
        A = U_true * Diagonal(S_true) * V_true'
        
        # Stream data column by column
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        for j in 1:n
            increment!(sk, A[:, j])
        end
        
        # Finalize and check
        finalize!(sk)
        
        # Reconstruct
        A_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        rel_error = norm(A - A_approx) / norm(A)
        
        @test rel_error < 0.1
        @test length(sk.Σ) == r
        @test issorted(sk.Σ, rev=true)
    end
    
    @testset "Comparison: Sketchy vs rSVD" begin
        # Same problem solved two ways
        m, n, true_rank = 150, 120, 15
        r = 15
        
        # Generate data
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 100
        A = U_true * Diagonal(S_true) * V_true'
        
        # Method 1: Sketchy (incremental)
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        full_increment!(sk, A)
        finalize!(sk)
        A_sketchy = sk.V * Diagonal(sk.Σ) * sk.W'
        
        # Method 2: rSVD (batch)
        U_rsvd, S_rsvd, V_rsvd = rsvd(A, r, p=10, q=2)
        A_rsvd = U_rsvd * Diagonal(S_rsvd) * V_rsvd'
        
        # Both should give good approximations
        err_sketchy = norm(A - A_sketchy) / norm(A)
        err_rsvd = norm(A - A_rsvd) / norm(A)
        
        @test err_sketchy < 0.1
        @test err_rsvd < 0.05
        
        # Singular values should be reasonably close
        rel_S_diff = norm(sk.Σ - S_rsvd) / norm(S_rsvd)
        @test rel_S_diff < 0.2  # Allow 20% difference due to different methods
    end
    
    @testset "Memory Efficiency: Sparse vs Gauss" begin
        m, n, r = 200, 150, 15
        
        # Create both types
        sk_gauss = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        sk_sparse = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse, verbose=false)
        
        # Check that sparse has fewer non-zeros
        nnz_gauss_Xi = length(sk_gauss.Ξ.Xi)
        nnz_sparse_Xi = nnz(sk_sparse.Ξ)
        
        @test nnz_sparse_Xi < nnz_gauss_Xi
        
        # Both should produce valid results
        X = randn(m, n)
        full_increment!(sk_gauss, X)
        full_increment!(sk_sparse, X)
        
        finalize!(sk_gauss)
        finalize!(sk_sparse)
        
        @test length(sk_gauss.Σ) == r
        @test length(sk_sparse.Σ) == r
    end
    
    @testset "Temporal Dynamics: Forgetting Factor" begin
        m, n, r = 80, 60, 8
        
        # Create data with temporal structure
        # First half: one pattern, second half: different pattern
        U1 = randn(m, r)
        V1 = randn(n÷2, r)
        S1 = sort(rand(r), rev=true) .* 50
        A1 = U1 * Diagonal(S1) * V1'
        
        U2 = randn(m, r)
        V2 = randn(n÷2, r)
        S2 = sort(rand(r), rev=true) .* 50
        A2 = U2 * Diagonal(S2) * V2'
        
        A = hcat(A1, A2)
        
        # Process with forgetting factor
        η = 0.95  # Slight forgetting
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        for j in 1:n
            increment!(sk, A[:, j], η, 1.0)
        end
        
        finalize!(sk)
        
        # Should still capture main structure
        @test length(sk.Σ) == r
        @test sk.Σ[1] > sk.Σ[end]
    end
    
    @testset "High Dimensional Streaming" begin
        m, n, r = 500, 400, 20
        batch_size = 50
        
        # Generate data in batches
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse, verbose=false)
        
        num_batches = n ÷ batch_size
        for b in 1:num_batches
            batch = randn(m, batch_size)
            
            # Process batch column by column
            for j in 1:batch_size
                increment!(sk, batch[:, j])
            end
        end
        
        @test sk.ct == n
        
        finalize!(sk)
        
        @test length(sk.Σ) == r
        @test all(sk.Σ .>= 0)
    end
    
    @testset "Error Estimation Accuracy" begin
        m, n, r = 150, 120, 12
        true_rank = 12
        
        # Create exact low-rank matrix
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 80
        A = U_true * Diagonal(S_true) * V_true'
        
        # Add small noise
        noise_level = 1e-3
        A_noisy = A + noise_level * randn(m, n)
        
        # Process with error estimation
        sk = init_sketchy(m=m, n=n, r=r, ErrorEstimate=true, q=100, 
                         ReduxMap=:gauss, verbose=false)
        full_increment!(sk, A_noisy)
        est_err, _ = finalize!(sk)
        
        # Reconstruct and compute actual error
        A_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        actual_err = norm(A_noisy - A_approx)
        
        # Estimated error should be in the right ballpark
        @test est_err >= 0
        # Order of magnitude check
        @test abs(log10(est_err) - log10(actual_err)) < 2.0
    end
    
    @testset "Spectral Decay Analysis" begin
        m, n, r = 150, 120, 15
        true_rank = 20
        
        # Create matrix with known spectral structure
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        # Exponential decay
        S_true = [100.0 * exp(-0.1 * i) for i in 1:true_rank]
        A = U_true * Diagonal(S_true) * V_true'
        
        # Process with spectral decay computation
        sk = init_sketchy(m=m, n=n, r=r, SpectralDecay=true, q=100, s=50,
                         ReduxMap=:gauss, verbose=false)
        full_increment!(sk, A)
        _, scree = finalize!(sk)
        
        # Check scree plot
        @test !isnothing(scree)
        @test length(scree) == sk.dims[:s]
        @test all(scree .>= 0)
        
        # Scree values should generally decrease
        # (not strictly monotonic due to approximation)
        @test scree[1] > scree[end÷2]
    end
    
    @testset "Rank Deficient Matrix" begin
        m, n, true_rank = 150, 120, 8
        r = 15  # Request more than true rank
        
        # Create rank-deficient matrix
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 60
        A = U_true * Diagonal(S_true) * V_true'
        
        # Process
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        full_increment!(sk, A)
        finalize!(sk)
        
        # Should detect that effective rank is lower
        # (singular values beyond true rank should be very small)
        significant_rank = sum(sk.Σ .> 1e-6 * sk.Σ[1])
        @test significant_rank <= true_rank + 3  # Allow small margin
    end
    
    @testset "Different Redux Maps Comparison" begin
        m, n, r = 100, 80, 10
        
        # Generate test data
        U_true = randn(m, r)
        V_true = randn(n, r)
        S_true = sort(rand(r), rev=true) .* 70
        A = U_true * Diagonal(S_true) * V_true'
        
        # Test all three redux maps
        redux_maps = [:gauss, :sparse, :ssrft]
        errors = Dict()
        
        for redux in redux_maps
            sk = init_sketchy(m=m, n=n, r=r, ReduxMap=redux, verbose=false)
            full_increment!(sk, A)
            finalize!(sk)
            
            A_approx = sk.V * Diagonal(sk.Σ) * sk.W'
            errors[redux] = norm(A - A_approx) / norm(A)
        end
        
        # All methods should give reasonable approximations
        @test errors[:gauss] < 0.15
        @test errors[:sparse] < 0.2
        @test errors[:ssrft] < 0.2
    end
    
    @testset "Incremental vs Batch Consistency" begin
        m, n, r = 100, 50, 10
        
        # Generate data
        X = rand(m, n)
        Σ = svdvals(X)[1:r]
        
        # Incremental processing
        sk1 = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        for j in 1:n
            increment!(sk1, X[:, j])
        end
        finalize!(sk1)
        
        # Batch processing
        sk2 = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        dump!(sk2, X)
        finalize!(sk2)
        
        # Results should be very similar
        @test norm(sk1.Σ - sk2.Σ) / norm(sk1.Σ) < 0.1
        @test norm(Σ - sk1.Σ) / norm(Σ) < 0.5
        @test norm(Σ - sk2.Σ) / norm(Σ) < 0.5
    end
    
    @testset "Large Scale Problem" begin
        m, n, r = 1000, 800, 20
        true_rank = 20
        
        # Generate large low-rank matrix
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 150
        A = U_true * Diagonal(S_true) * V_true'
        
        # Use sparse redux for efficiency
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse, verbose=false)
        
        # Process in chunks
        chunk_size = 100
        num_chunks = n ÷ chunk_size
        
        for i in 1:num_chunks
            start_col = (i-1) * chunk_size + 1
            end_col = i * chunk_size
            chunk = A[:, start_col:end_col]
            
            for j in 1:chunk_size
                increment!(sk, chunk[:, j])
            end
        end
        
        @test sk.ct == n
        
        finalize!(sk)
        
        # Check quality
        A_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        rel_error = norm(A - A_approx) / norm(A)
        
        @test rel_error < 0.15
        @test length(sk.Σ) == r
    end
    
    @testset "Adaptive rSVD Integration" begin
        m, n, true_rank = 200, 180, 18
        
        # Generate data with clear rank structure
        U_true = randn(m, true_rank)
        V_true = randn(n, true_rank)
        S_true = [100.0 * exp(-0.15 * i) for i in 1:true_rank]
        A = U_true * Diagonal(S_true) * V_true'
        
        # Use adaptive rSVD
        k_init = 30
        U_adp, S_adp, V_adp = rsvd_adaptive(A, k_init, tol=1e-2, p=10, q=2)
        
        detected_rank = length(S_adp)
        
        # Should detect approximately the true rank
        @test detected_rank >= true_rank - 3
        @test detected_rank <= true_rank + 3
        
        # Reconstruction quality
        A_adp = U_adp * Diagonal(S_adp) * V_adp'
        rel_error = norm(A - A_adp) / norm(A)
        @test rel_error < 0.05
    end
    
    @testset "Mixed Precision Compatibility" begin
        m, n, r = 80, 60, 8
        
        # Float32 data
        X_f32 = randn(Float32, m, n)
        
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Should handle Float32 input
        full_increment!(sk, X_f32)
        finalize!(sk)
        
        @test length(sk.Σ) == r
        @test all(sk.Σ .>= 0)
    end
    
    @testset "Robustness to Outliers" begin
        m, n, r = 100, 80, 10
        
        # Generate clean low-rank data
        U_true = randn(m, r)
        V_true = randn(n, r)
        S_true = sort(rand(r), rev=true) .* 60
        A = U_true * Diagonal(S_true) * V_true'
        
        # Add sparse outliers
        outlier_positions = rand(1:m*n, 50)
        A_outlier = copy(A)
        for pos in outlier_positions
            i = ((pos-1) % m) + 1
            j = ((pos-1) ÷ m) + 1
            A_outlier[i, j] += 100 * randn()  # Large outlier
        end
        
        # Process with sketchy
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        full_increment!(sk, A_outlier)
        finalize!(sk)
        
        # Should still capture underlying low-rank structure
        A_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        
        # Error relative to clean matrix
        rel_error_clean = norm(A - A_approx) / norm(A)
        @test rel_error_clean < 0.3  # Relaxed tolerance due to outliers
    end
end
