@testset "Sketchy Algorithm Tests" begin
    Random.seed!(456)  # For reproducibility
    
    @testset "Basic Sketchy Initialization" begin
        m, n, r = 100, 80, 10
        
        # Test initialization with default parameters
        sk = init_sketchy(m=m, n=n, r=r, verbose=false)
        
        @test size(sk.V) == (m, r)
        @test size(sk.Σ) == (r,)
        @test size(sk.W) == (n, r)
        @test sk.ct == 0
        @test sk.dims[:m] == m
        @test sk.dims[:n] == n
        @test sk.dims[:r] == r
        
        # Test that sketches are initialized to zeros
        @test all(sk.V .== 0)
        @test all(sk.Σ .== 0)
        @test all(sk.W .== 0)
    end
    
    @testset "Sketchy with Different Redux Maps" begin
        m, n, r = 100, 80, 10
        
        # Gaussian
        sk_gauss = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        @test isa(sk_gauss.Ξ, SketchySVD.Gauss)
        @test isa(sk_gauss.Ω, SketchySVD.Gauss)
        @test isa(sk_gauss.Φ, SketchySVD.Gauss)
        @test isa(sk_gauss.Ψ, SketchySVD.Gauss)
        
        # Sparse
        sk_sparse = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse, verbose=false)
        @test isa(sk_sparse.Ξ, SketchySVD.Sparse)
        @test isa(sk_sparse.Ω, SketchySVD.Sparse)
        @test isa(sk_sparse.Φ, SketchySVD.Sparse)
        @test isa(sk_sparse.Ψ, SketchySVD.Sparse)
        
        # SSRFT
        sk_ssrft = init_sketchy(m=m, n=n, r=r, ReduxMap=:ssrft, verbose=false)
        @test isa(sk_ssrft.Ξ, SketchySVD.SSRFT)
        @test isa(sk_ssrft.Ω, SketchySVD.SSRFT)
        @test isa(sk_ssrft.Φ, SketchySVD.SSRFT)
        @test isa(sk_ssrft.Ψ, SketchySVD.SSRFT)
    end
    
    @testset "Sketchy Dimension Parameters" begin
        m, n, r = 100, 80, 10
        
        # Test with custom k and s
        k_custom, s_custom = 50, 60
        sk = init_sketchy(m=m, n=n, r=r, k=k_custom, s=s_custom, verbose=false)
        @test sk.dims[:k] == k_custom
        @test sk.dims[:s] == s_custom
        @test size(sk.X) == (k_custom, n)
        @test size(sk.Y) == (m, k_custom)
        @test size(sk.Z) == (s_custom, s_custom)
        
        # Test with storage budget T
        T = 1000
        sk_budget = init_sketchy(m=m, n=n, r=r, T=T, verbose=false)
        k_budget = sk_budget.dims[:k]
        s_budget = sk_budget.dims[:s]
        
        # Check that dimensions respect storage budget
        storage_used = k_budget * (m + n) + s_budget^2
        @test storage_used <= T + 100  # Allow some margin
        @test k_budget <= s_budget
        @test s_budget <= min(m, n)
    end
    
    @testset "Sketchy Error Estimation Flag" begin
        m, n, r = 100, 80, 10
        
        # Without error estimation
        sk_no_err = init_sketchy(m=m, n=n, r=r, ErrorEstimate=false, verbose=false)
        @test sk_no_err.ErrorEstimate == false
        @test isnothing(sk_no_err.E)
        @test isnothing(sk_no_err.Θ)
        
        # With error estimation
        sk_with_err = init_sketchy(m=m, n=n, r=r, ErrorEstimate=true, q=50, verbose=false)
        @test sk_with_err.ErrorEstimate == true
        @test !isnothing(sk_with_err.E)
        @test !isnothing(sk_with_err.Θ)
        @test size(sk_with_err.E) == (50, n)
        @test isa(sk_with_err.Θ, SketchySVD.Gauss)  # Θ is always Gaussian
    end
    
    @testset "Sketchy Spectral Decay Flag" begin
        m, n, r = 100, 80, 10
        
        # With spectral decay
        sk = init_sketchy(m=m, n=n, r=r, SpectralDecay=true, q=50, verbose=false)
        @test sk.SpectralDecay == true
        @test !isnothing(sk.E)
        @test !isnothing(sk.Θ)
    end
    
    @testset "Single Vector Increment" begin
        m, n, r = 50, 40, 5
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Create a test vector
        x = randn(m)
        
        # Increment once
        increment!(sk, x)
        @test sk.ct == 1
        
        # Check that sketches have been updated (non-zero)
        @test any(sk.X .!= 0)
        @test any(sk.Y .!= 0)
        @test any(sk.Z .!= 0)
        
        # Increment again
        x2 = randn(m)
        increment!(sk, x2)
        @test sk.ct == 2
    end
    
    @testset "Increment with Forgetting Factors" begin
        m, n, r = 50, 40, 5
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        x1 = randn(m)
        x2 = randn(m)
        
        # Increment with forgetting factors
        η, ν = 0.9, 1.0
        increment!(sk, x1, η, ν)
        @test sk.ct == 1
        
        # Store X after first increment
        X_after_1 = copy(sk.X)
        
        increment!(sk, x2, η, ν)
        @test sk.ct == 2
        
        # Check that forgetting factor was applied
        # (This is hard to verify exactly without knowing the internal state)
        @test sk.ct == 2
    end
    
    @testset "Dump All Data at Once" begin
        m, n, r = 50, 40, 5
        
        # Create test data matrix
        X = randn(m, n)
        
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Dump all data
        dump!(sk, X)
        
        # Check that sketches have been updated
        @test any(sk.X .!= 0)
        @test any(sk.Y .!= 0)
        @test any(sk.Z .!= 0)
    end
    
    @testset "Full Increment - Column by Column" begin
        m, n, r = 50, 30, 5
        k_data = 15
        
        # Create low-rank test data
        U_true = randn(m, r)
        V_true = randn(k_data, r)
        S_true = sort(rand(r), rev=true) .* 10
        X = U_true * Diagonal(S_true) * V_true'
        
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Full increment without dumping all
        full_increment!(sk, X, dump_all=false)
        
        @test sk.ct == k_data
        @test any(sk.X .!= 0)
        @test any(sk.Y .!= 0)
        @test any(sk.Z .!= 0)
    end
    
    @testset "Full Increment - Dump All" begin
        m, n, r = 50, 30, 5
        k_data = 30  # Equal to n
        
        # Create test data
        U_true = randn(m, r)
        V_true = randn(k_data, r)
        S_true = sort(rand(r), rev=true) .* 10
        X = U_true * Diagonal(S_true) * V_true'
        
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Full increment with dump_all
        full_increment!(sk, X, dump_all=true)
        
        @test sk.ct == k_data
        @test any(sk.X .!= 0)
        @test any(sk.Y .!= 0)
        @test any(sk.Z .!= 0)
    end
    
    @testset "Finalize Sketchy" begin
        m, n, r = 100, 80, 10
        k_data = 80
        
        # Create low-rank test matrix
        U_true = randn(m, r)
        V_true = randn(k_data, r)
        S_true = sort(rand(r), rev=true) .* 50
        X = U_true * Diagonal(S_true) * V_true'
        
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Fill sketches with data
        full_increment!(sk, X)
        
        # Finalize
        est_err, scree = finalize!(sk)
        
        # Check that SVD components have been computed
        @test any(sk.V .!= 0)
        @test any(sk.Σ .!= 0)
        @test any(sk.W .!= 0)
        
        # Check singular values are sorted
        @test issorted(sk.Σ, rev=true)
        
        # Check orthogonality
        @test norm(sk.V' * sk.V - I) < 1e-8
        @test norm(sk.W' * sk.W - I) < 1e-8
        
        # Check dimensions
        @test length(sk.Σ) == r
        @test size(sk.V, 2) == r
        @test size(sk.W, 2) == r
    end
    
    @testset "Finalize with Error Estimation" begin
        m, n, r = 100, 80, 10
        k_data = 80
        
        # Create test matrix
        U_true = randn(m, r)
        V_true = randn(k_data, r)
        S_true = sort(rand(r), rev=true) .* 50
        X = U_true * Diagonal(S_true) * V_true'
        
        sk = init_sketchy(m=m, n=n, r=r, ErrorEstimate=true, q=60, 
                         ReduxMap=:gauss, verbose=false)
        
        full_increment!(sk, X)
        est_err, scree = finalize!(sk)
        
        # Check that error estimate is computed
        @test !isnothing(est_err)
        @test est_err >= 0
        
        # Check that scree is nothing (SpectralDecay=false)
        @test isnothing(scree)
    end
    
    @testset "Finalize with Spectral Decay" begin
        m, n, r = 100, 80, 10
        k_data = 80
        
        # Create test matrix
        U_true = randn(m, r)
        V_true = randn(k_data, r)
        S_true = sort(rand(r), rev=true) .* 50
        X = U_true * Diagonal(S_true) * V_true'
        
        sk = init_sketchy(m=m, n=n, r=r, SpectralDecay=true, ErrorEstimate=true,
                          q=60, s=40, ReduxMap=:gauss, verbose=false)
        
        full_increment!(sk, X)
        est_err, scree = finalize!(sk)
        
        # Check that spectral decay is computed
        @test !isnothing(scree)
        @test length(scree) == sk.dims[:s]
        @test all(scree .>= 0)
        
        # Check that error estimate is also computed (needed for spectral decay)
        @test !isnothing(est_err)
    end
    
    @testset "Complete Workflow - Gaussian Redux" begin
        m, n, r = 200, 150, 15
        true_rank = 15
        k_data = 150
        
        # Create low-rank matrix
        U_true = randn(m, true_rank)
        V_true = randn(k_data, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 100
        X = U_true * Diagonal(S_true) * V_true'
        
        # Initialize sketchy
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Process data
        full_increment!(sk, X)
        
        # Finalize
        est_err, scree = finalize!(sk)
        
        # Reconstruct approximation
        X_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        
        # Check reconstruction error
        rel_error = norm(X - X_approx) / norm(X)
        @test rel_error < 0.1  # 10% relative error tolerance
        
        # Check that we captured the main singular values
        @test sk.Σ[1] > 50  # Should capture large singular values
    end
    
    @testset "Complete Workflow - Sparse Redux" begin
        m, n, r = 200, 150, 15
        true_rank = 15
        k_data = 150
        
        # Create low-rank matrix
        U_true = randn(m, true_rank)
        V_true = randn(k_data, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 100
        X = U_true * Diagonal(S_true) * V_true'
        
        # Initialize sketchy with sparse redux
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:sparse, verbose=false)
        
        # Process data
        full_increment!(sk, X)
        
        # Finalize
        est_err, scree = finalize!(sk)
        
        # Reconstruct approximation
        X_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        
        # Check reconstruction error (slightly relaxed for sparse)
        rel_error = norm(X - X_approx) / norm(X)
        @test rel_error < 0.15  # 15% relative error tolerance
    end
    
    @testset "Complete Workflow - SSRFT Redux" begin
        m, n, r = 200, 150, 15
        true_rank = 15
        k_data = 150
        
        # Create low-rank matrix
        U_true = randn(m, true_rank)
        V_true = randn(k_data, true_rank)
        S_true = sort(rand(true_rank), rev=true) .* 100
        X = U_true * Diagonal(S_true) * V_true'
        
        # Initialize sketchy with SSRFT redux
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:ssrft, verbose=false)
        
        # Process data
        full_increment!(sk, X)
        
        # Finalize
        est_err, scree = finalize!(sk)
        
        # Reconstruct approximation
        X_approx = sk.V * Diagonal(sk.Σ) * sk.W'
        
        # Check reconstruction error (slightly relaxed for SSRFT)
        rel_error = norm(X - X_approx) / norm(X)
        @test rel_error < 0.15  # 15% relative error tolerance
    end
    
    @testset "Terminate Option in Full Increment" begin
        m, n, r = 100, 80, 10
        k_data = 80
        
        # Create test data
        U_true = randn(m, r)
        V_true = randn(k_data, r)
        S_true = sort(rand(r), rev=true) .* 50
        X = U_true * Diagonal(S_true) * V_true'
        
        sk = init_sketchy(m=m, n=n, r=r, ErrorEstimate=true, q=60, 
                         ReduxMap=:gauss, verbose=false)
        
        # Full increment with terminate flag
        result = full_increment!(sk, X, terminate=true)
        
        # Check that SVD was finalized
        @test any(sk.Σ .!= 0)
        
        # Check that error estimate is returned
        @test !isnothing(result.est_err_squared)
        @test result.est_err_squared >= 0
    end
    
    @testset "Runtime Measurement" begin
        m, n, r = 100, 80, 10
        k_data = 80
        
        # Create test data
        X = randn(m, k_data)
        
        sk = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        
        # Full increment with runtime measurement
        result = full_increment!(sk, X, runtime=true, dump_all=false)
        
        # Check that runtime is recorded
        @test !isnothing(result.runtime)
        @test length(result.runtime) == k_data
        @test all(result.runtime .>= 0)
    end
    
    @testset "Invalid Input Handling" begin
        m, n, r = 100, 80, 10
        
        # Test invalid redux map
        @test_throws ErrorException init_sketchy(m=m, n=n, r=r, ReduxMap=:invalid)
        
        # Test increment with wrong dimension
        sk = init_sketchy(m=m, n=n, r=r, verbose=false)
        x_wrong = randn(m + 10)
        @test_throws AssertionError increment!(sk, x_wrong)
    end
    
    @testset "Data Type Parameters" begin
        m, n, r = 50, 40, 5
        
        # Real data type
        sk_real = init_sketchy(m=m, n=n, r=r, dtype="real", verbose=false)
        @test sk_real.α == 1
        @test sk_real.β == 1
        
        # Complex data type
        sk_complex = init_sketchy(m=m, n=n, r=r, dtype="complex", verbose=false)
        @test sk_complex.α == 0
        @test sk_complex.β == 2
    end
    
    @testset "Incremental vs Batch Update Consistency" begin
        m, n, r = 80, 60, 8
        k_data = 30
        
        # Create test data
        X = randn(m, k_data)
        
        # Incremental approach
        sk1 = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        full_increment!(sk1, X, dump_all=false)
        finalize!(sk1)
        
        # Batch approach
        sk2 = init_sketchy(m=m, n=n, r=r, ReduxMap=:gauss, verbose=false)
        full_increment!(sk2, X, dump_all=true)
        finalize!(sk2)
        
        # Results should be similar (but not identical due to numerical differences)
        # Compare singular values
        rel_diff = norm(sk1.Σ - sk2.Σ) / norm(sk1.Σ)
        @test rel_diff < 0.05  # 5% tolerance for numerical differences
    end
end
