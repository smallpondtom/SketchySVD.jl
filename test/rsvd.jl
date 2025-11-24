@testset "Randomized SVD Tests" begin
    Random.seed!(123)  # For reproducibility
    tol = 1e-6

    # Create a low-rank test matrix
    m, n, true_rank = 1000, 500, 50
    U_true = randn(m, true_rank)
    V_true = randn(n, true_rank)
    S_true = sort(rand(true_rank), rev=true) .* 100
    A = U_true * Diagonal(S_true) * V_true'
    
    @testset "Basic rSVD" begin
        k = 50
        U, S, V = rsvd(A, k, p=10, q=0)
        A_approx = U * Diagonal(S) * V'
        rel_error = norm(A - A_approx) / norm(A)
        
        @test rel_error < tol
        @test size(U) == (m, k)
        @test size(V) == (n, k)
        @test length(S) == k
        @test issorted(S, rev=true)  # Singular values should be sorted
    end
    
    @testset "rSVD with Power Iterations" begin
        k = 50
        U2, S2, V2 = rsvd(A, k, p=10, q=2)
        A_approx2 = U2 * Diagonal(S2) * V2'
        rel_error2 = norm(A - A_approx2) / norm(A)
        
        @test rel_error2 < tol/10  # Expect better accuracy with power iterations
        @test size(U2) == (m, k)
        @test size(V2) == (n, k)
        @test length(S2) == k
    end
    
    @testset "Different Random Matrix Types" begin
        k = 50
        
        # Rademacher
        U3, S3, V3 = rsvd(A, k, p=10, q=1, rng=rademacher_rng)
        A_approx3 = U3 * Diagonal(S3) * V3'
        rel_error3 = norm(A - A_approx3) / norm(A)
        @test rel_error3 < tol
        
        # SRFT
        U4, S4, V4 = rsvd(A, k, p=10, q=1, rng=srft_rng)
        A_approx4 = U4 * Diagonal(S4) * V4'
        rel_error4 = norm(A - A_approx4) / norm(A)
        @test rel_error4 < tol * 2  # Slightly relaxed tolerance
        
        # Uniform
        U5, S5, V5 = rsvd(A, k, p=10, q=1, rng=uniform_rng)
        A_approx5 = U5 * Diagonal(S5) * V5'
        rel_error5 = norm(A - A_approx5) / norm(A)
        @test rel_error5 < tol * 2
        
        # Sparse Gaussian
        U6, S6, V6 = rsvd(A, k, p=10, q=1, rng=sparse_gaussian_rng(0.1))
        A_approx6 = U6 * Diagonal(S6) * V6'
        rel_error6 = norm(A - A_approx6) / norm(A)
        @test rel_error6 < tol * 2

        # Sparse 
        U6, S6, V6 = rsvd(A, k, p=10, q=1, rng=sparse_rng)
        A_approx6 = U6 * Diagonal(S6) * V6'
        rel_error6 = norm(A - A_approx6) / norm(A)
        @test rel_error6 < tol * 2
    end
    
    @testset "rsvd_transpose for Tall-Thin Matrices" begin
        # Create a tall-thin matrix (m >> n)
        m_tall, n_thin, rank = 5000, 100, 30
        U_tall = randn(m_tall, rank)
        V_tall = randn(n_thin, rank)
        S_tall = sort(rand(rank), rev=true) .* 100
        A_tall = U_tall * Diagonal(S_tall) * V_tall'
        
        k = 30
        
        # Test with transpose trick enabled (should use rsvd_transpose internally)
        U_trans, S_trans, V_trans = rsvd(A_tall, k, p=10, q=1, transpose_trick=true)
        A_approx_trans = U_trans * Diagonal(S_trans) * V_trans'
        rel_error_trans = norm(A_tall - A_approx_trans) / norm(A_tall)
        
        @test rel_error_trans < tol
        @test size(U_trans) == (m_tall, k)
        @test size(V_trans) == (n_thin, k)
        @test length(S_trans) == k
        
        # Compare with standard rSVD (transpose trick disabled)
        U_std, S_std, V_std = rsvd(A_tall, k, p=10, q=1, transpose_trick=false)
        
        # Singular values should be very close
        @test norm(S_trans - S_std) / norm(S_std) < 1e-8
        
        # Subspace distances should be small
        subspace_U = norm(U_trans * U_trans' - U_std * U_std') / sqrt(2)
        subspace_V = norm(V_trans * V_trans' - V_std * V_std') / sqrt(2)
        @test subspace_U < 1e-6
        @test subspace_V < 1e-6
    end
    
    @testset "rsvd_transpose Direct Call" begin
        # Create a tall-thin matrix
        m_tall, n_thin, rank = 3000, 80, 25
        U_tall = randn(m_tall, rank)
        V_tall = randn(n_thin, rank)
        S_tall = sort(rand(rank), rev=true) .* 100
        A_tall = U_tall * Diagonal(S_tall) * V_tall'
        
        k = 25
        
        # Direct call to rsvd_transpose
        U_direct, S_direct, V_direct = SketchySVD.rsvd_transpose(A_tall, k, 10, 1, randn)
        A_approx_direct = U_direct * Diagonal(S_direct) * V_direct'
        rel_error_direct = norm(A_tall - A_approx_direct) / norm(A_tall)
        
        @test rel_error_direct < tol
        @test size(U_direct) == (m_tall, k)
        @test size(V_direct) == (n_thin, k)
        @test length(S_direct) == k
        
        # Test orthogonality
        @test norm(U_direct' * U_direct - I) < 1e-10
        @test norm(V_direct' * V_direct - I) < 1e-10
    end
    
    @testset "rsvd_adaptive with Clear Rank" begin
        # Create matrix with clear rank structure
        m_adapt, n_adapt = 800, 600
        true_rank_adapt = 40
        U_adapt = randn(m_adapt, true_rank_adapt)
        V_adapt = randn(n_adapt, true_rank_adapt)
        # Create exponentially decaying singular values
        S_adapt = [100.0 * exp(-0.1 * i) for i in 1:true_rank_adapt]
        A_adapt = U_adapt * Diagonal(S_adapt) * V_adapt'
        
        k_init = 60  # Start with higher rank estimate
        tol_adapt = 1e-3
        
        U_adp, S_adp, V_adp = rsvd_adaptive(A_adapt, k_init, tol=tol_adapt, p=10, q=1)
        
        # Check that rank was correctly identified (should be close to true_rank_adapt)
        detected_rank = length(S_adp)
        @test detected_rank <= true_rank_adapt + 5  # Allow some margin
        @test detected_rank >= true_rank_adapt - 5
        
        # Check reconstruction quality
        A_approx_adp = U_adp * Diagonal(S_adp) * V_adp'
        rel_error_adp = norm(A_adapt - A_approx_adp) / norm(A_adapt)
        @test rel_error_adp < 0.01  # 1% error tolerance
        
        # All singular values should be above threshold
        @test all(S_adp .>= tol_adapt * S_adp[1])
    end
    
    @testset "rsvd_adaptive with No Truncation" begin
        # Matrix where all singular values are significant
        m_adp2, n_adp2, rank_adp2 = 500, 400, 50
        U_adp2 = randn(m_adp2, rank_adp2)
        V_adp2 = randn(n_adp2, rank_adp2)
        # All singular values relatively large
        S_adp2 = rand(rank_adp2) .* 50 .+ 50  # Between 50 and 100
        A_adp2 = U_adp2 * Diagonal(S_adp2) * V_adp2'
        
        k_init2 = 50
        tol_adp2 = 1e-10  # Very strict tolerance
        
        U_adp2_res, S_adp2_res, V_adp2_res = rsvd_adaptive(
            A_adp2, k_init2, tol=tol_adp2, p=5, q=1
        )
        
        # Should keep all k singular values since they're all large
        @test length(S_adp2_res) == k_init2
        
        # Check quality
        A_approx_adp2 = U_adp2_res * Diagonal(S_adp2_res) * V_adp2_res'
        rel_error_adp2 = norm(A_adp2 - A_approx_adp2) / norm(A_adp2)
        @test rel_error_adp2 < 1e-6
    end
    
    @testset "rsvd_adaptive with Aggressive Truncation" begin
        # Matrix with very fast decay
        m_adp3, n_adp3 = 600, 500
        true_rank_adp3 = 20
        U_adp3 = randn(m_adp3, true_rank_adp3)
        V_adp3 = randn(n_adp3, true_rank_adp3)
        # Very fast exponential decay
        S_adp3 = [100.0 * exp(-0.5 * i) for i in 1:true_rank_adp3]
        A_adp3 = U_adp3 * Diagonal(S_adp3) * V_adp3'
        
        k_init3 = 20
        tol_adp3 = 0.1  # Aggressive truncation
        
        U_adp3_res, S_adp3_res, V_adp3_res = rsvd_adaptive(
            A_adp3, k_init3, tol=tol_adp3, p=5, q=1
        )
        
        # Should truncate to much smaller rank
        detected_rank3 = length(S_adp3_res)
        @test detected_rank3 < true_rank_adp3
        @test detected_rank3 >= 5  # But not too aggressive
        
        # All kept singular values should be significant
        if detected_rank3 > 0
            @test all(S_adp3_res .>= tol_adp3 * S_adp3_res[1])
        end
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Test invalid inputs
        @test_throws ArgumentError rsvd(A, 0, p=5)  # k must be positive
        @test_throws ArgumentError rsvd(A, 50, p=-1)  # p must be non-negative
        @test_throws ArgumentError rsvd(A, 50, q=-1)  # q must be non-negative
        @test_throws ArgumentError rsvd(A, 1000, p=5)  # k+p > min(m,n)
        
        # Test with k = min(m,n) - p (boundary case)
        k_boundary = min(m, n) - 10
        U_bound, S_bound, V_bound = rsvd(A, k_boundary, p=10, q=0)
        @test size(U_bound, 2) == k_boundary
        @test size(V_bound, 2) == k_boundary
        @test length(S_bound) == k_boundary
    end
    
    @testset "Orthogonality Properties" begin
        k = 50
        U, S, V = rsvd(A, k, p=10, q=2)
        
        # Check orthonormality of U and V
        @test norm(U' * U - I) < 1e-10
        @test norm(V' * V - I) < 1e-10
        
        # Check that singular values are positive
        @test all(S .> 0)
    end
end