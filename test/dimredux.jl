@testset "Dimension Redux Tests" begin
    Random.seed!(789)
    
    @testset "Gauss - Basic Construction" begin
        k, n = 50, 100
        
        # Real field
        G_real = SketchySVD.Gauss(k, n, field="real")
        @test size(G_real) == (k, n)
        @test G_real.k == k
        @test G_real.n == n
        @test G_real.field == "real"
        @test isreal(G_real)
        @test !issparse(G_real)
        @test !G_real.transposeFlag
        @test eltype(G_real.Xi) <: Real
        
        # Complex field
        G_complex = SketchySVD.Gauss(k, n, field="complex")
        @test size(G_complex) == (k, n)
        @test G_complex.field == "complex"
        @test !isreal(G_complex)
        @test eltype(G_complex.Xi) <: Complex
        
        # Invalid field
        @test_throws ErrorException SketchySVD.Gauss(k, n, field="invalid")
    end
    
    @testset "Gauss - Transpose and Adjoint" begin
        k, n = 30, 50
        G = SketchySVD.Gauss(k, n, field="real")
        
        # Test adjoint
        G_adj = G'
        @test size(G_adj) == (n, k)  # Dimensions are swapped
        @test G_adj.transposeFlag == true
        @test !G.transposeFlag  # Original should be unchanged
        
        # Double transpose
        G_adj_adj = G_adj'
        @test size(G_adj_adj) == (k, n)
        @test G_adj_adj.transposeFlag == false
    end
    
    @testset "Gauss - Left Apply" begin
        k, n, p = 30, 50, 20
        G = SketchySVD.Gauss(k, n, field="real")
        A = randn(n, p)
        
        # Left apply: G * A should give (k, p) matrix
        result = G * A
        @test size(result) == (k, p)
        
        # Check consistency with explicit multiplication
        expected = G.Xi * A
        @test norm(result - expected) < 1e-10
        
        # Test with vector
        v = randn(n)
        result_v = G * v
        @test length(result_v) == k
    end
    
    @testset "Gauss - Right Apply" begin
        k, n, p = 30, 50, 20
        G = SketchySVD.Gauss(k, n, field="real")
        A = randn(p, k)
        
        # Right apply: A * G should give (p, n) matrix
        result = A * G
        @test size(result) == (p, n)
    end
    
    @testset "Gauss - Indexing" begin
        k, n = 10, 20
        G = SketchySVD.Gauss(k, n, field="real")
        
        # Get row
        row = G[2, :]
        @test length(row) == n
        @test row ≈ G.Xi[2, :]
        
        # Get column
        col = G[:, 3]
        @test length(col) == k
        @test col ≈ G.Xi[:, 3]
        
        # Submatrix indexing
        submat = G[1:5, 1:10]
        @test size(submat) == (5, 10)
        @test submat ≈ G.Xi[1:5, 1:10]
    end
    
    @testset "Gauss - Copy and Redraw" begin
        k, n = 20, 30
        G = SketchySVD.Gauss(k, n, field="real")
        
        # Copy
        G_copy = copy(G)
        @test G_copy.k == G.k
        @test G_copy.n == G.n
        @test G_copy.field == G.field
        
        # Redraw (should create new random matrix)
        G_new = SketchySVD.redraw(G)
        @test size(G_new) == size(G)
        @test G_new.field == G.field
        @test norm(G_new.Xi - G.Xi) > 0  # Should be different
        
        # Redraw with different size
        G_new2 = SketchySVD.redraw(G, 15, 25)
        @test size(G_new2) == (15, 25)
    end
    
    @testset "Sparse - Basic Construction" begin
        k, n = 50, 100
        zeta = 8
        
        # Real field
        S_real = SketchySVD.Sparse(k, n, zeta=zeta, field="real")
        @test size(S_real) == (k, n)
        @test S_real.k == k
        @test S_real.n == n
        @test S_real.field == "real"
        @test isreal(S_real)
        @test issparse(S_real)
        @test !S_real.transposeFlag
        
        # Check sparsity
        nnz_val = nnz(S_real)
        expected_nnz = n * zeta
        @test nnz_val == expected_nnz
        
        # Complex field
        S_complex = SketchySVD.Sparse(k, n, zeta=zeta, field="complex")
        @test S_complex.field == "complex"
        @test !isreal(S_complex)
        
        # Invalid zeta
        @test_throws ErrorException SketchySVD.Sparse(k, n, zeta=0)
        @test_throws ErrorException SketchySVD.Sparse(k, n, zeta=k+1)
    end
    
    @testset "Sparse - Entry Values" begin
        k, n, zeta = 50, 100, 8
        S = SketchySVD.Sparse(k, n, zeta=zeta, field="real")
        
        # Check that entries are ±1
        for val in S.Xi.nzval
            @test abs(val) ≈ 1.0
            @test val ∈ [-1.0, 1.0]
        end
    end
    
    @testset "Sparse - Left Apply" begin
        k, n, p = 30, 50, 20
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        A = randn(n, p)
        
        # Left apply
        result = S * A
        @test size(result) == (k, p)
        
        # Check consistency
        expected = S.Xi * A
        @test norm(result - expected) < 1e-10
        
        # Test with vector
        v = randn(n)
        result_v = S * v
        @test length(result_v) == k
    end
    
    @testset "Sparse - Right Apply" begin
        k, n, p = 30, 50, 20
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        A = randn(p, k)
        
        # Right apply
        result = A * S
        @test size(result) == (p, n)
    end
    
    @testset "Sparse - Transpose" begin
        k, n = 30, 50
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        
        S_adj = S'
        @test size(S_adj) == (n, k)
        @test S_adj.transposeFlag == true
    end
    
    @testset "Sparse - Indexing" begin
        k, n = 20, 30
        S = SketchySVD.Sparse(k, n, zeta=6, field="real")
        
        # Get row
        row = S[2, :]
        @test length(row) == n
        
        # Get column
        col = S[:, 3]
        @test length(col) == k
    end

    @testset "Sparse - Copy and Redraw" begin
        k, n = 20, 40
        R = SketchySVD.Sparse(k, n, field="real")
        
        # Copy
        R_copy = copy(R)
        @test R_copy.k == R.k
        @test R_copy.n == R.n
        @test R_copy.field == R.field
        
        # Redraw
        R_new = SketchySVD.redraw(R)
        @test size(R_new) == size(R)
    end
    
    @testset "SSRFT - Basic Construction" begin
        k, n = 50, 100
        
        # Real field
        R_real = SketchySVD.SSRFT(k, n, field="real")
        @test size(R_real) == (k, n)
        @test R_real.k == k
        @test R_real.n == n
        @test R_real.field == "real"
        @test isreal(R_real)
        @test !issparse(R_real)
        @test !R_real.transposeFlag
        
        # Check coordinate length
        @test length(R_real.coords) == k
        @test all(1 .<= R_real.coords .<= n)
        @test length(unique(R_real.coords)) == k  # All unique
        
        # Complex field
        R_complex = SketchySVD.SSRFT(k, n, field="complex")
        @test R_complex.field == "complex"
        @test !isreal(R_complex)
    end
    
    @testset "SSRFT - Permutation Matrices" begin
        k, n = 30, 60
        R = SketchySVD.SSRFT(k, n, field="real")
        
        # Check Pi1 and Pi2 properties
        @test size(R.Pi1) == (n, n)
        @test size(R.Pi2) == (n, n)
        @test issparse(R.Pi1)
        @test issparse(R.Pi2)
        
        # Each should have exactly n non-zero entries
        @test nnz(R.Pi1) == n
        @test nnz(R.Pi2) == n
        
        # Values should be ±1
        for val in R.Pi1.nzval
            @test abs(val) ≈ 1.0
        end
        for val in R.Pi2.nzval
            @test abs(val) ≈ 1.0
        end
    end
    
    @testset "SSRFT - Left Apply" begin
        k, n, p = 20, 40, 15
        R = SketchySVD.SSRFT(k, n, field="real")
        A = randn(n, p)
        
        # Left apply
        result = R * A
        @test size(result) == (k, p)
        
        # Result should be real for real input
        @test eltype(result) <: Real
        
        # Test with vector
        v = randn(n)
        result_v = R * v
        @test length(result_v) == k
        @test eltype(result_v) <: Real
    end
    
    @testset "SSRFT - Right Apply" begin
        k, n, p = 20, 40, 15
        R = SketchySVD.SSRFT(k, n, field="real")
        A = randn(p, k)
        
        # Right apply
        result = A * R
        @test size(result) == (p, n)
    end
    
    @testset "SSRFT - Transpose" begin
        k, n = 20, 40
        R = SketchySVD.SSRFT(k, n, field="real")
        
        R_adj = R'
        @test size(R_adj) == (n, k)
        @test R_adj.transposeFlag == true
    end
    
    @testset "SSRFT - Copy and Redraw" begin
        k, n = 20, 40
        R = SketchySVD.SSRFT(k, n, field="real")
        
        # Copy
        R_copy = copy(R)
        @test R_copy.k == R.k
        @test R_copy.n == R.n
        @test R_copy.field == R.field
        
        # Redraw
        R_new = SketchySVD.redraw(R)
        @test size(R_new) == size(R)
        @test R_new.coords != R.coords  # Should have different sampling
    end
    
    @testset "DimRedux Type Hierarchy" begin
        k, n = 30, 50
        
        G = SketchySVD.Gauss(k, n, field="real")
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        R = SketchySVD.SSRFT(k, n, field="real")
        
        @test G isa SketchySVD.DimRedux
        @test S isa SketchySVD.DimRedux
        @test R isa SketchySVD.DimRedux
    end
    
    @testset "Multiplication Consistency Across Types" begin
        k, n, p = 30, 50, 20
        A = randn(n, p)
        
        # Create all three types
        G = SketchySVD.Gauss(k, n, field="real")
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        R = SketchySVD.SSRFT(k, n, field="real")
        
        # All should produce same-sized output
        result_G = G * A
        result_S = S * A
        result_R = R * A
        
        @test size(result_G) == (k, p)
        @test size(result_S) == (k, p)
        @test size(result_R) == (k, p)
    end
    
    @testset "Large Scale Performance - Gauss" begin
        k, n = 100, 500
        G = SketchySVD.Gauss(k, n, field="real")
        A = randn(n, 100)
        
        # Should complete without error
        result = G * A
        @test size(result) == (k, 100)
    end
    
    @testset "Large Scale Performance - Sparse" begin
        k, n = 100, 500
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        A = randn(n, 100)
        
        # Should complete without error and be faster than dense
        result = S * A
        @test size(result) == (k, 100)
    end
    
    @testset "Large Scale Performance - SSRFT" begin
        k, n = 100, 500
        R = SketchySVD.SSRFT(k, n, field="real")
        A = randn(n, 100)
        
        # Should complete without error
        result = R * A
        @test size(result) == (k, 100)
    end
    
    @testset "In-place Operations - Gauss" begin
        k, n, p = 30, 50, 20
        G = SketchySVD.Gauss(k, n, field="real")
        A = randn(n, p)
        C = zeros(k, p)
        
        # Test LeftApply!
        SketchySVD.LeftApply!(G, A, C)
        @test any(C .!= 0)
        @test size(C) == (k, p)
        
        # Compare with non-in-place
        C_expected = G * A
        @test norm(C - C_expected) < 1e-10
    end
    
    @testset "In-place Operations - Sparse" begin
        k, n, p = 30, 50, 20
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        A = randn(n, p)
        C = zeros(k, p)
        
        # Test LeftApply!
        SketchySVD.LeftApply!(S, A, C)
        @test any(C .!= 0)
        @test size(C) == (k, p)
    end
    
    @testset "In-place Operations - SSRFT" begin
        k, n, p = 30, 50, 20
        R = SketchySVD.SSRFT(k, n, field="real")
        A = randn(n, p)
        C = zeros(k, p)
        
        # Test LeftApply!
        SketchySVD.LeftApply!(R, A, C)
        @test all(C .≈ 0.0)
        @test size(C) == (k, p)
    end
    
    @testset "Complex Field Operations" begin
        k, n, p = 20, 30, 15
        
        # Gauss with complex
        G_c = SketchySVD.Gauss(k, n, field="complex")
        A_c = randn(ComplexF64, n, p)
        result_G = G_c * A_c
        @test size(result_G) == (k, p)
        @test eltype(result_G) <: Complex
        
        # Sparse with complex
        S_c = SketchySVD.Sparse(k, n, zeta=8, field="complex")
        result_S = S_c * A_c
        @test size(result_S) == (k, p)
        
        # SSRFT with complex
        R_c = SketchySVD.SSRFT(k, n, field="complex")
        result_R = R_c * A_c
        @test size(result_R) == (k, p)
    end
    
    @testset "Edge Cases - Small Dimensions" begin
        k, n = 2, 5
        
        G = SketchySVD.Gauss(k, n, field="real")
        S = SketchySVD.Sparse(k, n, zeta=2, field="real")
        R = SketchySVD.SSRFT(k, n, field="real")
        
        A = randn(n, 3)
        
        @test size(G * A) == (k, 3)
        @test size(S * A) == (k, 3)
        @test size(R * A) == (k, 3)
    end
    
    @testset "View Operations - Gauss" begin
        k, n = 20, 30
        G = SketchySVD.Gauss(k, n, field="real")
        
        # Create a view
        G_view = view(G, 1:10, 1:15)
        @test size(G_view) == (10, 15)
    end
    
    @testset "View Operations - Sparse" begin
        k, n = 20, 30
        S = SketchySVD.Sparse(k, n, zeta=6, field="real")
        
        # Create a view
        S_view = view(S, 1:10, 1:15)
        @test size(S_view) == (10, 15)
    end
    
    @testset "mul! with α and β parameters" begin
        k, n, p = 20, 30, 15
        
        # Test Gauss
        G = SketchySVD.Gauss(k, n, field="real")
        B = randn(n, p)
        C = randn(k, p)
        C_orig = copy(C)
        
        α, β = 2.0, 0.5
        mul!(C, G, B, α, β)
        
        # Check: C = α*G*B + β*C_orig
        expected = α * (G * B) + β * C_orig
        @test norm(C - expected) < 1e-8
        
        # Test Sparse
        S = SketchySVD.Sparse(k, n, zeta=8, field="real")
        C2 = copy(C_orig)
        mul!(C2, S, B, α, β)
        @test size(C2) == (k, p)
    end
end
