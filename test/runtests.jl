using SketchySVD
using Test
using LinearAlgebra
using Random
using SparseArrays

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

@testset "SketchySVD" begin
    testfile("rsvd.jl", "Randomized SVD")
    testfile("dimredux.jl", "Dimension Reduction Maps")
    testfile("sketchy.jl", "Sketchy Algorithm")
    testfile("integration.jl", "Integration Tests")
end