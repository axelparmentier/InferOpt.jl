using Revise
using Aqua
using InferOpt
using JuliaFormatter
using Test

format(InferOpt; verbose=true)

includet("utils/dataset.jl")
includet("utils/error.jl")
includet("utils/perf.jl")
includet("utils/pipeline.jl")

@testset verbose = true "InferOpt.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(InferOpt; deps_compat=true, project_extras=true, ambiguities=false)
    end
    @testset verbose = true "Jacobian approx" begin
        include("jacobian_approx.jl")
    end
    @testset verbose = true "Argmax" begin
        include("argmax.jl")
    end
    @testset verbose = true "Ranking" begin
        include("ranking.jl")
    end
    @testset verbose = true "Paths" begin
        include("paths.jl")
    end
    @testset verbose = true "Frank-Wolfe" begin
        include("frank_wolfe.jl")
    end
    @testset verbose = true "Tutorial" begin
        include("tutorial.jl")
    end
end
