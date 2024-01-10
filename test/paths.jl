@testitem "Paths - imit - SPO+ (θ)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθ();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=SPOPlusLoss(shortest_path_maximizer),
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - SPO+ (θ & y)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθy();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=SPOPlusLoss(shortest_path_maximizer),
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - MSE IdentityRelaxation" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=normalize ∘ IdentityRelaxation(shortest_path_maximizer),
        loss=mse_kw,
        error_function=mse_kw,
    )
end

# @testitem "Paths - imit - MSE Interpolation" default_imports = false begin
#     include("InferOptTestUtils/src/InferOptTestUtils.jl")
#     using InferOpt, .InferOptTestUtils, Random
#     Random.seed!(63)

#     test_pipeline!(
#         PipelineLossImitation;
#         instance_dim=(5, 5),
#         true_maximizer=shortest_path_maximizer,
#         maximizer=Interpolation(shortest_path_maximizer; λ=5.0),
#         loss=mse_kw,
#         error_function=mse_kw,
#     )
# end  # TODO: make it work (doesn't seem to depend on λ)

@testitem "Paths - imit - MSE PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=PerturbedAdditive(shortest_path_maximizer; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - MSE PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=PerturbedMultiplicative(shortest_path_maximizer; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - MSE RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=RegularizedFrankWolfe(
            shortest_path_maximizer;
            Ω=half_square_norm,
            Ω_grad=identity_kw,
            frank_wolfe_kwargs=(;
                max_iteration=10, line_search=FrankWolfe.Adaptive(; relaxed_smoothness=true)
            ),
        ),
        loss=mse_kw,
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - FYL PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            PerturbedAdditive(shortest_path_maximizer; ε=1.0, nb_samples=5)
        ),
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - FYL PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            PerturbedMultiplicative(shortest_path_maximizer; ε=1.0, nb_samples=5)
        ),
        error_function=mse_kw,
        epochs=100,
    )
end

@testitem "Paths - imit - FYL PerturbedAdditive{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Distributions
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            PerturbedAdditive(
                shortest_path_maximizer; ε=1.0, nb_samples=5, perturbation=LogNormal(0, 1)
            ),
        ),
        error_function=mse_kw,
    )
end

@testitem "Paths - imit - FYL RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            RegularizedFrankWolfe(
                shortest_path_maximizer;
                Ω=half_square_norm,
                Ω_grad=identity_kw,
                frank_wolfe_kwargs=(;
                    max_iteration=10,
                    line_search=FrankWolfe.Adaptive(; relaxed_smoothness=true),
                ),
            ),
        ),
        error_function=mse_kw,
        epochs=100,
    )
end

@testitem "Paths - exp - Pushforward PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedAdditive(shortest_path_maximizer; ε=1.0, nb_samples=10), cost
        ),
        error_function=mse_kw,
        true_encoder=true_encoder,
        cost=cost,
        epochs=500,
    )
end

@testitem "Paths - exp - Pushforward PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedMultiplicative(shortest_path_maximizer; ε=1.0, nb_samples=10), cost
        ),
        error_function=mse_kw,
        true_encoder=true_encoder,
        cost=cost,
        epochs=500,
    )
end

@testitem "Paths - exp - Pushforward RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe,
        FrankWolfe, InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=(5, 5),
        true_maximizer=shortest_path_maximizer,
        maximizer=identity_kw,
        loss=Pushforward(
            RegularizedFrankWolfe(
                shortest_path_maximizer;
                Ω=half_square_norm,
                Ω_grad=identity_kw,
                frank_wolfe_kwargs=(;
                    max_iteration=10,
                    line_search=FrankWolfe.Adaptive(; relaxed_smoothness=true),
                ),
            ),
            cost,
        ),
        error_function=mse_kw,
        true_encoder=true_encoder,
        cost=cost,
        epochs=200,
    )
end
