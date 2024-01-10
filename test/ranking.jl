@testitem "Ranking - imit - SPO+ (θ)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθ();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - SPO+ (θ & y)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθy();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=SPOPlusLoss(ranking),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE IdentityRelaxation" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=normalize ∘ IdentityRelaxation(ranking),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE Interpolation" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=Interpolation(ranking; λ=5.0),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedAdditive(ranking; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=PerturbedMultiplicative(ranking; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - MSE RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=RegularizedFrankWolfe(
            ranking;
            Ω=half_square_norm,
            Ω_grad=identity_kw,
            frank_wolfe_kwargs=(;
                max_iteration=10, line_search=FrankWolfe.Adaptive(; relaxed_smoothness=true)
            ),
        ),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - FYL PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(PerturbedAdditive(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - FYL PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(PerturbedMultiplicative(ranking; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - FYL PerturbedAdditive{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random, Distributions, LinearAlgebra
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            PerturbedAdditive(ranking; ε=1.0, nb_samples=5, perturbation=LogNormal(0, 1))
        ),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - imit - FYL RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            RegularizedFrankWolfe(
                ranking;
                Ω=half_square_norm,
                Ω_grad=identity_kw,
                frank_wolfe_kwargs=(;
                    max_iteration=10,
                    line_search=FrankWolfe.Adaptive(; relaxed_smoothness=true),
                ),
            ),
        ),
        error_function=hamming_distance,
    )
end

@testitem "Ranking - exp - Pushforward PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(PerturbedAdditive(ranking; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Ranking - exp - Pushforward PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(PerturbedMultiplicative(ranking; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Ranking - exp - Pushforward PerturbedAdditive{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Distributions
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedAdditive(ranking; ε=1.0, nb_samples=10, perturbation=LogNormal(0, 1)),
            cost,
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=500,
    )
end

@testitem "Ranking - exp - Pushforward PerturbedMultiplicative{LogNormal}" default_imports =
    false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Distributions
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedMultiplicative(
                ranking; ε=1.0, nb_samples=10, perturbation=LogNormal(0, 1)
            ),
            cost,
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
        epochs=500,
    )
end

@testitem "Ranking - exp - Pushforward PerturbedOracle{LogNormal}" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random, Distributions
    Random.seed!(63)

    p(θ) = MvLogNormal(θ, I)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(PerturbedOracle(ranking, p; nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Ranking - exp - Pushforward RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe,
        FrankWolfe, InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=ranking,
        maximizer=identity_kw,
        loss=Pushforward(
            RegularizedFrankWolfe(
                ranking;
                Ω=half_square_norm,
                Ω_grad=identity_kw,
                frank_wolfe_kwargs=(;
                    max_iteration=10,
                    line_search=FrankWolfe.Adaptive(; relaxed_smoothness=true),
                ),
            ),
            cost,
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end
