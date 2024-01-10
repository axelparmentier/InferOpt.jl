@testitem "Argmax - imit - SPO+ (θ)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθ();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=SPOPlusLoss(one_hot_argmax),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - SPO+ (θ & y)" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitationθy();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=SPOPlusLoss(one_hot_argmax),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - SSVM" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=InferOpt.ZeroOneStructuredSVMLoss(),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - MSE SparseArgmax" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=SparseArgmax(),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - MSE SoftArgmax" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=SoftArgmax(),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - MSE PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - MSE PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=10),
        loss=mse_kw,
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - MSE RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=RegularizedFrankWolfe(
            one_hot_argmax;
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

@testitem "Argmax - imit - FYL SparseArgmax" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(SparseArgmax()),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - FYL SoftArgmax" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(SoftArgmax()),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - FYL PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - FYL PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=5)),
        error_function=hamming_distance,
    )
end

@testitem "Argmax - imit - FYL RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe, FrankWolfe, InferOpt, .InferOptTestUtils, Random
    Random.seed!(63)

    test_pipeline!(
        PipelineLossImitation();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=FenchelYoungLoss(
            RegularizedFrankWolfe(
                one_hot_argmax;
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

@testitem "Argmax - exp - Pushforward PerturbedAdditive" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=Pushforward(PerturbedAdditive(one_hot_argmax; ε=1.0, nb_samples=10), cost),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Argmax - exp - Pushforward PerturbedMultiplicative" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=Pushforward(
            PerturbedMultiplicative(one_hot_argmax; ε=1.0, nb_samples=10), cost
        ),
        error_function=hamming_distance,
        true_encoder=true_encoder,
        cost=cost,
    )
end

@testitem "Argmax - exp - Pushforward RegularizedFrankWolfe" default_imports = false begin
    include("InferOptTestUtils/src/InferOptTestUtils.jl")
    using DifferentiableFrankWolfe,
        FrankWolfe, InferOpt, .InferOptTestUtils, LinearAlgebra, Random
    Random.seed!(63)

    true_encoder = encoder_factory()
    cost(y; instance) = dot(y, -true_encoder(instance))
    test_pipeline!(
        PipelineLossExperience();
        instance_dim=5,
        true_maximizer=one_hot_argmax,
        maximizer=identity_kw,
        loss=Pushforward(
            RegularizedFrankWolfe(
                one_hot_argmax;
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
