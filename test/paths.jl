using Flux
using Graphs
using GridGraphs
using InferOpt
using LinearAlgebra
using Random
using Test

Random.seed!(63)

## Main functions

nb_features = 5
encoder_factory() = Chain(Dense(nb_features, 1), dropfirstdim, make_positive)
true_encoder = encoder_factory()
cost(y; instance) = dot(y, -true_encoder(instance))
error_function(ŷ, y) = half_square_norm(ŷ - y)

function true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
    g = AcyclicGridGraph{Int,R}(-θ)
    path = grid_topological_sort(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

## Pipelines

pipelines_imitation_θ = [
    # SPO+
    (encoder=encoder_factory(), maximizer=identity, loss=SPOPlusLoss(true_maximizer)),
]

pipelines_imitation_y = [
    # Perturbed + FYL
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=5)),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=5)),
    ),
    # Perturbed + other loss
    (
        encoder=encoder_factory(),
        maximizer=PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=10),
        loss=Flux.Losses.mse,
    ),
    (
        encoder=encoder_factory(),
        maximizer=PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10),
        loss=Flux.Losses.mse,
    ),
    # Generic regularized + FYL
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=FenchelYoungLoss(
            RegularizedGeneric(true_maximizer, half_square_norm, identity)
        ),
    ),
    # Generic regularized + other loss
    (
        encoder=encoder_factory(),
        maximizer=RegularizedGeneric(true_maximizer, half_square_norm, identity),
        loss=Flux.Losses.mse,
    ),
]

pipelines_experience = [
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=ProbabilisticComposition(
            PerturbedAdditive(true_maximizer; ε=1.0, nb_samples=10), cost
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=ProbabilisticComposition(
            PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10), cost
        ),
    ),
    (
        encoder=encoder_factory(),
        maximizer=identity,
        loss=ProbabilisticComposition(
            RegularizedGeneric(true_maximizer, half_square_norm, identity), cost
        ),
    ),
]

## Dataset generation

data_train, data_test = generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features=nb_features,
    instance_dim=(10, 10),
    nb_instances=100,
    noise_std=0.01,
);

## Test loop

for pipeline in pipelines_imitation_θ
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_imitation_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    test_pipeline!(
        pipeline,
        pipeline_loss_imitation_θ;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=100,
        verbose=true,
        setting_name="paths - imitation_θ",
    )
end

for pipeline in pipelines_imitation_y
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    test_pipeline!(
        pipeline,
        pipeline_loss_imitation_y;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=200,
        verbose=true,
        setting_name="paths - imitation_y",
    )
end

for pipeline in pipelines_experience
    pipeline = deepcopy(pipeline)
    (; encoder, maximizer, loss) = pipeline
    pipeline_loss_experience(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
    test_pipeline!(
        pipeline,
        pipeline_loss_experience;
        true_encoder=true_encoder,
        true_maximizer=true_maximizer,
        data_train=data_train,
        data_test=data_test,
        error_function=error_function,
        cost=cost,
        epochs=1000,
        verbose=true,
        setting_name="paths - experience",
    )
end
