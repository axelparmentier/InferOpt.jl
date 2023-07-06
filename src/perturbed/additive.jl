"""
    PerturbedAdditive <: AbstractPerturbed

Differentiable normal perturbation of a black-box maximizer: the input undergoes `θ -> θ + εZ` where `Z ∼ N(0, I)`.

Reference: <https://arxiv.org/abs/2002.08676>

See [`AbstractPerturbed`](@ref) for more details.

# Fields
- `oracle`
- `ε`
- `nb_samples`
- `rng`
- `seed`
- `perturbation`
- `grad_logdensity`
"""
struct PerturbedAdditive{P,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel,G} <:
       AbstractPerturbed{parallel}
    oracle::O
    ε::Float64
    nb_samples::Int
    rng::R
    seed::S
    perturbation::P
    grad_logdensity::G

    function PerturbedAdditive{P,O,R,S,parallel,G}(
        oracle::O,
        ε::Float64,
        nb_samples::Int,
        rng::R,
        seed::S,
        perturbation::P,
        grad_logdensity::G,
    ) where {P,O,R<:AbstractRNG,S<:Union{Nothing,Int},parallel,G}
        @assert parallel isa Bool
        return new{P,O,R,S,parallel,G}(
            oracle, ε, nb_samples, rng, seed, perturbation, grad_logdensity
        )
    end
end

function Base.show(io::IO, perturbed::PerturbedAdditive)
    (; oracle, ε, rng, seed, nb_samples, perturbation) = perturbed
    perturb = isnothing(perturbation) ? "Normal(0, 1)" : "$perturbation"
    return print(
        io, "PerturbedAdditive($oracle, $ε, $nb_samples, $(typeof(rng)), $seed, $perturb)"
    )
end

"""
    PerturbedAdditive(maximizer[; ε=1.0, nb_samples=1])
"""
function PerturbedAdditive(
    oracle::O;
    ε=1.0,
    nb_samples=1,
    rng::R=MersenneTwister(0),
    seed::S=nothing,
    is_parallel::Bool=false,
    perturbation::P=nothing,
    grad_logdensity::G=nothing,
) where {O,R,S,P,G}
    return PerturbedAdditive{P,O,R,S,is_parallel,G}(
        oracle, float(ε), nb_samples, rng, seed, perturbation, grad_logdensity
    )
end

function sample_perturbations(perturbed::PerturbedAdditive, θ::AbstractArray)
    (; rng, seed, nb_samples, ε, perturbation) = perturbed
    seed!(rng, seed)
    return [θ .+ ε .* rand(rng, perturbation, size(θ)) for _ in 1:nb_samples]
end

function sample_perturbations(perturbed::PerturbedAdditive{Nothing}, θ::AbstractArray)
    (; rng, seed, nb_samples, ε) = perturbed
    seed!(rng, seed)
    return [θ .+ ε .* randn(rng, size(θ)) for _ in 1:nb_samples]
end

function perturbation_grad_logdensity(
    ::RuleConfig, perturbed::PerturbedAdditive{Nothing}, θ::AbstractArray, η::AbstractArray
)
    (; ε) = perturbed
    return (η .- θ) ./ ε
end
