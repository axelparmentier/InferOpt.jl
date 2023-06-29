"""
    AbstractPerturbed{F, parallel}

Differentiable perturbation of a black box optimizer of type `F`.
The parameter `parallel` is a boolean value, equal to true if the perturbations are run in parallel.

# Applicable functions

- [`compute_probability_distribution(perturbed::AbstractPerturbed, θ)`](@ref)
- `(perturbed::AbstractPerturbed)(θ)`

# Available subtypes

- [`PerturbedAdditive`](@ref)
- [`PerturbedMultiplicative`](@ref)

These subtypes share the following fields:

- `maximizer::F`: black box optimizer
- `ε::Float64`: magnitude of the perturbation
- `nb_samples::Int`: number of random samples for Monte-Carlo computations
- `rng::AbstractRNG`: random number generator
- `seed::Union{Nothing,Int}`: random seed
"""
abstract type AbstractPerturbed{F,parallel} end

"""
    sample_perturbations(perturbed::AbstractPerturbed, θ)

Draw random perturbations `Z` which will be applied to the objective direction `θ`.
"""
function sample_perturbations(perturbed::AbstractPerturbed, θ::AbstractArray)
    (; rng, seed, nb_samples) = perturbed
    seed!(rng, seed)
    Z_samples = [randn(rng, size(θ)) for _ in 1:nb_samples]
    return Z_samples
end

# Non parallelized version
function compute_atoms(
    perturbed::AbstractPerturbed{F,false},
    θ::AbstractArray,
    Z_samples::Vector{<:AbstractArray};
    kwargs...,
) where {F}
    return [perturb_and_optimize(perturbed, θ, Z; kwargs...) for Z in Z_samples]
end

# Parallelized version
function compute_atoms(
    perturbed::AbstractPerturbed{F,true},
    θ::AbstractArray,
    Z_samples::Vector{<:AbstractArray};
    kwargs...,
) where {F}
    return ThreadsX.map(Z -> perturb_and_optimize(perturbed, θ, Z; kwargs...), Z_samples)
end

function compute_probability_distribution(
    perturbed::AbstractPerturbed,
    θ::AbstractArray,
    Z_samples::Vector{<:AbstractArray};
    kwargs...,
)
    atoms = compute_atoms(perturbed, θ, Z_samples; kwargs...)
    weights = ones(length(atoms)) ./ length(atoms)
    probadist = FixedAtomsProbabilityDistribution(atoms, weights)
    return probadist
end

"""
    compute_probability_distribution(perturbed::AbstractPerturbed, θ)

Turn random perturbations of `θ` into a distribution on polytope vertices.
"""
function compute_probability_distribution(
    perturbed::AbstractPerturbed, θ::AbstractArray; kwargs...
)
    Z_samples = sample_perturbations(perturbed, θ)
    return compute_probability_distribution(perturbed, θ, Z_samples; kwargs...)
end

"""
    (perturbed::AbstractPerturbed)(θ)

Apply `compute_probability_distribution(perturbed, θ)` and return the expectation.
"""
function (perturbed::AbstractPerturbed)(θ::AbstractArray; kwargs...)
    probadist = compute_probability_distribution(perturbed, θ; kwargs...)
    return compute_expectation(probadist)
end
