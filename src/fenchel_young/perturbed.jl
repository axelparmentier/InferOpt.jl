function fenchel_young_F_and_first_part_of_grad(
    perturbed::AbstractPerturbed, θ::AbstractArray{<:Real}; kwargs...
)
    Z_samples = sample_perturbations(perturbed, θ)
    F_and_y_samples = ThreadsX.map(
        Z -> fenchel_young_F_and_first_part_of_grad(perturbed, θ, Z; kwargs...), Z_samples
    )
    return mean(first, F_and_y_samples), mean(last, F_and_y_samples)
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedAdditive,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .+ ε .* Z
    y = maximizer(θ_perturbed; kwargs...)
    F = dot(θ_perturbed, y)
    return F, y
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedAdditive{<:GeneralizedMaximizer},
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    θ_perturbed = θ .+ ε .* Z
    y = maximizer(θ_perturbed; kwargs...)
    F = objective_value(maximizer, θ_perturbed, y; kwargs...)
    return F, y
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedMultiplicative,
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    eZ = exp.(ε .* Z .- ε^2)
    θ_perturbed = θ .* eZ
    y = maximizer(θ_perturbed; kwargs...)
    F = dot(θ_perturbed, y)
    y_scaled = y .* eZ
    return F, y_scaled
end

function fenchel_young_F_and_first_part_of_grad(
    perturbed::PerturbedMultiplicative{<:GeneralizedMaximizer},
    θ::AbstractArray{<:Real},
    Z::AbstractArray{<:Real};
    kwargs...,
)
    (; maximizer, ε) = perturbed
    eZ = exp.(ε .* Z .- ε^2)
    θ_perturbed = θ .* eZ
    y = maximizer(θ_perturbed; kwargs...)
    F = objective_value(maximizer, θ_perturbed, y; kwargs...)
    y_scaled = y .* eZ
    return F, y_scaled
end
