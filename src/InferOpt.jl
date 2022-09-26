module InferOpt

using ChainRulesCore
using Krylov: gmres
using LinearAlgebra
using LinearOperators: LinearOperator
using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister, rand, seed!
using Requires
using SimpleTraits: SimpleTraits
using SimpleTraits: @traitdef, @traitfn, @traitimpl
using SparseArrays
using Statistics
using StatsBase: StatsBase, sample
using Test
using ThreadsX

function __init__()
    @require FrankWolfe="f55ce6ea-fdc5-4628-88c5-0087fe54bd30" include("fw.jl")
end

include("utils/probability_distribution.jl")
include("utils/pushforward.jl")

include("interpolation/interpolation.jl")

include("regularized/isregularized.jl")
include("regularized/regularized_utils.jl")
include("regularized/soft_argmax.jl")
include("regularized/sparse_argmax.jl")
include("regularized/regularized_generic.jl")

include("perturbed/generalized_maximizer.jl")
include("perturbed/abstract_perturbed.jl")
include("perturbed/additive.jl")
include("perturbed/multiplicative.jl")

include("fenchel_young/perturbed.jl")
include("fenchel_young/fenchel_young.jl")

include("spo/spoplus_loss.jl")

include("ssvm/isbaseloss.jl")
include("ssvm/zeroone_baseloss.jl")
include("ssvm/ssvm_loss.jl")

export FixedAtomsProbabilityDistribution
export compute_expectation, compress_distribution!
export Pushforward
export compute_probability_distribution

export Interpolation

export DifferentiableFrankWolfe
export LMOWrapper

export half_square_norm
export shannon_entropy, negative_shannon_entropy
export one_hot_argmax, ranking
export IsRegularized
export soft_argmax, sparse_argmax
export RegularizedGeneric

export PerturbedAdditive
export PerturbedMultiplicative

export FenchelYoungLoss

export SPOPlusLoss

export IsBaseLoss
export ZeroOneBaseLoss
export StructuredSVMLoss

export GeneralizedMaximizer, objective_value

end
