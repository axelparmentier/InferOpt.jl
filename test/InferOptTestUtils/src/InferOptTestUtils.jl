module InferOptTestUtils

using Flux
using Flux.Losses: mse
using Graphs
using GridGraphs
using LinearAlgebra
using Statistics
using Test
using UnicodePlots

include("const.jl")
include("maximizers.jl")
include("dataset.jl")
include("error.jl")
include("loss.jl")
include("perf.jl")
include("pipeline.jl")

export DECREASE, EPOCHS, NB_FEATURES, NB_INSTANCES, NOISE_STD, VERBOSE
export shortest_path_maximizer, max_pricing, g, h
export encoder_factory, generate_dataset
export mse, mape, normalized_mape, hamming_distance, normalized_hamming_distance
export init_perf, update_perf!
export dropfirstdim, make_positive
export PipelineLossImitation, PipelineLossImitationθ, PipelineLossImitationθy
export PipelineLossExperience
export PipelineLossImitationLoss
export test_pipeline!
export mse_kw, identity_kw

end
