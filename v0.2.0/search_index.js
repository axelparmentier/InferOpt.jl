var documenterSearchIndex = {"docs":
[{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"EditURL = \"https://github.com/axelparmentier/InferOpt.jl/blob/main/test/tutorial.jl\"","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/#Context","page":"Tutorial","title":"Context","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Let us imagine that we observe the itineraries chosen by a public transport user in several different networks, and that we want to understand their decision-making process (a.k.a. recover their utility function).","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"More precisely, each point in our dataset consists in:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"a graph G\na shortest path P from the top left to the bottom right corner","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We don't know the true costs that were used to compute the shortest path, but we can exploit a set of features to approximate these costs. The question is: how should we combine these features?","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We will use InferOpt to learn the appropriate weights, so that we may propose relevant paths to the user in the future.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using Flux\nusing Graphs\nusing GridGraphs\nusing InferOpt\nusing LinearAlgebra\nusing ProgressMeter\nusing Random\nusing Statistics\nusing Test\nusing UnicodePlots\n\nRandom.seed!(63);\nnothing #hide","category":"page"},{"location":"tutorial/#Grid-graphs","page":"Tutorial","title":"Grid graphs","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"For the purposes of this tutorial, we consider grid graphs, as implemented in GridGraphs.jl. In such graphs, each vertex corresponds to a couple of coordinates (i j), where 1 leq i leq h and 1 leq j leq w.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"To ensure acyclicity, we only allow the user to move right, down or both. Since the cost of a move is defined as the cost of the arrival vertex, any grid graph is entirely characterized by its cost matrix theta in mathbbR^h times w.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"h, w = 50, 100\ng = AcyclicGridGraph(rand(h, w));\nnothing #hide","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"For convenience, GridGraphs.jl also provides custom functions to compute shortest paths efficiently. Let us see what those paths look like.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"p = path_to_matrix(g, grid_topological_sort(g, 1, nv(g)));\nspy(p)","category":"page"},{"location":"tutorial/#Dataset","page":"Tutorial","title":"Dataset","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"As announced, we do not know the cost of each vertex, only a set of relevant features. Let us assume that the user combines them using a shallow neural network.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"nb_features = 5\n\ntrue_encoder = Chain(Dense(nb_features, 1), z -> dropdims(z; dims=1));\nnothing #hide","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The true vertex costs computed from this encoding are then used within shortest path computations. To be consistent with the literature, we frame this problem as a linear maximization problem, which justifies the change of sign in front of theta.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"function linear_maximizer(θ)\n    g = AcyclicGridGraph(-θ)\n    path = grid_topological_sort(g, 1, nv(g))\n    return path_to_matrix(g, path)\nend;\nnothing #hide","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We now have everything we need to build our dataset.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"nb_instances = 30\n\nX_train = [randn(nb_features, h, w) for n in 1:nb_instances];\nθ_train = [true_encoder(x) for x in X_train];\nY_train = [linear_maximizer(θ) for θ in θ_train];\nnothing #hide","category":"page"},{"location":"tutorial/#Learning","page":"Tutorial","title":"Learning","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We create a trainable model with the same structure as the true encoder but another set of randomly-initialized weights.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"initial_encoder = Chain(Dense(nb_features, 1), z -> dropdims(z; dims=1));\nnothing #hide","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Here is the crucial part where InferOpt intervenes: the choice of a clever loss function that enables us to","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"differentiate through the shortest path maximizer, even though it is a combinatorial operation\nevaluate the quality of our model based on the paths that it recommends","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"regularized_predictor = PerturbedAdditive(linear_maximizer; ε=1.0, nb_samples=5);\nloss = FenchelYoungLoss(regularized_predictor);\nnothing #hide","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The regularized predictor is just a thin wrapper around our linear_maximizer, but with a very different behavior:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"p_regularized = regularized_predictor(θ_train[1]);\nspy(p_regularized)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Instead of choosing just one path, it spreads over several possible paths, allowing its output to change smoothly as theta varies. Thanks to this smoothing, we can now train our model with a standard gradient optimizer.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"encoder = deepcopy(initial_encoder)\nopt = ADAM();\nlosses = Float64[]\nfor epoch in 1:200\n    l = 0.0\n    for (x, y) in zip(X_train, Y_train)\n        grads = gradient(Flux.params(encoder)) do\n            l += loss(encoder(x), y)\n        end\n        Flux.update!(opt, Flux.params(encoder), grads)\n    end\n    push!(losses, l)\nend;\nnothing #hide","category":"page"},{"location":"tutorial/#Results","page":"Tutorial","title":"Results","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Since the Fenchel-Young loss is convex, it is no wonder that optimization worked like a charm.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"lineplot(losses; xlabel=\"Epoch\", ylabel=\"Loss\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"To assess performance, we can compare the learned weights with their true (hidden) values","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"learned_weight = encoder[1].weight / norm(encoder[1].weight)\ntrue_weight = true_encoder[1].weight / norm(true_encoder[1].weight)\nhcat(learned_weight, true_weight)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We are quite close to recovering the exact user weights. But in reality, it doesn't matter as much as our ability to provide accurate path predictions. Let us therefore compare our predictions with the actual paths on the training set.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"normalized_hamming(x, y) = mean(x[i] != y[i] for i in eachindex(x))","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Y_train_pred = [linear_maximizer(encoder(x)) for x in X_train];\n\ntrain_error = mean(\n    normalized_hamming(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred)\n)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Not too bad, at least compared with our random initial encoder.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Y_train_pred_initial = [linear_maximizer(initial_encoder(x)) for x in X_train];\n\ntrain_error_initial = mean(\n    normalized_hamming(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred_initial)\n)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This is definitely a success. Of course in real prediction settings we should measure performance on a test set as well. This is left as an exercise to the reader.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This page was generated using Literate.jl.","category":"page"},{"location":"math/#Mathematical-background","page":"Mathematical background","title":"Mathematical background","text":"","category":"section"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"Our goal is make machine learning models more expressive by incomporating combinatorial optimization algorithms as layers. For a broader perspective on the interactions between machine learning and combinatorial optimization, please refer to the following review papers:","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"tip: Reference\nMachine Learning for Combinatorial Optimization: A Methodological Tour d’Horizon","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"tip: Reference\nEnd-to-end Constrained Optimization Learning: A Survey","category":"page"},{"location":"math/#Combinatorial-optimization-layers","page":"Mathematical background","title":"Combinatorial optimization layers","text":"","category":"section"},{"location":"math/#Linear-formulation","page":"Mathematical background","title":"Linear formulation","text":"","category":"section"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"Our package is centered around the integration of Linear Programs (LPs) such as","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"theta longmapsto haty(theta) = argmax_y in mathcalY theta^T y","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"into machine learning pipelines. Here, theta is an objective vector, while mathcalY is a finite subset of mathbbR^d. Since the optimum of an LP is always reached at a vertex of the feasible polytope, we can start by replacing mathcalY with its convex hull mathcalC = mathrmconv(mathcalY).","category":"page"},{"location":"math/#Implementation-doesn't-matter","page":"Mathematical background","title":"Implementation doesn't matter","text":"","category":"section"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"Our framework does not constrain the actual procedure used to find a solution haty(theta). As long as the problem we solve involves a linear objective over a polytope mathcalC, any optimization oracle is fair game, and we do not care about implementation details.","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"Example 1: If we consider a Mixed Integer Linear Program (MILP), the convex hull of the integer solutions often cannot be expressed in a concise way. In that case, we will most likely use a MILP solver on mathcalY instead of an LP solver on mathcalC, even though both formulations are theoretically equivalent.\nExample 2: For some applications, we don't even have to rely on mathematical programming solvers such as CPLEX or Gurobi. For instance, Dijkstra's algorithm for shortest paths or the Edmonds-Karp algorithm for maximum flows can also be used to tackle LPs with specific structure.","category":"page"},{"location":"math/#The-problem-with-differentiability","page":"Mathematical background","title":"The problem with differentiability","text":"","category":"section"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"Let us suppose that our LP is one of several layers in a machine learning pipeline. To learn the parameters of the other layers, we would like to use a gradient algorithm, which requires the whole pipeline to be differentiable. Unfortunately, the argmin of an LP is a piecewise-constant function, able to jump discontinuously between polytope vertices due to very small shifts in the objective vector theta.","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"The first contribution of InferOpt.jl consists in several methods for computing approximate differentials of LP layers.","category":"page"},{"location":"math/#Loss-functions-for-structured-learning","page":"Mathematical background","title":"Loss functions for structured learning","text":"","category":"section"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"In order to train our pipeline, we also need a loss function. Ideally, this loss should be aware of the combinatorial optimization layer. Furthermore, we want to adapt it to the kind of data at our disposal: do we have access to precomputed solutions (\"learning by imitation\") or just to previous problem instances (\"learning by experience\")?","category":"page"},{"location":"math/","page":"Mathematical background","title":"Mathematical background","text":"The second contribution of InferOpt.jl is a catalogue of structured loss functions, many of them with nice properties such as differentiability or even convexity.","category":"page"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/axelparmentier/InferOpt.jl/blob/main/README.md\"","category":"page"},{"location":"#InferOpt.jl","page":"Home","title":"InferOpt.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue)","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"InferOpt.jl is a toolbox for using combinatorial optimization algorithms within machine learning pipelines.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It allows you to differentiate through things that should not be differentiable, such as Mixed Integer Linear Programs or graph algorithms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package is at a very early development stage, so proceed with caution!","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install the stable version, open a Julia REPL and run the following command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg; Pkg.add(\"InferOpt\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"To install the development version, run this command instead:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg; Pkg.add(url=\"https://github.com/axelparmentier/InferOpt.jl\")","category":"page"},{"location":"implementation/#Implementation","page":"Implementation","title":"Implementation","text":"","category":"section"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"Here we describe the technical details of the InferOpt.jl codebase.","category":"page"},{"location":"implementation/#Wrapping-optimizers","page":"Implementation","title":"Wrapping optimizers","text":"","category":"section"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"In the Mathematical background, we saw that our package provides a principled way to approximate combinatorial problems with machine learning. More specifically, we implement several ways to convert combinatorial problems into differentiable layers of a machine learning pipeline.","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"Since we want our package to be as generic as possible, we don't make any assumptions on the kind of algorithm used to solve these combinatorial problems. We only ask the user to provide a function called maximizer, which takes theta as argument and returns a solution haty(theta) = argmax_y in mathcalY theta^T y.","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"This function is then wrapped into a callable Julia struct that can be used (for instance) within neural networks from the Flux.jl library.","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"tip: Reference\nFlux: Elegant Machine Learning with Julia","category":"page"},{"location":"implementation/#Leveraging-Automatic-Differentiation","page":"Implementation","title":"Leveraging Automatic Differentiation","text":"","category":"section"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"To achieve this goal, we leverage Julia's Automatic Differentiation (AD) ecosystem, which revolves around the ChainRules.jl package. See the paper below for an overview of this ecosystem:","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"tip: Reference\nAbstractDifferentiation.jl: Backend-Agnostic Differentiable Programming in Julia","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"If you need a refresher on forward and reverse-mode AD, the following survey is a good starting point:","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"tip: Reference\nAutomatic Differentiation in Machine Learning: a Survey","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"In machine learning (especially deep learning), reverse-mode AD is by far the most common. Therefore, as soon as we define a new type of layer, we must make it possible to compute the backward pass through this layer. In other words, for each function b = f(a), we need to implement a \"pullback function\" that takes a sensitivity delta_b and returns the associated sensitivity delta_a = delta_b fracmathrmdbmathrmda.","category":"page"},{"location":"implementation/","page":"Implementation","title":"Implementation","text":"See the ChainRules.jl documentation for more details.","category":"page"},{"location":"algorithms/#API-Reference","page":"Algorithms & API","title":"API Reference","text":"","category":"section"},{"location":"algorithms/#Index","page":"Algorithms & API","title":"Index","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]","category":"page"},{"location":"algorithms/#Interpolation","page":"Algorithms & API","title":"Interpolation","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nDifferentiation of Blackbox Combinatorial Solvers","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]\nPages = [\"interpolation/interpolation.jl\"]","category":"page"},{"location":"algorithms/#InferOpt.Interpolation","page":"Algorithms & API","title":"InferOpt.Interpolation","text":"Interpolation{F}\n\nPiecewise-linear interpolation of a black-box optimizer.\n\nFields\n\nmaximizer::F: underlying argmax function\nλ::Float64: smoothing parameter (smaller = more faithful approximation, larger = more informative gradients)\n\nReference: https://arxiv.org/abs/1912.02175\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Smart-\"Predict,-then-Optimize\"","page":"Algorithms & API","title":"Smart \"Predict, then Optimize\"","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nSmart \"Predict, then Optimize\"","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]\nPages = [\"spo/spoplus_loss.jl\"]","category":"page"},{"location":"algorithms/#InferOpt.SPOPlusLoss","page":"Algorithms & API","title":"InferOpt.SPOPlusLoss","text":"SPOPlusLoss{F}\n\nConvex surrogate of the Smart \"Predict-then-Optimize\" loss.\n\nFields\n\nmaximizer::F: linear maximizer function of the form θ ⟼ ŷ(θ) = argmax θᵀy\nα::Float64: convexification parameter\n\nReference: https://arxiv.org/abs/1710.08005\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Structured-Support-Vector-Machines","page":"Algorithms & API","title":"Structured Support Vector Machines","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nStructured learning and prediction in computer vision, Chapter 6","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]\nPages = [\"ssvm/isbaseloss.jl\", \"ssvm/ssvm_loss.jl\", \"ssvm/zeroone_baseloss.jl\"]","category":"page"},{"location":"algorithms/#InferOpt.IsBaseLoss","page":"Algorithms & API","title":"InferOpt.IsBaseLoss","text":"IsBaseLoss{L}\n\nTrait-based interface for loss functions δ(y, y_true), which are the base of the more complex StructuredSVMLoss.\n\nFor δ::L to comply with this interface, the following methods must exist:\n\n(δ)(y, y_true)\ncompute_maximizer(δ, θ, α, y_true)\n\nAvailable implementations\n\nZeroOneBaseLoss\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#InferOpt.compute_maximizer-Union{Tuple{L}, Tuple{Type{IsBaseLoss{L}}, L, Any, Any, Any}} where L","page":"Algorithms & API","title":"InferOpt.compute_maximizer","text":"compute_maximizer(δ, θ, α, y_true)\n\nCompute argmax_y {δ(y, y_true) + α θᵀ(y - y_true)} to deduce the gradient of a StructuredSVMLoss.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.StructuredSVMLoss","page":"Algorithms & API","title":"InferOpt.StructuredSVMLoss","text":"StructuredSVMLoss{L}\n\nLoss associated with the Structured Support Vector Machine.\n\nℓ(θ, y_true) = max_y {δ(y, y_true) + α θᵀ(y - y_true)}\n\nFields\n\nbase_loss::L:  of the IsBaseLoss trait\nα::Float64\n\nReference: http://www.nowozin.net/sebastian/papers/nowozin2011structured-tutorial.pdf (Chapter 6)\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#InferOpt.ZeroOneBaseLoss","page":"Algorithms & API","title":"InferOpt.ZeroOneBaseLoss","text":"ZeroOneBaseLoss\n\n0-1 loss for multiclass classification: δ(y, y_true) = 0 if y = y_true, and 1 otherwise.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Regularized-optimizers","page":"Algorithms & API","title":"Regularized optimizers","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nLearning with Fenchel-Young Losses","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]\nPages = [\"regularized/frank_wolfe.jl\", \"regularized/isregularized.jl\", \"regularized/soft_argmax.jl\", \"regularized/sparse_argmax.jl\", \"regularized/regularized_utils.jl\"]","category":"page"},{"location":"algorithms/#InferOpt.IsRegularized","page":"Algorithms & API","title":"InferOpt.IsRegularized","text":"IsRegularized{P}\n\nTrait-based interface for regularized prediction functions ŷ(θ) = argmax {θᵀy - Ω(y)}.\n\nFor predictor::P to comply with this interface, the following methods must exist:\n\n(predictor)(θ)\ncompute_regularization(predictor, y)\n\nAvailable implementations\n\none_hot_argmax\nsoft_argmax\nsparse_argmax\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#InferOpt.soft_argmax-Tuple{AbstractVector{<:Real}}","page":"Algorithms & API","title":"InferOpt.soft_argmax","text":"soft_argmax(z)\n\nSoft argmax activation function s(z) = (e^zᵢ / ∑ e^zⱼ)ᵢ.\n\nCorresponds to regularized prediction on the probability simplex with entropic penalty.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.simplex_projection_and_support-Tuple{AbstractVector{<:Real}}","page":"Algorithms & API","title":"InferOpt.simplex_projection_and_support","text":"simplex_projection_and_support(z)\n\nCompute the Euclidean projection p of z on the probability simplex (also called sparse_argmax), and the indicators s of its support.\n\nReference: https://arxiv.org/abs/1602.02068.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.sparse_argmax-Tuple{AbstractVector{<:Real}}","page":"Algorithms & API","title":"InferOpt.sparse_argmax","text":"sparse_argmax(z)\n\nCompute the Euclidean projection of the vector z onto the probability simplex.\n\nCorresponds to regularized prediction on the probability simplex with square norm penalty.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.half_square_norm-Tuple{AbstractArray{<:Real}}","page":"Algorithms & API","title":"InferOpt.half_square_norm","text":"half_square_norm(x)\n\nCompute the squared Euclidean norm of x and divide it by 2.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.isproba-Tuple{Real}","page":"Algorithms & API","title":"InferOpt.isproba","text":"isproba(x)\n\nCheck whether x ∈ [0,1].\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.isprobadist-Union{Tuple{AbstractVector{R}}, Tuple{R}} where R<:Real","page":"Algorithms & API","title":"InferOpt.isprobadist","text":"isprobadist(p)\n\nCheck whether the elements of p are nonnegative and sum to 1.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.one_hot_argmax-Union{Tuple{AbstractVector{R}}, Tuple{R}} where R<:Real","page":"Algorithms & API","title":"InferOpt.one_hot_argmax","text":"one_hot_argmax(z)\n\nOne-hot encoding of the argmax function.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.positive_part-Tuple{Any}","page":"Algorithms & API","title":"InferOpt.positive_part","text":"positive_part(x)\n\nCompute max(x,0).\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.ranking-Tuple{AbstractVector{<:Real}}","page":"Algorithms & API","title":"InferOpt.ranking","text":"ranking(θ[; rev])\n\nCompute the vector r such that rᵢ is the rank of θᵢ in θ.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.shannon_entropy-Union{Tuple{AbstractVector{R}}, Tuple{R}} where R<:Real","page":"Algorithms & API","title":"InferOpt.shannon_entropy","text":"shannon_entropy(p)\n\nCompute the Shannon entropy of a probability distribution: H(p) = -∑ pᵢlog(pᵢ).\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#Perturbed-optimizers","page":"Algorithms & API","title":"Perturbed optimizers","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nLearning with Differentiable Perturbed Optimizers","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]\nPages = [\"perturbed/abstract_perturbed.jl\", \"perturbed/additive.jl\", \"perturbed/composition.jl\", \"perturbed/multiplicative.jl\"]","category":"page"},{"location":"algorithms/#InferOpt.AbstractPerturbed","page":"Algorithms & API","title":"InferOpt.AbstractPerturbed","text":"AbstractPerturbed{F}\n\nDifferentiable perturbation of a black-box optimizer.\n\nAvailable subtypes\n\nPerturbedAdditive{F}\nPerturbedMultiplicative{F}\n\nRequired fields\n\nmaximizer::F: underlying argmax function\nε::Float64: noise scaling parameter\nrng::AbstractRNG: random number generator\nseed::Union{Nothing,Int}: random seed\nnb_samples::Int: number of random samples for Monte-Carlo computations\n\nRequired methods\n\n(perturbed)(θ, Z; kwargs...)\ncompute_y_and_F(perturbed, θ, Z; kwargs...)\n\nOptional methods\n\nrrule(perturbed, θ; kwargs...)\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#InferOpt.AbstractPerturbed-Tuple{AbstractArray{<:Real}, AbstractArray{<:Real}}","page":"Algorithms & API","title":"InferOpt.AbstractPerturbed","text":"(perturbed)(θ, Z; kwargs...)\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.compute_y_and_F-Tuple{AbstractPerturbed, AbstractArray{<:Real}, AbstractArray{<:Real}}","page":"Algorithms & API","title":"InferOpt.compute_y_and_F","text":"compute_y_and_F(perturbed, θ, Z; kwargs...)\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.PerturbedAdditive","page":"Algorithms & API","title":"InferOpt.PerturbedAdditive","text":"PerturbedAdditive{F}\n\nDifferentiable normal perturbation of a black-box optimizer: the input undergoes θ -> θ + εZ where Z ∼ N(0, I).\n\nSee also: AbstractPerturbed{F}.\n\nReference: https://arxiv.org/abs/2002.08676\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#InferOpt.PerturbedComposition","page":"Algorithms & API","title":"InferOpt.PerturbedComposition","text":"PerturbedComposition{F,P<:AbstractPerturbed{F},G}\n\nComposition of a differentiable perturbed black-box optimizer with an arbitrary function.\n\nSuitable for direct regret minimization (learning by experience) when said function is a cost.\n\nFields\n\nperturbed::P: underlying AbstractPerturbed{F} wrapper\ng::G: function taking an array y and some kwargs as inputs\n\nThe method rrule(perturbed_composition, θ; kwargs...) must be implemented individually for each specific type P.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Base.:∘-Tuple{Any, AbstractPerturbed}","page":"Algorithms & API","title":"Base.:∘","text":"∘(g, perturbed)\n\nCreate a PerturbedComposition object from perturbed and g.\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#ChainRulesCore.rrule-Tuple{PerturbedComposition, AbstractArray{<:Real}}","page":"Algorithms & API","title":"ChainRulesCore.rrule","text":"rrule(perturbed_composition, θ; kwargs...)\n\n\n\n\n\n","category":"method"},{"location":"algorithms/#InferOpt.PerturbedMultiplicative","page":"Algorithms & API","title":"InferOpt.PerturbedMultiplicative","text":"PerturbedMultiplicative{F}\n\nDifferentiable log-normal perturbation of a black-box optimizer: the input undergoes θ -> θ ⊙ exp[εZ - ε²/2] where Z ∼ N(0, I).\n\nSee also: AbstractPerturbed{F}.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Fenchel-Young-losses","page":"Algorithms & API","title":"Fenchel-Young losses","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nLearning with Fenchel-Young Losses","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"Modules = [InferOpt]\nPages = [\"fenchel_young/fenchel_young.jl\"]","category":"page"},{"location":"algorithms/#InferOpt.FenchelYoungLoss","page":"Algorithms & API","title":"InferOpt.FenchelYoungLoss","text":"FenchelYoungLoss{P}\n\nFenchel-Young loss associated with a given regularized prediction function.\n\nFields\n\npredictor::P: prediction function of the form ŷ(θ) = argmax {θᵀy - Ω(y)}\n\nReference: https://arxiv.org/abs/1901.02324\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Implicit-differentiation","page":"Algorithms & API","title":"Implicit differentiation","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"tip: Reference\nEfficient and Modular Implicit Differentiation","category":"page"},{"location":"algorithms/","page":"Algorithms & API","title":"Algorithms & API","text":"note: Stay tuned!\nThis will soon be implemented thanks to the recent package ImplicitDifferentiation.jl.","category":"page"}]
}