## Dimensions and parameters

N = 1000
dim_x = 10
dim_y = 8
σ = 0.01
epochs = 200

## True model

true_model = Dense(dim_x, dim_y; bias=false)
true_optimizer = Dict("argmax" => one_hot_argmax, "ranking" => ranking)

## Learning pipelines

pipelines = Dict(
    "argmax" => Dict(
        "y" => [
            # Structured SVM
            (model=Dense(dim_x, dim_y), loss=StructuredSVMLoss(ZeroOneLoss())),
            # Regularized prediction: explicit
            (model=Dense(dim_x, dim_y), loss=FenchelYoungLoss(one_hot_argmax)),
            (model=Dense(dim_x, dim_y), loss=FenchelYoungLoss(sparsemax)),
            (model=Dense(dim_x, dim_y), loss=FenchelYoungLoss(InferOpt.softmax)),
            # Perturbations
            (
                model=Dense(dim_x, dim_y),
                loss=FenchelYoungLoss(Perturbed(one_hot_argmax; ε=0.1, M=10)),
            ),
            (
                model=Chain(Dense(dim_x, dim_y), Perturbed(one_hot_argmax; ε=0.2, M=10)),
                loss=Flux.Losses.mse,
            ),
        ],
        "θ" => [(model=Dense(dim_x, dim_y), loss=SPOPlusLoss(one_hot_argmax))],
        "(θ,y)" => [(model=Dense(dim_x, dim_y), loss=SPOPlusLoss(one_hot_argmax))],
        "none" => [],
    ),
    "ranking" => Dict(
        "y" => [
            (
                model=Dense(dim_x, dim_y),
                loss=FenchelYoungLoss(Perturbed(ranking; ε=0.1, M=10)),
            ),
            (
                model=Chain(Dense(dim_x, dim_y), Perturbed(ranking; ε=0.1, M=10)),
                loss=Flux.Losses.mse,
            ),
            (
                model=Chain(Dense(dim_x, dim_y), Interpolation(ranking; λ=10.0)),
                loss=Flux.Losses.mse,
            ),
        ],
        "θ" => [(model=Dense(dim_x, dim_y), loss=SPOPlusLoss(ranking))],
        "(θ,y)" => [(model=Dense(dim_x, dim_y), loss=SPOPlusLoss(ranking))],
        "none" => [],
    ),
)

## Test loop

for setting in ["argmax", "ranking"], target in ["y", "θ", "none"]
    @testset verbose = true "Setting: $setting - Target: $target" begin
        @unpack X, thetas, Y = generate_dataset(
            true_model, true_optimizer[setting]; N=N, dim_x=dim_x, dim_y=dim_y, σ=σ
        )
        X_train, X_test = train_test_split(X)
        thetas_train, thetas_test = train_test_split(thetas)
        Y_train, Y_test = train_test_split(Y)

        if target == "y"
            data = zip(X_train, Y_train)
        elseif target == "θ"
            data = zip(X_train, thetas_train)
        else
            data = zip(X_train)
        end

        for pipeline in pipelines[setting][target]
            @unpack model, loss = pipeline

            @info "Testing probability simplex" setting target model loss

            opt = ADAM()

            training_losses = Float64[]
            test_accuracies = Float64[]
            parameter_errors = Float64[]

            flux_loss_no_target(x) = loss(model(x))
            flux_loss_single_target(x, t) = loss(model(x), t)
            flux_loss_double_target(x, t1, t2) = loss(model(x), t1, t2)

            if target == "none"
                flux_loss = flux_loss_no_target
            elseif target == "(θ,y)"
                flux_loss = flux_loss_double_target
            else
                flux_loss = flux_loss_single_target
            end

            @showprogress for _ in 1:epochs
                l = sum(flux_loss(x, t) for (x, t) in data)

                Y_test_pred = generate_predictions(model, true_optimizer[setting], X_test)
                a = mean(
                    1 - normalized_hamming_distance(y_pred, y) for
                    (y_pred, y) in zip(Y_test_pred, Y_test)
                )

                w = model isa Dense ? model.weight : model[1].weight
                true_w = true_model.weight
                e = normalized_mape(true_w, w)

                push!(training_losses, l)
                push!(test_accuracies, a)
                push!(parameter_errors, e)

                Flux.train!(flux_loss, Flux.params(model), data, opt)
            end

            test_name = first("$model - $loss", 100)
            @testset "$test_name" begin
                @test training_losses[end] < training_losses[1]
                @test test_accuracies[end] > test_accuracies[1]
                @test parameter_errors[end] < parameter_errors[1]
                @test test_accuracies[end] > 0.8
            end

            if SHOW_PLOTS
                plot_results(training_losses, test_accuracies, parameter_errors)
            end
        end
    end
end
