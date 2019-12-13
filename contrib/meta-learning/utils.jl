"""
```
eval_model(model, x::AbstractArray, testx::AbstractArray, task=SineWave(); 
           opt=Descent(1e-2), updates=32)
```
Evaluates the `model` on a sine wave `task` training to sample `x` with `updates`
amount of gradient steps using `opt`. 
Evaluation loss is calculated based on the mean squared error
between model predictions and sine wave values on `testx`.
"""
function eval_model(model, x::AbstractArray, testx::AbstractArray, task=SineWave(); 
                    opt=Descent(0.02), updates=32)
    weights = params(model)
    prev_weights = deepcopy(Flux.data.(weights))

    y = task(x)
    testy = task(testx)
    init_preds = model(testx')
    test_loss = Flux.mse(init_preds, testy')

    test_losses = Float32[]
    push!(test_losses, Flux.data(test_loss))

    print(task, "\n")
    @printf("Before finetuning, Loss = %f\n", test_loss)
    for i in 1:updates
        l = Flux.mse(model(x'), y')
        Flux.back!(l)
        Flux.Optimise._update_params!(opt, weights)
        test_loss = Flux.mse(model(testx'), testy')
        push!(test_losses, Flux.data(test_loss))
        @printf("After %d fits, Loss = %f\n", i, test_loss)
    end
    final_preds = model(testx')

    # reset weights to state before finetune
    Flux.loadparams!(model, prev_weights)

    return (x=x, testx=testx, y=y, testy=testy, 
            initial_predictions=Array(Flux.data(init_preds)'),
            final_predictions=Array(Flux.data(final_preds)'), 
            test_losses=test_losses)
end

function plot_eval_data(data::NamedTuple, title="")
    return plot([data.x, data.testx, data.testx, data.testx], 
                [data.y, data.testy, data.initial_predictions, data.final_predictions],
                line=[:scatter :path :path :path],
                label=["Sampled points", "Ground truth", "Before finetune", "After finetune"],
                foreground_color_legend=:white, background_color_legend=:transparent,
                title=title, 
                xlim=(-5.5, 5.5))
end
