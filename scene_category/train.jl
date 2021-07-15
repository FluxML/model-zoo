model = ResNet(18) |> gpu

info("Model Load Complete")

accuracy(x, y) = mean(argmax(model(x), 0:(label_count-1)) .== argmax(y, 0:(label_count-1)))

loss(x, y) = crossentropy(model(x), y)

opt = SGD(params(model), 0.1)

info("Checking architecture on sample data")

model(train[1][1] |> gpu)

info("Architecture Check successful")

@epochs 5 begin
    tic()
    for i in train
        l = loss(i[1] |> gpu, i[2] |> gpu)
        println("The loss for current minibatch is $l")
        Flux.back!(l)
        opt()
    end
    toc()
end

model = model |> cpu
@save "model_checkpoint.bson" model
