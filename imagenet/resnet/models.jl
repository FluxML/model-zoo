include("standardresnet.jl")

# Standard ResNet models for Imagenet as described in the Paper "Deep Residual Learning for Image Recognition"
# Uncomment the model that you want to use
ResNet18 = StandardResnet(BasicBlock, [2, 2, 2, 2])

# ResNet34 = StandardResnet(BasicBlock, [3, 4, 6, 3])

# ResNet50 = StandardResnet(Bottleneck, [3, 4, 6, 3])

# ResNet101 = StandardResnet(Bottleneck, [3, 4, 23, 3])

# ResNet152 = StandardResnet(Bottleneck, [3, 8, 36, 3])

# Test the model on some random data point to verify if its running properly
ResNet18(rand(224,224,3,10))

loss(x, y) = crossentropy(ResNet18(x), y)

accuracy(x, y) = mean(argmax(ResNet18(x)) .== argmax(y))

opt = ADAM(params(ResNet18))

# Demonstrating the training on randomly generated data
imgs = [rand(224,224,3) for i in 1:2000]
labels = Flux.onehotbatch(rand(0:1000,2000))

train = [(cat(4, imgs[i]...), labels[:,i]) for i in partition(1:2000, 10)]

Flux.train!(loss, train, opt)
