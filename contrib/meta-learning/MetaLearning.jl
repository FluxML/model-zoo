# Meta-Learning
# ==============
#
# This is a shortened version of https://www.domluna.me/meta-learning/.
#
# > Meta-learning is an algorithm which learns to learn.
#
# We will use meta-learning to train a neural net which
# can be fintuned to accurately models a sine wave after 
# training for few iterations on a handful of x, y coordinates.
#
# References:
#
# [Model-Agnostic Meta-Learning for Fast Adapdation of Deep Networks](https://arxiv.org/abs/1703.03400).
# [Reptile: A Scalable Meta-Learning Algorithm](https://blog.openai.com/reptile/)

using Flux
using Printf
using Plots
using Distributions: Uniform, Normal
using Statistics: mean
using Base.Iterators: partition
using Random: randperm, seed!

include("utils.jl")
include("linear.jl")

# Sine Waves
# -----------
#
# We'll be training the neural network on sine waves in the 
# x coordinate of range $[-5, 5]$. The sine wave can be shifted left
# or right by the `phase_shift` and shrunk or streched by the `amplitude`.

struct SineWave
    amplitude::Float32
    phase_shift::Float32
end
SineWave() = SineWave(rand(Uniform(0.1, 5)), rand(Uniform(0, 2pi)))

(s::SineWave)(x::AbstractArray) = s.amplitude .* sin.(x .+ s.phase_shift)

function Base.show(io::IO, s::SineWave)
    print(io, "SineWave(amplitude = ", s.amplitude, ", phase shift = ", s.phase_shift, ")")
end

# Now let's make some waves ...

x = LinRange(-5, 5, 100)
#-
wave = SineWave(4, 1)
#-
plot(x, wave(x))

# Nice, time to learn to learn! 



# `tanh` activations seem to produce smoother lines than `relu`.

# First-Order MAML (FOMAML)
# --------------------------
#
# MAML can be succinctly described as:
# 
# $$
# \phi_i = \theta - \alpha \bigtriangledown_{\theta} Loss(D_{t_i}^{train}; \theta) \\
# \theta \leftarrow \theta - \beta \sum_i \bigtriangledown_{\theta} Loss(D_{t_i}^{test}; \phi_i)
# $$
# 
# The first line shows the inner update, which optimizes $\theta$ towards a solution for the task 
# training set $D_{t_i}^{train}$ producing $\phi_i$. The following line is the meta update which 
# aims to generalize to new data. This involves evaluating all $\phi_i$ on the test sets $D_{t_i}^{test}$ 
# and accumulating the resulting gradients. $\alpha$ and $\beta$ are learning rates.
# 
# The difference between MAML and FOMAML (first-order MAML) is the inner gradient, shown is red 
# is ignored during backpropagation:
# 
# $$
# \theta = \theta - \beta \sum_i \bigtriangledown_{\theta} Loss( \theta - \alpha {\color{red} \bigtriangledown_{\theta} Loss(\theta, D_{t_i}^{train})}, D_{t_i}^{test})
# $$
# 
# We'll focus on FOMAML since it's less computationally expensive and achieves similar performance 
# to MAML in practice and has a simpler implementation.
# 
# FOMAML generalizes by further adjusting parameters based on how performance on a validation 
# or test set (used interchangeably) during the meta update. This is best shown visually:
# 
# ![fomaml gradient updates](./fomaml_grad.png)
# 
# $\theta$ is the starting point, $\phi_{3}$ is the parameter values after 3 gradient updates on a task. 
# Notice before the meta update the parameters shift in the direction of the new task's solution (red arrow) 
# but after the meta update they change direction (blue arrow). This illustrates how the meta update 
# adjusts the gradient for task generalization.

function fomaml(model; meta_opt=Descent(0.01), inner_opt=Descent(0.02), epochs=30_000, 
              n_tasks=3, train_batch_size=10, eval_batch_size=10, eval_interval=1000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = Float32.(range(-5, stop=5, length=50))

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))

        for _ in 1:n_tasks
            task = SineWave()

            x = Float32.(rand(dist, train_batch_size))
            y = task(x)
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights)

            for w in weights
                w.data .-= Flux.Optimise.apply!(inner_opt, w.data, grad[w].data)
            end

            testy = task(testx)
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(testx'), testy'), weights)

            # reset weights and accumulate gradients
            for (w1, w2) in zip(weights, prev_weights)
                w1.data .= w2
                w1.grad .+= grad[w1].data
            end

        end

        Flux.Optimise._update_params!(meta_opt, weights)

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            eval_model(model, evalx, testx, SineWave())
        end

    end
end

# Training a neural net using FOMAML.

seed!(0)
fomaml_model = Chain(Linear(1, 64, tanh), Linear(64, 64, tanh), Linear(64, 1))
fomaml(fomaml_model, meta_opt=Descent(0.01), inner_opt=Descent(0.02), epochs=50_000, n_tasks=3)

# Evaluate the model on 10 (x, y) sampled uniformly from $[-5, 5]$

x = rand(Uniform(-5, 5), 10)
testx = LinRange(-5, 5, 50)
#-
fomaml_data = eval_model(fomaml_model, x, testx, wave, updates=32, opt=Descent(0.02))
#-
p = plot_eval_data(fomaml_data, "FOMAML - SGD Optimizer")
#-
plot(p)

# Reptile
# --------------------------
#
# Reptile optimizes to find a point (parameter representation) in manifold space which is 
# closest in euclidean distance to a point in each task's manifold of optimal solutions. 
# To achieve this we minimize the expected value for all tasks $t$ of $(\theta - \phi_{*}^{t})^2$ 
# where $\theta$ is the model's parameters and $\phi_{*}^{t}$ are the optimal parameters for task $t$.
# 
# $$
# E_{t}[\frac{1}{2}(\theta - \phi_{*}^{t})^2]
# $$
# 
# In each iteration of Reptile we sample a task and update $\theta$ using SGD:
# 
# $$
# \theta \leftarrow \theta - \alpha \bigtriangledown_{\theta} \frac{1}{2}(\theta - \phi_{*}^{t})^2 \\
# \theta \leftarrow \theta - \alpha(\theta - \phi_{*}^{t})
# $$
# 
# In practice an approximation of $\phi_{*}^{t}$ is used since it's not feasible to compute. The approximation
# is $\phi$ after $i$ gradient steps $\phi_{i}^{t}$.
# 
# ![Reptile gradient updates](./reptile_grad.png)
# 
# This a Reptile update after training for 3 gradient steps on task data. Note with Reptile there's 
# no train and test data, just data. The direction of the gradient update (blue arrow) is directly 
# in the direction towards $\phi_i$.  It's kind of crazy that this actually works. Section 5 of 
# the [Reptile paper](https://arxiv.org/abs/1803.02999) has an analysis showing the gradients of 
# MAML, FOMMAL and Reptile are similar within constants.

function reptile(model; meta_opt=Descent(0.1), inner_opt=Descent(0.02), epochs=30_000, 
                 train_batch_size=10, eval_batch_size=10, eval_interval=1000)
    weights = params(model)
    dist = Uniform(-5, 5)
    testx = Float32.(range(-5, stop=5, length=50))
    x = testx

    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))
        task = SineWave()

        # Train on task for k steps on the dataset
        y = task(x)
        for idx in partition(randperm(length(x)), train_batch_size)
            l = Flux.mse(model(x[idx]'), y[idx]')
            Flux.back!(l)
            Flux.Optimise._update_params!(inner_opt, weights)
        end

        # Reptile update
        for (w1, w2) in zip(weights, prev_weights)
            gw = Flux.Optimise.apply!(meta_opt, w2, w1.data - w2)
            @. w1.data = w2 + gw
        end

        if i % eval_interval == 0
            @printf("Iteration %d, evaluating model on random task...\n", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            eval_model(model, evalx, testx, SineWave())
        end

    end
end

# Training a neural net using Reptile.

seed!(0)
reptile_model = Chain(Linear(1, 64, tanh), Linear(64, 64, tanh), Linear(64, 1))
reptile(reptile_model, meta_opt=Descent(0.1), inner_opt=Descent(0.02), epochs=50_000)

# Evaluate the model on 10 x coordinates sampled uniformly from $[-5, 5]$

reptile_data = eval_model(reptile_model, x, testx, wave, updates=32, opt=Descent(0.02))
#-
p = plot_eval_data(reptile_data, "Reptile - SGD Optimizer")
#-
plot(p)

# Testing Robustness
# -------------------

# To test if the FOMAML and Reptile representations *learned to learn quickly with minimal data* we'll 
# finetune on 5 datapoints for 10 update steps. The x values are sampled from a uniform distribution 
# of $[0, 5]$, the right half of the sine wave. Can the entire wave be learned?

x = rand(Uniform(0, 5), 5)

# First FOMAML:

fomaml_data = eval_model(fomaml_model, x, testx, wave, updates=10, opt=Descent(0.02))
#-
p = plot_eval_data(fomaml_data, "FOMAML - 5 samples, 10 updates")
#-
plot(p)

# Now Reptile:

reptile_data = eval_model(reptile_model, x, testx, wave, updates=10, opt=Descent(0.02))
#-
p = plot_eval_data(reptile_data, "Reptile - 5 samples, 10 updates")
#-
plot(p)

# Lastly, let's see how quickly a representation is learned for a new task. We'll use the same 
# 5 element sample as before and train for 10 gradient updates.
 
plot([fomaml_data.test_losses, reptile_data.test_losses], 
     label=["FOMAML", "Reptile"], 
     xlabel="Updates", ylabel="Loss")

# Cool! Both FOMAML and Reptile learn a very useful representation within a few updates. 
