#= TODO
* Add any Flux specific explanation
* Handle target value better for dataset y's; Is there a way to ensure that time series always have a batched axis using MLDataPattern, esp lazily?
* Decide what level we want this tutorial at - Flux or some higher level?
* Clean up and make more idiomatic - particularly on how timeseries are batched
* Make any tests or integrations
* Put things in functions as needed
* Eachbatch - size or maxsize? what's more popular
* Flux on master for Dense behavior. Needs to be latest when this is published
* In train_model! is it critical to return and assign the model (`linear = train_model!(linear, single_step_1h, opt; bs=16, epochs=20)`)? I found that without doing it, Flux.update! would work during training, but then calling the model outside wouldn't be mutated? could this deal with calling params at the beginning?
* Early Stopping could use some work. Not sure it does exactly what I want it to
* Need to convert data to Float32? Without it, I get this when running conv_model. Related to Params being Float32?
    ┌ Warning: Slow fallback implementation invoked for conv!  You probably don't want this; check your datatypes.
    │   yT = Float64
    │   T1 = Float64
    │   T2 = Float32
    └ @ NNlib ~/.julia/packages/NNlib/fxLrD/src/conv.jl:206

------------------------------------------------------------------------
=#

# # Time series forecasting
# This tutorial serves as a `Julia` implementation of the Tensorflow time series forecasting tutorial here:  
# 
# [Tensorflow Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)

# ## Setup

using ZipFile
using CSV
using DataFrames
using StatsPlots #re-exports Plots.jl functions
using Dates
using FFTW
using CUDA
using Flux
using MLDataPattern

using Random: seed!
using Statistics: mean, std
using Flux: unsqueeze
import StatsPlots: plot

seed!(4231)
ENV["LINES"] = 20

CUDA.allowscalar(false)

# ## The weather dataset

function download_data(; fname="jena_climate_2009_2016.zip")
    DATA_PATH = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    isfile(fname) || download(DATA_PATH, fname)

    zip = ZipFile.Reader(fname) #https://juliadata.github.io/CSV.jl/stable/index.html#Example:-reading-from-a-zip-file-1
    csv = zip.files[1]
    df = CSV.File(csv) |> DataFrame
    close(zip)
    return df
end

df = download_data();

df = df[6:6:end, :]

df[!,"Date Time"] = Dates.DateTime.(df[:,"Date Time"], "dd.mm.yyyy HH:MM:SS"); #https://en.wikibooks.org/wiki/Introducing_Julia/Working_with_dates_and_times

col = ["Date Time", "T (degC)", "p (mbar)", "rho (g/m**3)"]

df[:,col]

@df df plot(cols(1), cols(2:4); layout=(3, 1)) # from https://github.com/JuliaPlots/StatsPlots.jl

@df df[1:480,:] plot(cols(1), cols(2:4); layout=(3, 1))

# ## Inspect and cleanup
@show describe(df)

df."p (mbar)"; #cool that this works

replace!(df[!,"wv (m/s)"], -9999.0 => 0); # https://juliadata.github.io/DataFrames.jl/stable/man/getting_started/#Replacing-Data
replace!(df[!,"max. wv (m/s)"], -9999.0 => 0);

@show describe(df)

# ## Feature engineering

histogram2d(df[!,"wd (deg)"], df[!,"wv (m/s)"], bins=(75,75), xlabel="wd (deg)", ylabel="wv (m/s)")

wd_rad = df[!,"wd (deg)"] * π / 180
df.Wx = df[!,"wv (m/s)"] .* cos.(wd_rad)
df.Wy = df[!,"wv (m/s)"] .* sin.(wd_rad)
df."max Wx" = df[!,"max. wv (m/s)"] .* cos.(wd_rad)
df."max Wy" = df[!,"max. wv (m/s)"] .* sin.(wd_rad);

histogram2d(df.Wx, df.Wy, bins=(75,75), xlabel="Wind X [m/s]", ylabel="Wind Y [m/s]")

timestamp_s = Dates.datetime2unix.(df."Date Time");

day = 24*60*60 #seconds in a day
year = 365.2425 * day #seconds in a year

df[!,"Day sin"] = sin.(timestamp_s * (2 * π / day))
df[!,"Day cos"] = cos.(timestamp_s * (2 * π / day))
df[!,"Year sin"] = sin.(timestamp_s * (2 * π / year))
df[!,"Year cos"] = cos.(timestamp_s * (2 * π / year));

plot(df[1:25,"Day sin"], legend=false)
plot!(df[1:25,"Day cos"])
xlabel!("Time [h]")
title!("Time of Day Signal")

fftrans = FFTW.rfft(df[!,"T (degC)"])
f_per_dataset = 1:size(fftrans)[1]

n_samples_h = size(df[!,"T (degC)"])[1]
hours_per_year = 24 * 365.2524
years_per_dataset = n_samples_h / hours_per_year
f_per_year = f_per_dataset / years_per_dataset;

plot(f_per_year, abs.(fftrans), xscale=:log10, ylim=(0, 400000), xlim=(0.3,Inf), leg=false)
xticks!([1, 365.2524], ["1/Year", "1/Day"])
xlabel!("Frequency (log scale)")

# ## Split the data
# drop columns you don't want to use further
select!(df, Not([:("wv (m/s)"),:("max. wv (m/s)"), :("wd (deg)"), :("Date Time")]));

column_indices = pairs(names(df))
indices_columns = Dict(value => key for (key, value) in column_indices)
df = convert.(Float32, df) # Don't need high precision; reduces errors later on when using Params - gradients are Float32


n = size(df)[1]
train_df = df[1:round(Int,n*0.7, RoundDown),:]
valid_df = df[round(Int,n*0.7, RoundUp):round(Int,n*0.9, RoundDown),:]
test_df = df[round(Int,n*0.9, RoundUp):end,:]; # matching TF tutorial exactly, can also use partition

num_features = size(df)[2]

# ## Normalize the data
train_mean = mean.(eachcol(train_df))
train_std = std.(eachcol(train_df))

train_df = (train_df .- train_mean') ./ train_std'
valid_df = (valid_df .- train_mean') ./ train_std'
test_df = (test_df .- train_mean') ./ train_std'

df_std = (df .- train_mean') ./ train_std'
df_std = stack(df_std)

violin(df_std.variable, df_std.value, xrotation=30.0, legend=false, xticks=:all) # use plotattr() to learn about keywords

boxplot!(df_std.variable, df_std.value, fillalpha=0.75, outliers=false)


# ## Data Windowing
# We are going to make use of MLDataPattern's `slidingwindow` to generate the windows
# https://mldatapatternjl.readthedocs.io/en/latest/documentation/dataview.html?highlight=slidingwindow#labeled-windows
h = 6 # historical window length
f = 1 # future window length

train_loader = slidingwindow(i -> i+h:i+h+f-1, Array(train_df)', h, stride=1)

# We will define our own WindowGenerator, some constructors, and plotting functions. The data from the WindowGenerator will be used in training.
"""
Calculates the windows used for modeling, the target index, and features useful for plotting.
"""
mutable struct WindowGenerator
    train # training windows
    valid # validation windows
    h # historical steps in window
    f # target steps in window
    label_indices # indices for plotting predictions in a window
    target_idx::Array{Int, 1} # indices to be predicted
end

"""
Specify `h` historical points, `f` target points. By default, the target points are assumed to follow after all the historical points (`offset = h`).
By setting `offset = 1`, the targets for each historical point will be the next point in time.
"""
function WindowGenerator(h, f, train_df, valid_df, label_columns::Vector{String}; offset=h)
    train = slidingwindow(i -> i+offset:i+offset+f-1, Array(train_df)', h, stride=1)
    valid = slidingwindow(i -> i+offset:i+offset+f-1, Array(valid_df)', h, stride=1)
    
    label_indices = (offset + 1):(offset + f)

    target_idx = findall(x->x in label_columns, names(train_df))
        
    return WindowGenerator(train, valid, h, f, label_indices, target_idx)
end

WindowGenerator(h, f, train_df, valid_df, label_columns::String; offset=h) = 
        WindowGenerator(h, f, train_df, valid_df, label_columns=[label_columns]; offset=offset)

WindowGenerator(h, f, train_df, valid_df; label_columns, offset=h) = 
        WindowGenerator(h, f, train_df, valid_df, label_columns; offset=offset)

"""
Plots the historical data and the target(s) for prediction. 
"""
function plot(wg::WindowGenerator)
    idx = wg.target_idx[1] # Currently, will only plot the first target index but could be redone to plot all label columns.
    sw = wg.valid
    plots = []
    for it in 1:3
        i = rand(1:size(sw,1))
        p = plot(sw[i][1][idx, :], leg=false)
        scatter!(wg.label_indices,sw[i][2][idx,:]')
        it == 3 && xlabel!("Given $(wg.h)h as input, predict $(wg.f)h into the future.")
        push!(plots, p)
    end
    plot(plots..., layout=(3,1))
end

# Some usage examples of WindowGenerator
WindowGenerator(6, 1, train_df, valid_df, label_columns=["T (degC)"])
WindowGenerator(6, 1, train_df, valid_df, label_columns=["T (degC)", "Wx"])
wg = WindowGenerator(6, 1, train_df, valid_df, label_columns="T (degC)")

plot(wg)

# Make a utility function for batching lazily-evaluated timeseries from slidingwindow
"""
Takes in t, which is an array of tuples of (`sequence`, `target`) where `sequence` is an array of timestamp x features, 
and target is either an array or a single value. Outputs a tuple of batched 

julia> z = [([1 2 3; 2 3 4], 5), ([6 7 8; 7 8 9], 0)]
2-element Array{Tuple{Array{Int64,2},Int64},1}:
 ([1 2 3; 2 3 4], 5)
 ([6 7 8; 7 8 9], 0)

julia> batch_ts(z)
([1 2 3; 2 3 4]

[6 7 8; 7 8 9], [5]

[0])

julia> size(batch_ts(z)[1])
(2, 3, 2)

julia> size(batch_ts(z)[2])
(1, 1, 2)
"""
batch_ts(t) = reduce((x, y) -> (cat(x[1], y[1], dims=3), cat(x[2], y[2], dims=3)), t)
batch_ts(t::Tuple) = (unsqueeze(t[1],3), unsqueeze(t[2],3)) # handle batch size of 1

# To understand better what the above code does, let's look at an imitation training loop and look at everything's dimensions
practice_df = train_df[1:10,:]
a = slidingwindow(i -> i+h:i+h+f-1, Array(practice_df)', h, stride=1)

for i in eachbatch(shuffleobs(a), size=2)
    (x,y) = batch_ts(i)
    @show x
    @show y
    println()
end

# # Single Step Models
loss(x,y) = Flux.Losses.mse(x, y)

# ### Baseline - 1h
struct Baseline
    label_index::Int
end

(m::Baseline)(x) = x[m.label_index,:,:]

target = "T (degC)"

# Since this model repeats the last point, we make slidingwindows with 1 historical and 1 target point.
single_step_1h = WindowGenerator(1, 1, train_df, valid_df, label_columns=target);

baseline_model = Baseline(wg.target_idx[1])

function run_single_step_baseline(wg, model)
    preds = Float32[]
    reals = Float32[]
    for (x,y) in wg.train
        val = model(x)[1]
        push!(preds, val)
        push!(reals, y[model.label_index]) #figure out how to do validation
    end

    l = loss(preds, reals)
    return l
end

run_single_step_baseline(single_step_1h, baseline_model)

# ### Baseline - 24h
# Let's try to predict the next hour's value for 24 hours
single_step_24h = WindowGenerator(24, 24, train_df, valid_df, label_columns=target; offset=1);

"""
Plots the historical window, the target(s) for prediction, and the model's predictions.
"""
function plot(wg::WindowGenerator, model; set=:valid)
    data = getfield(wg,set)
    i = rand(1:size(data,1))
    plot(1:wg.h, data[i][1][2,:], lab="Inputs", shape=:circ, m=2, leg=:outerright)
    scatter!(wg.label_indices, data[i][2][2,:], lab="Labels", c=:green)
    z = model(unsqueeze(data[i][1],3))[:]
    scatter!(wg.label_indices, z, lab="Predictions", shape=:star5, m=6, c=:orange)
end

plot(single_step_24h, baseline_model)

# ### Linear Models
# ##### 1 hour
linear = Dense(size(single_step_1h.train[1][1],1), 1; initW=Flux.glorot_uniform, initb=Flux.zeros)
opt = Flux.Optimise.ADAM(0.01)

function train_model!(model, wg::WindowGenerator, opt; epochs=20, bs=16, dev=Flux.gpu, conv=false)
    model = model |> dev
    ps = params(model)
    t = shuffleobs(wg.train)
    v = batch_ts(getobs(wg.valid))
    v = conv ? ((unsqueeze(v[1],3), v[2]) |> dev) : (v |> dev) # handle validation dimensions if conv network

    local l
    vl_prev = Inf
    for e in 1:epochs
        for d in eachbatch(t, size=bs)
            x, y = batch_ts(d)
            y = y[wg.target_idx,:,:]
            conv && (x = unsqueeze(x,3))
            x, y = x |> dev, y |> dev
            gs = gradient(ps) do 
                l = loss(model(x),y)
            end
            Flux.update!(opt, ps, gs)
        end
        l = round(l;digits=4)
        vl = round(loss(model(v[1]),v[2][wg.target_idx,:,:]); digits=4)
        println("Epoch $e/$epochs - train loss: $l, valid loss: $vl")
        # crude early-stopping
        # vl_prev < (vl - 0.001) && break 
        # vl_prev = vl
    end
    model = model |> cpu
end

@time linear = train_model!(linear, single_step_1h, opt; bs=32, epochs=20)

# ##### 24 hr
plot(single_step_24h, linear)

bar(names(train_df), linear.W[:], xrotation=30.0, legend=false, xticks=:all, tickfontsize=6)

# ### Dense
dense = Chain(
    Dense(19, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
)

@time dense = train_model!(dense, single_step_1h, opt; bs=32, epochs=20)

plot(single_step_24h, dense)


# ### Multi-step Dense
# Now we are going to use 3 historical hours to predict 1 hour in the future.
single_step_3h = WindowGenerator(3, 1, train_df, valid_df, label_columns=target);

plot(single_step_3h)

multi_step_dense = Chain(
    i -> reshape(i, :, 1, size(i)[end]), # flatten first two dimensions, but preserve batch dimension
    Dense(19*3, 32, relu),
    Dense(32, 32, relu),
    Dense(32, 1)
)

@time multi_step_dense = train_model!(multi_step_dense, single_step_3h, opt; bs=32, epochs=20)

plot(single_step_3h, multi_step_dense)

# ### Convolutional Neural Network
conv_model = Chain(
    Conv((19,3), 1=>32, relu), # need to explain why this conv pattern
    x -> Flux.flatten(x),
    Dense(32, 32, relu),
    Dense(32, 1)
)

single_step_3h = WindowGenerator(3, 1, train_df[1:60,:], valid_df, label_columns=target);

@time conv_model = train_model!(conv_model, single_step_3h, opt; bs=20, epochs=2, conv=true)


#not learning :( but dimensions are right?
# maybe convolutions arent?

# https://github.com/FluxML/Flux.jl/issues/1465


# ### Recurrent Neural Network

# ### Performance

# ### Multi-output Models

# #### Baseline

# #### Dense

# #### RNN

# #### Advanced: Residual Connections

# #### Performance

# # Multi-Step Models

# ### Baselines

# ## Single-shot Models

# ### Linear

# ### Dense

# ### CNN

# ### RNN

# ## Advanced Autoregressive model
# ### RNN

# ## Performance

# Next Steps
