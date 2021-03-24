# # Time series forecasting
# This tutorial is inspired by a [Tensorflow](https://www.tensorflow.org/tutorials/structured_data/time_series) Time Series Forecasting Tutorial.  
# It implements the data analysis, plotting, and model training from scratch using Julia tools such as Plots.jl, Flux, and DataFrames.jl.


# ## Setup
# Importing the usual suspects. Two interesting notes: 
# * StatsPlots re-exports Plots.jl functions, so we don't need that separately
# * MLDataPattern is useful for preprocessing data for ML in a lazy (read: non-memory intensive) way. We will be seeing more of it
using ZipFile
using CSV
using DataFrames
using StatsPlots
using Dates
using FFTW
using CUDA
using Flux
using MLDataPattern

using Random: seed!
using Statistics: mean, std
using Flux: unsqueeze

# Set a seed to make this reproducible.
seed!(4231)

ENV["LINES"] = 20;

# Disallow scalar indexing for CUDA - prevents slow accesses to GPU memory
CUDA.allowscalar(false)

# ## The weather dataset
# We are going to be using weather data for 7 years from the Max Planck Institute: https://www.bgc-jena.mpg.de/wetter/

function download_data(; fname="jena_climate_2009_2016.zip")
    DATA_PATH = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    isfile(fname) || download(DATA_PATH, fname)

    zip = ZipFile.Reader(fname) 
    csv = zip.files[1]
    df = CSV.File(csv) |> DataFrame
    close(zip)
    return df
end

# More discussion on reading zipped CSVs here: https://juliadata.github.io/CSV.jl/stable/index.html#Example:-reading-from-a-zip-file-1

df = download_data();

df = df[6:6:end, :]

df[!,"Date Time"] = Dates.DateTime.(df[:,"Date Time"], "dd.mm.yyyy HH:MM:SS"); 
# More info on working with dates and times: https://en.wikibooks.org/wiki/Introducing_Julia/Working_with_dates_and_times

col = ["Date Time", "T (degC)", "p (mbar)", "rho (g/m**3)"]

df[:,col]

# One easy way to plot data from a datafrom is with the StatsPlots `@df` macro
@df df plot(cols(1), cols(2:4); layout=(3, 1)) 

@df df[1:480,:] plot(cols(1), cols(2:4); layout=(3, 1))

# ## Inspect and cleanup
@show describe(df)

df."p (mbar)"; #cool that this works

# Need to replace bad database values 
replace!(df[!,"wv (m/s)"], -9999.0 => 0); 
replace!(df[!,"max. wv (m/s)"], -9999.0 => 0);

@show describe(df)

# ## Feature engineering
# Bucket wind direction and speed
histogram2d(df[!,"wd (deg)"], df[!,"wv (m/s)"], bins=(75,75), xlabel="wd (deg)", ylabel="wv (m/s)")

# It would be better to decompose this into X and Y velocities
wd_rad = df[!,"wd (deg)"] * π / 180
df.Wx = df[!,"wv (m/s)"] .* cos.(wd_rad)
df.Wy = df[!,"wv (m/s)"] .* sin.(wd_rad)
df."max Wx" = df[!,"max. wv (m/s)"] .* cos.(wd_rad)
df."max Wy" = df[!,"max. wv (m/s)"] .* sin.(wd_rad);

histogram2d(df.Wx, df.Wy, bins=(75,75), xlabel="Wind X [m/s]", ylabel="Wind Y [m/s]")

# Modify timestamp into a continuous "time of day" signal to deal with periodicity.
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

# If you don't know the frequency to choose for your periodicity, you can take the FFT. 
# Here you can see that the yearly and daily periodicities are the most prominent.
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
# Drop columns that won't be used further.
select!(df, Not([:("wv (m/s)"),:("max. wv (m/s)"), :("wd (deg)"), :("Date Time")]));

column_indices = pairs(names(df))
indices_columns = Dict(value => key for (key, value) in column_indices)
df = convert.(Float32, df) # Don't need high precision; reduces errors later on when using Params - gradients are Float32

# Split the data into a training and validation set
n = size(df)[1]
train_df = df[1:round(Int,n*0.7, RoundDown),:]
valid_df = df[round(Int,n*0.7, RoundUp):round(Int,n*0.9, RoundDown),:]
test_df = df[round(Int,n*0.9, RoundUp):end,:]; # matching TF tutorial exactly, can also use partition

num_features = size(df,2)

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
# We will define our own WindowGenerator, some constructors, and plotting functions. The data from the WindowGenerator will be used in training.
include("window_generator.jl")

h = 6; # historical window length
f = 1; # future window length

# WindowGenerator makes use of MLDataPattern's `slidingwindow` to generate the windows. It is good at flexibly generating sequences data, though requires a bit of mind-bending to fully understand how it generates sequences.
# [slidingwindow docs](https://mldatapatternjl.readthedocs.io/en/latest/documentation/dataview.html?highlight=slidingwindow#labeled-windows)
slidingwindow(i -> i+h:i+h+f-1, Array(train_df)', h, stride=1)

# WindowGenerator is implemented to make sequence-generating simpler. It uses `slidingwindow` behind the scenes to create sequences.
@doc WindowGenerator

# Create a WindowGenerator
wg = WindowGenerator(6, 1, train_df, valid_df, label_columns="T (degC)");

# Plotting methods are implemented for raw data and with predictions.
plot(wg)


# We will also make use of the utility function for batching lazily-evaluated timeseries from slidingwindow
include("batch_ts.jl")


# To understand better what all the above code does, let's look at an imitation training loop and look at everything's dimensions
practice_df = train_df[1:10,:]
a = slidingwindow(i -> i+h:i+h+f-1, Array(practice_df)', h, stride=1)

for i in eachbatch(shuffleobs(a), size=2)
    (x,y) = batch_ts(i)
    @show x
    @show y
    println()
end
#=
That took the data at 10 timestamps (`train_df[1:10,:]`) and threw it into 4 'observations' using all the data it could while requesting 6 historical and 1 target point. This works out to the following:
* (1,2,3,4,5,6), 7
* (2,3,4,5,6,7), 8
* (3,4,5,6,7,8), 9
* (4,5,6,7,8,9), 10

It then shuffles these obserations, and lazily batches them into 2 batches each of size 2.
=#

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

# Demonstrate the 'training error' for this model, although there is no actual training.
function run_single_step_baseline(wg, model)
    preds = Float32[]
    reals = Float32[]
    for (x,y) in wg.train
        val = model(x)[1]
        push!(preds, val)
        push!(reals, y[model.label_index]) 
    end

    l = loss(preds, reals)
    return l
end

run_single_step_baseline(single_step_1h, baseline_model)

# ### Baseline - 24h
# Let's try to predict the next hour's value for 24 hours
single_step_24h = WindowGenerator(24, 24, train_df, valid_df, label_columns=target; offset=1);

plot(single_step_24h, baseline_model)


# ### Linear Models
# ##### 1 hour
# Set up a fairly generic training loop for the rest of the models. It takes in a model, and a WindowGenerator object, and uses the ADAM optimizer to train the model.
opt = Flux.Optimise.ADAM(0.01)

function train_model!(model, wg::WindowGenerator, opt; epochs=20, bs=16, dev=Flux.gpu)
    model = model |> dev
    ps = params(model)
    t = shuffleobs(wg.train)
    v = batch_ts(getobs(wg.valid))

    local l
    vl_prev = Inf
    for e in 1:epochs
        for d in eachbatch(t, size=bs)
            x, y = batch_ts(d)
            y = y[wg.target_idx,:,:]
            x, y = x |> dev, y |> dev
            gs = gradient(ps) do 
                l = loss(model(x),y)
            end
            Flux.update!(opt, ps, gs)
        end
        l = round(l;digits=4)
        vl = round(loss(model(v[1]),v[2][wg.target_idx,:,:]); digits=4)
        println("Epoch $e/$epochs - train loss: $l, valid loss: $vl")
        #=
        crude early-stopping
        vl_prev < (vl - 0.001) && break 
        vl_prev = vl 
        =#
    end
    model = model |> cpu
end

# Now, train the linear model 
linear = Dense(size(single_step_1h.train[1][1],1), 1; initW=Flux.glorot_uniform, initb=Flux.zeros)

@time linear = train_model!(linear, single_step_1h, opt; bs=32, epochs=20)

# ##### 24 hours
# Show the model's predictions for each point in the window.
plot(single_step_24h, linear)

# Visualize the weights of the linear regression. Naturally, since we are trying to predict future temperatures, we would expect that the current temperature is on of the most useful features. 
# This is shown by it having a high positive weight.
bar(names(train_df), linear.W[:], xrotation=30.0, legend=false, xticks=:all, tickfontsize=6)

# ### Dense
# A model with 2 Fully-connected layers and relu non-linearities.
dense = Chain(
    Dense(19, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
)

@time dense = train_model!(dense, single_step_1h, opt; bs=32, epochs=20)

# Dense model's predicions
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

# Show the model's 1 hour predicion using 3 hours of historical data.
plot(single_step_3h, multi_step_dense)

# ### Convolutional Neural Network

# conv_model = Chain(
#     Conv((19,), 3=>32, relu), # need to explain why this conv pattern
#     x -> Flux.flatten(x),
#     Dense(32, 32, relu),
#     Dense(32, 1),
#     x -> unsqueeze(x, 1)
# )

conv_model = Chain(
    x -> permutedims(x, [2,1,3]), # put data in NTime x NCovariates X NBatch https://github.com/FluxML/Flux.jl/issues/1465
    Conv((3,), 19=>32, relu), # convolve over 3 inputs - 19 variables -> 32 filters
    x -> Flux.flatten(x),
    Dense(32, 32, relu),
    Dense(32, 1),
    x -> unsqueeze(x, 1)
)

single_step_3h = WindowGenerator(3, 1, train_df, valid_df, label_columns=target);

@time conv_model = train_model!(conv_model, single_step_3h, opt; bs=32, epochs=20)

# Show the model's 1 hour predicion using 3 hours of historical data.
plot(single_step_3h, conv_model)


# ### Recurrent Neural Network

rnn_single_24h = Chain(
    LSTM(19, 32), # input features * output dimensionality
    Dense(32, 1),
    x->reshape(x,:) # lstm output * single number
)

# Flux uses vectors of observations for RNNs, rather than the bulk tensor
a = [rand(Float32, 19) for i in 1:24]
b = rnn_single_24h.(a)
a[1]
rnn_single_24h(a[1])

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
