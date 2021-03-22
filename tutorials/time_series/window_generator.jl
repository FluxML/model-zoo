import StatsPlots: plot

"""
Calculates the windows used for modeling, the target index, and features useful for plotting.
"""
mutable struct WindowGenerator
    train                       # training windows
    valid                       # validation windows
    h                           # historical steps in window
    f                           # target steps in window
    label_indices               # indices for plotting predictions in a window
    target_idx::Array{Int, 1}   # indices to be predicted
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