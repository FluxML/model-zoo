# Generates an AR(p) process with coefficients `ϕ`.
# `ϕ` should be provided as a vector and it represents the coefficients of the AR model.
# Hence the order of the generated process is equal to the length of `ϕ`.
# `s` indicates the total length of the series to be generated.
function generate_process(ϕ::AbstractVector{Float32}, s::Int)
    s > 0 || error("s must be positive")
    # Generate white noise
    ϵ = randn(Float32, s)
    # Initialize time series
    X = zeros(Float32, s)
    p = length(ϕ)
    X[1] = ϵ[1]
    # Reverse the order of the coefficients for multiplication later on
    ϕ = reverse(ϕ) 
    # Fill first p observations
    for t ∈ 1:p-1
        X[t+1] = X[1:t]'ϕ[1:t] + ϵ[t+1]
    end
    # Compute values iteratively
    for t ∈ p+1:s
        X[t] = X[t-p:t-1]'ϕ + ϵ[t]
    end
    X
end

# Create batches of a time series `X` by splitting the series into
# sequences of length `s`. Each new sequence is shifted by `r` steps.
# When s == r,  the series is split into non-overlapping batches.
function batch_timeseries(X, s::Int, r::Int)
    r > 0 || error("r must be positive")
    # If X is passed in format T×1, reshape it
    if isa(X, AbstractVector)       
        X = permutedims(X)
    end
    T = size(X, 2)
    s ≤ T || error("s cannot be longer than the total series")
    # Ensure uniform sequence lengths by dropping the first observations until
    # the total sequence length matches a multiple of the batchsize
    X = X[:, ((T - s) % r)+1:end]   
    [X[:, t:r:end-s+t] for t ∈ 1:s] # Output
end

