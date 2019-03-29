using Flux, Trebuchet
using Flux.Tracker: forwarddiff
using Statistics: mean
using Random

lerp(x, lo, hi) = x*(hi-lo)+lo

function shoot(wind, angle, weight)
  Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

shoot(ps) = forwarddiff(p -> shoot(p...), ps)

Random.seed!(0)

model = Chain(Dense(2, 16, σ),
              Dense(16, 64, σ),
              Dense(64, 16, σ),
              Dense(16, 2)) |> f64

θ = params(model)

function aim(wind, target)
  angle, weight = model([wind, target])
  angle = σ(angle)*90
  weight = weight + 200
  angle, weight
end

distance(wind, target) =
  shoot(Tracker.collect([wind, aim(wind, target)...]))

function loss(wind, target)
  try
    (distance(wind, target) - target)^2
  catch e
    # Roots.jl sometimes give convergence errors, ignore them
    param(0)
  end
end

DIST  = (20, 100)	# Maximum target distance
SPEED =   5 # Maximum wind speed

target() = (randn() * SPEED, lerp(rand(), DIST...))

meanloss() = mean(sqrt(loss(target()...)) for i = 1:100)

opt = ADAM()

dataset = (target() for i = 1:100_000)
cb = Flux.throttle(() -> @show(meanloss()), 10)

Flux.train!(loss, θ, dataset, opt, cb = cb)
