using Flux, Trebuchet
using Flux.Tracker: forwarddiff

function shoot(wind, angle, weight)
  Trebuchet.shoot((wind, angle, weight))[2]
end

shoot(ps) = forwarddiff(p -> shoot(p...), ps)

Tracker.gradient(shoot, [3.0, 4.0, 1.0])
Tracker.ngradient(shoot, [3.0, 4, 1])

model = Chain(Dense(2, 64, tanh),
              Dense(64, 256, tanh),
              Dense(256, 32, tanh),
              Dense(32, 2))
θ = params(model)
opt = ADAM()

aim(wind_speed, target_dist) =
  shoot(vcat([wind_speed], Float64.(model([wind_speed, target_dist]))))

loss(wind_speed, target_dist) =
  (aim(wind_speed, target_dist) - target_dist)^2

MAX_DIST  = 500	# Maximum target distance
MAX_SPEED =  10 # Maximum wind speed

target() = (randn() * MAX_SPEED, (rand()+1)/2 * MAX_DIST)

meanloss() = mean(loss(target()...) for i = 1:100)

dataset = (target() for i = 1:100_000)
cb = Flux.throttle(() -> @show(meanloss()), 10)

Flux.train!(loss, θ, dataset, opt, cb = cb)
