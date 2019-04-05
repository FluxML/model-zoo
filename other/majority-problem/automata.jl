using Images

struct State
  st::Vector{Bool}
end

Base.length(st::State) = length(st.st)

Base.getindex(st::State, i::Integer) = st.st[mod(i-1, length(st))+1]

Base.getindex(st::State, i::AbstractVector) = map(i -> st[i], i)

State(n::Integer) = State([rand(Bool) for _ = 1:n])

step(st::State, rule) =
  State([rule(st, i) for i = 1:length(st)])

isconverged(st::State) = all(st.st) || !any(st.st)

function outcome(st::State)
  isconverged(st) || error("State is not converged")
  return all(st.st)
end

function converge(st, rule; max = 100)
  isconverged(st) && return outcome(st)
  for i = 1:max
    st = step(st, rule)
    isconverged(st) && return outcome(st)
  end
  return
end

vote(x) = count(x) > length(x)÷2

function accuracy(rule; width = 100, tests = 1000, max = 100)
  s = 0
  for _ = 1:tests
    st = State(width)
    s += vote(st.st) == converge(st, rule, max = max)
  end
  return s / tests
end

function image(st, rule; steps = length(st))
  im = falses(steps, length(st))
  im[1, :] = st.st
  for i = 2:steps
    st = step(st, rule)
    im[i, :] = st.st
  end
  return Gray.(im)
end

# Rules

neighbourhood(st, i, radius) = st[(-radius:radius).+i]

function majority(s, i; radius = 3)
  x = neighbourhood(s, i, radius)
  sum(x) > length(x)÷2
end

function rule110(s, i)
  p, q, r = neighbourhood(s, i, 1)
  ((1 + p)*q*r + q + r) % 2 |> Bool
end

function shuffle(s, i)
  if s[i]
    s[i] + s[i-1] + s[i-3] > 1
  else
    s[i] + s[i+1] + s[i+3] > 1
  end
end

st = State(1000)
out = converge(st, shuffle)
image(st, shuffle)

accuracy(shuffle, tests = 1000)
