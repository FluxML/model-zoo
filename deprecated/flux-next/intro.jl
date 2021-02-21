# Flux vNext
# ==========

# Optimisation Basics
# -------------------

using Flux
using Flux: step!

# Using Flux is very simple. You write a program, and we'll make tweaks to that
# program so that it gets gradually better.

# What does it mean to make a program "better"? That's up to you – your program
# returns a score, called the "loss", which determines how well the program
# is doing. Flux's job is to minimise that score. For example, let's take the
# simplest possible program, one that simply returns a constant.

w = 1
loss = () -> w

loss()

# This program doesn't look very interesting, but we can still do something
# interesting with it. The core function that optimises programs for us is
# `step!`. We have to pass `step!` an *optimiser* called `Descent`; this basically
# tells Flux how agressive to be, but we'll talk more about that later.

opt = Descent(1)
step!(loss, opt)

# `step!` returns the same loss as before, `1`. But something more interesting
# has happened; try running `loss()` again.

loss()

# It went down! And if we keep calling `step!` in a loop, it'll keep going down.

for i = 1:10
  @show step!(loss, opt)
end

loss()

# Of course, this case is pretty easy: we can always improve the parameter by
# making `w` less.

w

# Here's something harder; now our `loss` is always positive, so it can't keep
# improving indefinitely. Things will stop improving once we hit the *minimum*
# of this function (which we happen to know is at $w = 0$, where $loss = 0$.)

w = 1
loss = () -> w^2

opt = Descent(0.2)
for i = 1:10
  @show step!(loss, opt)
end

w

# You can see that our loss gradually tends towards $0$, and so does $w$. Note,
# however, that Flux will never say: "Ok, we're done here, here's the best value
# for $w$." Though there are tools that can do this in simple cases, Flux is
# designed to scale to extremely complex problems where this is no longer
# possible. So we only make tweaks and it's up to you when to finish.

# Let's put these ideas towards something a little more interesting. Say we want
# to solve $5x = 10$, to find an $x$ that makes this true. What's our program?
# Well, to start with we want to take $f(x) = 5x$. Then our loss should be something like
# $f(x) - 10$, so that it measures how far the $f(x)$ is from where we want it to
# be. This doesn't quite work, however, since the loss will be low (negative) if
# $f(x)$ is `-Inf`! So we can use our squaring trick again here, to make
# sure that $f(x) - 10$ tends to zero.

x = 1 # Our initial guess
f = x -> 5x

opt = Descent(0.01)

for i = 1:10
  l = step!(opt) do
    (f(x) - 10)^2
  end
  @show l
end

# Our loss ended up being pretty low. How's our function looking?

5x

# That looks pretty good. So we're beginning to be able to use Flux to solve
# problems where we know what the *output* should look like, but we're not
# sure what the *input* should be to get there.

# You now arguably understand everything you need to do productive ML. But let's
# look over a few more examples to see how it looks in practice.

# Optimising Colours
# ------------------

# Just like Julia more generally, Flux has good support for custom types.
# This means we can carry out optimisation on things like colours!

# This example uses the excellent Colors.jl. Colors contains, among other
# things, a `colordiff` function which uses fancy colour theory algorithms to
# estimate the *perceptual* difference between two colours. We can use this
# directly in our loss function.

using Colors

target = RGB(1, 0, 0)
colour = RGB(1, 1, 1)
[target, colour]
#-
opt = Descent(0.01)

for i = 1:10
  step!(opt, target) do y
    colordiff(colour, y)
  end
end

[colour, target]

# `colour` started out white and is now red. That makes sense, as we've
# minimised the distance between the two colours. But we can also *maximize*
# with a simple minus sign.

colour1 = RGB(1, 1, 1)

for i = 1:10
  step!(opt, target) do y
    -colordiff(colour1, y)
  end
end

[colour1, target]

# Now we have green, a colour that's arguably very different from red. However,
# there's a subtlety here; notice what happens if we use a different colour as
# our starting point.

colour2 = RGB(0, 0, 1)

for i = 1:10
  step!(opt, target) do y
    -colordiff(colour2, y)
  end
end

[colour2, target]

# Now we have a dark blue! If we look directly at `colourdiff` we'll see that
# green is better.

colordiff(target, colour1), colordiff(target, colour2)

# So why do we get blue here? This is another case where it's important that
# Flux optimises programs through a series of small tweaks. In this case, even
# though green is better overall, making our colour slightly more green actually
# makes our score worse temporarily.

colordiff(target, RGB(0, 0, 0.4)), colordiff(target, RGB(0, 0.1, 0.4))

# This is known as a *local optimimum*. It's important to understand how Flux
# optimises programs and what this means for you, so we'll cover this in more
# detail in future.

# Keras in 5 lines
# -----------------

# [Working on making this an MNIST demo, but here's the gist of it.]

# Dummy data.

x = rand(10)
y = [1, 0]

# Logistic regresion.

using Flux: crossentropy

W = randn(2, 10)
b = zeros(2)

predict = x -> softmax(W * x .+ b)

opt = Descent(0.1)

loss = (x, y) -> crossentropy(predict(x), y)

step!(loss, opt, x, y)

# Multi-layer perceptron.

function dense(in, out, σ = identity)
  W = randn(out, in)
  b = zeros(out)
  x -> σ.(W * x .+ b)
end

chain(fs...) = x -> foldl((x, m) -> m(x), fs, init = x)

model = chain(dense(10, 5, relu), dense(5, 2), softmax)

# Doesn't quite work yet.

## step!(opt, x, y) do x, y
##   crossentropy(model(x), y)
## end
