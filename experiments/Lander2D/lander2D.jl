using CombinedUncertainDiffEq
using ComponentArrays
using ConcreteStructs
using DifferentialEquations
using DiffEqUncertainty
using Plots
using UnPack


## Functions
# Two-dimensional vector with x and y components
Vec2(; x=0.0, y=0.0) = ComponentArray(; x, y)

# Dynamics
function simple_lander_2D!(D, vars, params, t; T=0.0, θ=0.0)
    @unpack r, v = vars
    @unpack g, m = params

    Fx, Fy = T .* sincos(θ)

    D.r .= v
    D.v.x = Fx/m
    D.v.y = -g + Fy/m

    return nothing
end

# For applying control inputs functions through keyword arguments
function apply_inputs(func; kwargs...)
    simfun(dx, x, p, t) = func(dx, x, p, t; map(f->f(x,p,t), (;kwargs...))...)
    return simfun
end

# Neural stuff
dense_layer(n_in, n_out) = ComponentArray(W=rand(n_out, n_in), b=zeros(Float32, n_out))

# Controls
@concrete struct Saturated <: Function
    f
    lb
    ub
end
(f::Saturated)(args...; kwargs...) = clamp(f.f(args...; kwargs...), f.lb, f.ub)

function dense_neural_controller(layer_size, act_fun=identity, out_fun=only)
    layers = ComponentArray(;
        L1 = dense_layer(2, layer_size),
        L2 = dense_layer(layer_size, 1),
    )

    f = function (x, p, t)
        @unpack L1, L2 = p
        out = act_fun.(L2.W * act_fun.(L1.W * x .+ L1.b) .+ L2.b)
        return out_fun(out)
    end

    return layers, f
end

soft_sign(x, α=1e2) = 1 / (1 + exp(-α*x))
≤ₛ(x, y) = soft_sign(y - x)

# Loss function for training
meets_requirements(sol, reqs) = all(reqs.lower .≤ₛ sol.u[end] .≤ₛ reqs.upper)

# Return true if vehicle hits ground
hits_ground(vars, t, integrator) = vars.r.y


## Single-run Problem Definition
# Control functions
thrust = function (x, p, t)
    @unpack mag, tstart, duration = p.control.thrust
    return if tstart ≤ t ≤ tstart+duration
        mag
    else
        zero(mag)
    end
end
angle = Saturated((x, p, t) -> -0.1x.v.x - 0.005x.r.x, -π/2, π/2)

# Initial conditions
ic = ComponentArray{Float64}(
    r = Vec2(y=127),
    v = Vec2(x=15, y=-50),
)

# Parameters
p = ComponentArray(;
    g = 9.80665,
    m = 50,
    control = (
        thrust = (
            mag = 1000,
            tstart = 0,
            duration = 5,
        ),
    ),
)

# ODE callback function (stop simulation when vehicle hits the ground)
cb = ContinuousCallback(hits_ground, terminate!, save_positions=(false,false))

# Single-run problem
sim_fun = apply_inputs(simple_lander_2D!; T=thrust, θ=angle)
prob = ODEProblem(sim_fun, ic, (0.0, 100.0), p; callback=cb)


## Koopman Problem Definition
uncertain_ic = ComponentArray{Any}(
    r = (
        x = 0.0,
        y = 120..150,
    ),
    v = (
        x = 0 ± 20,
        y = -45 ± 5,
    ),
)

uncertain_p = ComponentArray{Any}(;
    g = 9.80665,
    m = 50 ± 3,
    control = (
        thrust = (
            mag = 1000 ± 10,
            tstart = 0.0..0.5,
            duration = 5 ± 0.1,
        ),
    ),
)

requirements = ComponentArray(
    lower = (
        r = (
            x = -50, 
            y = -50,
        ),
        v = (
            x = -2,
            y = -5,
        ),
    ),
    upper = (
        r = (
            x = 50, 
            y = 50,
        ),
        v = (
            x = 2,
            y = 5,
        ),
    )
)

##
koop_sol = koopman_expectation(
    sol -> meets_requirements(sol, requirements),
    prob,
    uncertain_ic,
    uncertain_p, 
    Tsit5();
    quad_alg = CubaDivonne(),
)