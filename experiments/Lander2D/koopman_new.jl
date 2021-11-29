using CombinedUncertainDiffEq
using ComponentArrays
using ConcreteStructs
using Cubature
using DifferentialEquations
using DiffEqUncertainty
using LinearAlgebra
using Plots; plotlyjs()
using Statistics
using UnPack


## Functions
mid(x) = x
mid(x::Interval) = (x.lo + x.hi) / 2

range_width(x) = 0
range_width(x::Interval) = x.hi - x.lo

function soft_step(x; halfwidth=0.1)
    if x < -halfwidth
        zero(x)
    elseif x > halfwidth
        one(x)
    else
        typeof(x)((sinpi(x/halfwidth/2) + 1)/2)
    end
end

function soft_between(x, lo, hi; kwargs...)
    sgn_lo = soft_step(x-lo; kwargs...)
    sgn_hi = soft_step(hi-x; kwargs...)
    return sgn_lo * sgn_hi
end

# ODE function for 2D lander problem
function lander_2D!(D, vars, params, t; T=0.0, θ=0.0)
    @unpack r, v, m_prop = vars
    @unpack g, Isp, m_dry = params

    m = m_dry + m_prop
    Fx = T*sin(θ)
    Fy = T*cos(θ)

    D.r.x = v.x
    D.r.y = v.y
    D.v.x = Fx/m
    D.v.y = -g + Fy/m
    D.m_prop = -T / (Isp * g)

    return nothing
end

# Thrust control
constant_trust(vars, p, t) = (vars.m_prop > 0) * p.control.thrust_mag

# Angle control
function angle_pd(vars, params, t)
    @unpack r, v = vars
    @unpack kp, kd = params.control
    return -kd*v.x - kp*r.x
end


## Build problem
function build_prob()
    # Initial conditions
    ic = ComponentArray{Any}(
        r = (
            x = 0.0,
            y = 120..150,
        ),
        v = (
            x = 0 ± 20,
            y = -45 ± 5,
        ),
        m_prop = 20.0,
    )

    # Parameters
    p = ComponentArray{Any}(;
        g = 9.80665,
        m_dry = 70 ± 3,
        Isp = 60 ± 5,
        control = (
            kp = 0.01,
            kd = 0.05,
            thrust_mag = 1400..1450,
        ),
    )

    # Callback functions
    hits_ground = ContinuousCallback((vars, t, integrator) -> vars.r.y, terminate!)
    burns_out = ContinuousCallback((vars, t, integrator) -> vars.m_prop, identity)

    f! = WithControls(lander_2D!; T=constant_trust, θ=angle_pd)
    cb = CallbackSet(burns_out, hits_ground)
    return ic, p, ODEProblem(f!, mid.(ic), (0.0, 100.0), mid.(p); callback=hits_ground)
end

##
const ic, p, ODE_prob = build_prob();

##
# Requirement functions
const requirements = ComponentArray(
    x = -0.5..0.5,
    y = -10..0,
)
velocity_req(v) = (v[1] ∈ requirements[1]) && (v[2] ∈ requirements[2])
meets_requirements(sol) = velocity_req(sol.u[end].v)
function soft_meets_requirements(sol; halfwidth=0.1)
    v = sol.u[end].v
    return mapreduce(*, v, requirements) do v, req
        Δ = (req.hi - req.lo) * halfwidth
        soft_between(v, req.lo-Δ, req.hi+Δ; halfwidth=Δ)
    end
    return prod(soft_between.(v, first.(requirements).v.-halfwidth, last.(requirements).v.+halfwidth; halfwidth))
end

const widths = [1, 20] #range_width.(requirements)
g_single = (v,w) -> (v/w)^2
g = sol -> mapreduce(g_single, +, sol.u[end].v, widths)
g_vec = (vx,vy) -> g((;u=[(;v=[vx,vy])]))
plotify_obj(f) = (vx,vy) -> f((;u=[(;v=[vx,vy])]))


## Simulate
@time mc_sol = mc_solve(ODE_prob, ic, p, Tsit5(), EnsembleThreads(); trajectories=5000);
vx = [sol[3, end] for sol in mc_sol.u]
vy = [sol[4, end] for sol in mc_sol.u]


## Solve
@time mc_exp = mc_expectation(g, ODE_prob, ic, p, Tsit5(), EnsembleSerial(); trajectories=200000)
@time koop_exp = koopman_expectation(g, ODE_prob, ic, p, Tsit5();
    quadalg=CubaDivonne(),
    # iabstol=3e-2,
    # ireltol=3e-2,
)


## Plot Simulation
plot(mc_sol, vars=(1,2))
# color = 2 .- Int.(meets_requirements.(mc_sol.u))
# scatter(vx, vy, markercolor=color, legend=false, markersize=1)


##
run_with_trajectories(trajectories) = mc_expectation(exp_fun, ODE_prob, ic, p, Tsit5(), EnsembleThreads(); trajectories)
function running_expectation(trajectories)
    mc_sol = mc_solve(ODE_prob, ic, p, Tsit5(), EnsembleThreads(); trajectories)
    outs = g.(mc_sol.u)
    return cumsum(outs) ./ (1:length(outs))
end


## Plot
plot()
# contour!(-1.5:0.1:1.5, -60:0, plotify_obj(exp_fun), colorbar=false, colorbartitle="g(x)")
# scatter!(vx, vy, marker_z=1 .- plotify_obj(meets_requirements).(vx, vy), markersize=1, color=:ivory, markerstrokewidth=0, legend=false, size=(600,400), colorbar=false)
scatter!(vx, vy, marker_z=plotify_obj(meets_requirements).(vx, vy), markersize=1, color=:RdYlGn_4, markerstrokewidth=0, legend=false, size=(600,400), colorbar=false)
plot!(reduce(CombinedUncertainDiffEq.:×, requirements), opacity=0, color=3, linecolor=3, linealpha=1, linewidth=2)
# xlims!(-1.5, 1.5)
# ylims!(-60, 0)
xlabel!("Horizontal Velocity (m/s)")
ylabel!("Vertical Velocity (m/s)")
title!("Velocity Requirements")


##  Running expectation of Monte Carlos
i = 200:200:200_000
mc_runs = reduce(hcat, running_expectation(i[end]) for j in 1:20)

## Plot Monte Carlo vs Koopman
plot(i, mc_runs[i,:];
    legend = false,
    # label = "Monte Carlo",
    primary = false,
)

hline!([koop_exp.u], lw=2, label="koopman", legend=:bottomright)
# ylims!(0.095, 0.11)
# ylims!(0.065, 0.1)
title!("Convergence of Monte Carlo Solution")
xlabel!("Number of Iterations")
ylabel!("Expectation")


##
using DiffEqFlux