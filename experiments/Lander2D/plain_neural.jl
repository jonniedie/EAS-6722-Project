using CombinedUncertainDiffEq
using ComponentArrays
using DiffEqFlux
using DifferentialEquations
using Flux: glorot_uniform, relu
using ForwardDiff
using Plots; plotlyjs()
# using ReachabilityAnalysis
using UnPack
using Serialization
using Statistics
# using Zygote


## Includes
include("../../general_utils.jl")
include("../../controller_utils.jl")
include("eom.jl")


##
function build_prob()
    # Neural network controller
    params, controller = chain(
        dense_layer( 5, 10, tanh),
        dense_layer(10, 10, tanh),
        dense_layer(10,  2),
    )
    ctrl_ax = Axis(θ=1, tgo=2)
    controller = (x->ComponentArray(x, ctrl_ax)) ∘ controller
    
    # # PD controller
    # params = ComponentArray(kp=0.01, kd=0.05)
    # # controller = (x,p,t) -> -p.kp*x[1] - p.kd*x[3]
    # controller = (x,p,t) -> -p[1]*x[1] - p[2]*x[3]

    # Assign controls to ODE function
    f = WithController(lander_2D!, controller, params)

    p = ComponentArray{Any}(
        m_dry = 1.0 ± 0.1,
        Isp = 60 ± 1,
        T = 14..16,
        thrust_on = false,
    )
    x0 = ComponentArray{Any}(
        r = (
            x = 0 ± 1,
            y = 40 ± 5,
        ),
        v = (
            x = 0 ± 5,
            y = (-10)..(-5),
        ),
        m_prop = 0.1,
        # tgo = 1.0,
    )

    # Callback functions
    hits_ground = ContinuousCallback((vars, t, integrator) -> vars.r.y, terminate!)
    # turns_on = ContinuousCallback((vars, t, integrator) -> vars.tgo, identity;
    #     affect_neg! = integrator -> (integrator.p.thrust_on=true;),
    #     repeat_nudge=Inf,
    # )
    turns_on = PresetTimeCallback([1.0], integrator -> (integrator.p.thrust_on=true;))
    burns_out = ContinuousCallback((vars, t, integrator) -> vars.m_prop, integrator -> (integrator.p.thrust_on=false;))
    cb = CallbackSet(burns_out, turns_on, hits_ground)

    return ODEProblem{true}(f, x0, (0.0, 100.0), p; callback=cb)
end


##
const widths = [1, 3, 50] #range_width.(requirements)

loss_single(v,w) = (v/w)^4

loss(sol) = mapreduce(loss_single, +, view(sol.u[end], [3:4; 1]), widths)

function run_with_params(x, p)
    @unpack ode_fun, controller = p.prob.f.f
    f! = WithController(; ode_fun, controller, params=x)
    ax_u0 = getaxes(p.prob.u0)
    ax_p = getaxes(p.prob.p)
    u0_CoV = (u,p) -> eltype(x).(ComponentArray(u, ax_u0))
    p_CoV = (u,p) -> ComponentArray(p, ax_p)
    _u0 = CombinedUncertainDiffEq.recursive_convert(CombinedUncertainDiffEq.to_distribution, p.prob.u0)
    _p = CombinedUncertainDiffEq.recursive_convert(CombinedUncertainDiffEq.to_distribution, p.prob.p)
    prob_func = function (prob, i, repeat)
        __u0 = u0_CoV(CombinedUncertainDiffEq._rand.(_u0), _p)
        __p = p_CoV(_u0, CombinedUncertainDiffEq._rand.(_p))
        kwargs = deepcopy(p.prob.kwargs)
        kwargs[:callback].discrete_callbacks[1].condition.tstops[1] = ForwardDiff.value(controller(__u0, x, 0.0).tgo)
        # prob = remake(p.prob; f=f!, kwargs=kwargs)
        remake(prob, f=f!, kwargs=kwargs, u0=__u0, p=__p)
    end
    mc_prob = EnsembleProblem(prob; prob_func=prob_func)
    sols =  solve(mc_prob, Tsit5(), EnsembleThreads(); trajectories=200)
    return mean(loss, sols.u)
    # # return koopman_expectation(loss, prob, prob.u0, prob.p, Tsit5();
    # #     sensealg=ForwardDiffSensitivity(),
    # #     u0_CoV,
    # #     p_CoV,
    # # ).u
    # return mc_expectation(loss, prob, prob.u0, prob.p, Tsit5(), EnsembleThreads();
    #     sensealg=ForwardDiffSensitivity(),
    #     trajectories=200,
    #     u0_CoV,
    #     p_CoV,
    # )
end

prob = build_prob()
gradable_run = x -> run_with_params(x, (;prob))
gradable_run2(x) = run_with_params(x, (;prob))

##
@time mc_exp = mc_expectation(loss, prob, prob.u0, prob.p, Tsit5(), EnsembleThreads(); trajectories=20_000)
# @time koop_exp = koopman_expectation(loss, prob, prob.u0, prob.p, Tsit5();
#     # quadalg=CubaDivonne(),
#     # iabstol=3e-2,
#     # ireltol=3e-2,
# )


##
@time ForwardDiff.gradient(gradable_run2, prob.f.f.params)
# Zygote.gradient(gradable_run, prob.f.f.params)
# ReverseDiff.gradient(gradable_run, prob.f.f.params)


##
loss_history = Float32[]
iter = 0
cb = function (p, l, pred=nothing)
    global iter += 1
    println("Iter: $iter | Loss: $l")
    push!(loss_history, l)
    return false
end
opt_alg = nothing
ad_type = DiffEqFlux.GalacticOptim.AutoForwardDiff()
@time out = DiffEqFlux.sciml_train(gradable_run, prob.f.f.params, opt_alg, ad_type; maxiters=300, cb)
# @time out = DiffEqFlux.sciml_train(gradable_run, out.u, opt_alg, ad_type; maxiters=200, cb)
# @time out = DiffEqFlux.sciml_train(gradable_run2, deserialize("control_params_from_work"), opt_alg, ad_type; maxiters=500, cb)
# @time out = DiffEqFlux.sciml_train(gradable_run, out.u, LBFGS(), ad_type; cb)


##
opt_prob = deepcopy(prob)
opt_prob.f.f.params .= out.u
# opt_prob.f.f.params .= deserialize("control_params_from_work")
mc_sol = mc_solve(opt_prob, opt_prob.u0, opt_prob.p, Tsit5(), EnsembleThreads(); trajectories=200)
vels = reduce(hcat, [u[end][3:4] for u in mc_sol.u])


##
function plot_in_out(mc_sol, idx_in::AbstractString, idx_out::AbstractString)
    idx_in = Symbol.(split(idx_in, "."))
    idx_out = Symbol.(split(idx_out, "."))
    in_vals = [reduce(getproperty, idx_in; init=sol[begin]) for sol in mc_sol.u]
    out_vals = [reduce(getproperty, idx_out; init=sol[end]) for sol in mc_sol.u]
    return scatter(in_vals, out_vals; legend=false)
end

##
plt_loss = plot(loss_history; yscale=:log10, legend=false, xlabel="Iteration", ylabel="Loss")

##
plt_sensitivity = plot(
    scatter([sol.prob.p.m_dry for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u], ylabel="|v.x|₂"),
    scatter([sol.prob.p.Isp for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u]),
    scatter([sol.prob.p.T for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u]),
    scatter([sol[1].v.x for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u]),
    scatter([sol[1].v.y for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u]),
    scatter([sol[1].r.x for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u]),
    scatter([sol[1].r.y for sol in mc_sol.u], [abs2(sol[end].v.x) for sol in mc_sol.u]),
    scatter([sol.prob.p.m_dry for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u], ylabel="|v.y|₂"),
    scatter([sol.prob.p.Isp for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u]),
    scatter([sol.prob.p.T for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u]),
    scatter([sol[1].v.x for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u]),
    scatter([sol[1].v.y for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u]),
    scatter([sol[1].r.x for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u]),
    scatter([sol[1].r.y for sol in mc_sol.u], [abs2(sol[end].v.y) for sol in mc_sol.u]),
    scatter([sol.prob.p.m_dry for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="m_dry", ylabel="|r.x|₂"),
    scatter([sol.prob.p.Isp for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="Isp"),
    scatter([sol.prob.p.T for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="T"),
    scatter([sol[1].v.x for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="v.x"),
    scatter([sol[1].v.y for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="v.y"),
    scatter([sol[1].r.x for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="r.x"),
    scatter([sol[1].r.y for sol in mc_sol.u], [abs2(sol[end].r.x) for sol in mc_sol.u], xlabel="r.y"),
    layout=(3,7),
    legend=false,
    markersize=1,
    axis=false,
)

##
plt_pos = plot(mc_sol; vars=(1,2), xlabel="Horizontal Position (m)", ylabel="Vertical Position (m)")

##
plt_vel = plot(
    plot(
        plot(mc_sol; vars=3, xlabel=nothing, ylabel="Horizontal Velocity (m/s)"),
        plot(mc_sol; vars=4, xlabel="Time (s)", ylabel="Vertical Velocity (m/s)"),
        layout=(2,1),
    ),
    scatter(vels[1,:], vels[2,:]; xlabel="Horizontal Velocity (m/s)", ylabel="Vertical Velocity (m/s)", ms=3),
    legend=false,
)