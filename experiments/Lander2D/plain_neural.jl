using CombinedUncertainDiffEq
using ComponentArrays
using DiffEqFlux
using DifferentialEquations
using Flux: glorot_uniform, relu
using ForwardDiff
using Plots
using ReachabilityAnalysis
using UnPack
using Zygote


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
        dense_layer(10,  1),
    )
    controller = first ∘ controller
    
    # # PD controller
    # params = ComponentArray(kp=0.01, kd=0.05)
    # # controller = (x,p,t) -> -p.kp*x[1] - p.kd*x[3]
    # controller = (x,p,t) -> -p[1]*x[1] - p[2]*x[3]

    # Assign controls to ODE function
    f = WithController(lander_2D!, controller, params)

    p = ComponentArray{Any}(
        m_dry = 60 ± 3,
        Isp = 75 ± 5,
        T = 1100..1150,
    )
    x0 = ComponentArray{Any}(
        r = (
            x = 0.0,
            y = 160..170,
        ),
        v = (
            x = 0 ± 5,
            y = -45 ± 5,
        ),
        m_prop = 10.0,
    )

    # Callback functions
    hits_ground = ContinuousCallback((vars, t, integrator) -> vars.r.y, terminate!)
    burns_out = ContinuousCallback((vars, t, integrator) -> vars.m_prop, integrator->(integrator.p.T=0.0;))
    cb = CallbackSet(burns_out, hits_ground)

    return ODEProblem{true}(f, x0, (0.0, 100.0), p; callback=cb)
end


##
const widths = [1, 5, 10] #range_width.(requirements)

loss_single(v,w) = (v/w)^4

loss(sol) = mapreduce(loss_single, +, view(sol.u[end], [3:4; 1]), widths)

function run_with_params(x, p)
    @unpack ode_fun, controller = p.prob.f.f
    f! = WithController(; ode_fun, controller, params=x)
    prob = remake(p.prob; f=f!)
    ax_u0 = getaxes(p.prob.u0)
    ax_p = getaxes(p.prob.p)
    u0_CoV = (u,p) -> eltype(x).(ComponentArray(u, ax_u0))
    p_CoV = (u,p) -> ComponentArray(p, ax_p)
    # return koopman_expectation(loss, prob, prob.u0, prob.p, Tsit5();
    #     sensealg=ForwardDiffSensitivity(),
    #     u0_CoV,
    #     p_CoV,
    # ).u
    return mc_expectation(loss, prob, prob.u0, prob.p, Tsit5(), EnsembleSerial();
        sensealg=ForwardDiffSensitivity(),
        trajectories=100,
        u0_CoV,
        p_CoV,
    )
end

prob = build_prob()
gradable_run(x) = run_with_params(x, (;prob))

##
@time mc_exp = mc_expectation(loss, prob, prob.u0, prob.p, Tsit5(), EnsembleSerial(); trajectories=20_000)
# @time koop_exp = koopman_expectation(loss, prob, prob.u0, prob.p, Tsit5();
#     # quadalg=CubaDivonne(),
#     # iabstol=3e-2,
#     # ireltol=3e-2,
# )


##
@time ForwardDiff.gradient(gradable_run, prob.f.f.params)
# Zygote.gradient(gradable_run, prob.f.f.params)
# ReverseDiff.gradient(gradable_run, prob.f.f.params)


##
loss_history = Float32[]
cb = function (p, l, pred=nothing)
    push!(loss_history, l)
    return false
end
opt_alg = nothing
ad_type = DiffEqFlux.GalacticOptim.AutoForwardDiff()
@time out = DiffEqFlux.sciml_train(gradable_run, prob.f.f.params, opt_alg, ad_type; maxiters=500, cb)
@time out = DiffEqFlux.sciml_train(gradable_run, out.u, LBFGS(), ad_type; cb)


##
opt_prob = deepcopy(prob)
opt_prob.f.f.params .= out.u
mc_sol = mc_solve(opt_prob, opt_prob.u0, opt_prob.p, Tsit5(), EnsembleSerial(); trajectories=200)
vels = reduce(hcat, [u[end][3:4] for u in mc_sol.u])