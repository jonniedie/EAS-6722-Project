using ComponentArrays
using ConcreteStructs
using DiffEqFlux
using Flux: glorot_uniform, relu
using LinearAlgebra
import ReachabilityAnalysis.IntervalArithmetic
using NeuralNetworkAnalysis
using NeuralVerification: Network, Layer, ReLU, Sigmoid, Tanh, Id
using Plots
import Polyhedra
using ReachabilityAnalysis
using Setfield
using StaticArrays
using Symbolics
using UnPack

const NNA = NeuralNetworkAnalysis

## Includes
include("../../general_utils.jl")
include("../../controller_utils.jl")
include("eom_reach.jl")

# # Neural network controller
# params, controller = chain(
#     dense_layer( 9, 20, tanh),
#     dense_layer(20, 20, tanh),
#     dense_layer(20,  1),
# )
# controller = first ∘ controller

# PD controller
params = ComponentArray(kp=0.01, kd=0.05)
controller = (x,p,t) -> -p.kp*x[1] - p.kd*x[3]

f_on! = WithController(lander_thrust_on!, controller, params)
f_off! = WithController(lander_thrust_off!, controller, params)
f_terminated! = WithController(lander_terminated!, controller, params)


##
# Symobics
vars = @variables rx, ry, vx, vy, m_prop, Isp, m_dry, T, θ

# Guard
above_ground = HPolyhedron([ry > 0, m_prop ≥ 0.1], vars)
burnt_out = HPolyhedron([ry > 0, m_prop ≤ 0.1], vars)
below_ground = HalfSpace(ry ≤ 0, vars)

# Modes
thrust_on = @system(x'=f_on!(x), x ∈ above_ground, dims=9)
thrust_off = @system(x'=f_off!(x), x ∈ burnt_out, dims=9)
terminated = @system(x'=f_terminated!(x), x ∈ below_ground, dims=9)
sim_modes = [thrust_on, thrust_off, terminated]

# Mode Transitions
automaton = LightAutomaton(3)
hits_ground = @map(x -> x, dim:9, x ∈ below_ground)
burns_out = @map(x -> x, dim:9, x ∈ HalfSpace(m_prop ≤ 0.1, vars))
resetmaps = Any[]
# Transition 1: thrust on -> thrust off
add_transition!(automaton, 1, 2, 1)
push!(resetmaps, burns_out)
# Transition 2: thrust on -> terminated
add_transition!(automaton, 1, 3, 2)
push!(resetmaps, hits_ground)
# Transition 3: thrust off -> terminated
add_transition!(automaton, 2, 3, 3)
push!(resetmaps, hits_ground)

# System Definition
H = HybridSystem(; automaton, modes=sim_modes, resetmaps=identity.(resetmaps))


## Initial Conditions
x0 = Any[
    0.0,        # rx
    120..150,   # ry
    0 ± 20,     # vx
    -45 ± 5,    # vy
    10.0,       # m_prop
    70 ± 3,     # Isp
    60 ± 5,     # m_dry
    900..950,   # T
    0.0,        # θ
] |> to_set |> concretize

ic = [(1, x0)]


## Solve plain problem
ivp = InitialValueProblem(H, ic)
# alg = TMJets(; orderT=9, disjointness=BoxEnclosure(), maxsteps=10000)
# sol = solve(ivp, alg; T=10.0);


## Optimization setup
function solve_with_params(x, p)
    @unpack prob, alg, T = p
    ivp = @set prob.s.modes[1].f.params = x
    Tx = eltype(x)
    alg = TMJets(;
        orderT=alg.orderT,
        abstol=Tx(alg.abstol),
        minabstol=Tx(alg.minabstol),
        ε=Tx(alg.ε),
        δ=Tx(alg.δ),
    )
    return solve(ivp, alg; tspan=Tx.((0, T)))
end


##
p = (
    prob = ivp,
    alg = TMJets(;
        orderT=6,
        # orderQ=1,
        # maxsteps=100,
        # validatesteps=5,
        # disjointness=BoxEnclosure(),
        # minabstol=1e-10,
        # adaptive=false,
        abstol=1e-7,
    ),
    T = 10.0,
)

@time sol = solve_with_params(ComponentArray(kp=0.01, kd=0.2), p);
# @time sol = solve_with_params(params, p);
plot(sol[1], vars=(1,2), aspect_ratio=1)


##
requirements = convert(Hyperrectangle, (-2..2)×(-5..10))
function loss(x, p)
    on_ground = solve_with_params(x, p)[2][end].X
    vels = project(on_ground, (3,4))
    return area(vels ∩ requirements) / area(vels)
end