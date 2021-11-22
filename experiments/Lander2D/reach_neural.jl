using ComponentArrays
using DiffEqFlux
using Flux
using LinearAlgebra
import ReachabilityAnalysis.IntervalArithmetic
using NeuralNetworkAnalysis
using NeuralVerification: Network, Layer, ReLU, Sigmoid, Tanh, Id
using Plots
import Polyhedra
using ReachabilityAnalysis
using StaticArrays
using Symbolics
using UnPack

const NNA = NeuralNetworkAnalysis


## Includes
include("../../general_utils.jl")
include("../../controller_utils.jl")
include("eom.jl")


## Controller definitions
# Neural network controller
nn_controller = Network([
    make_layer( 9, 20, ReLU()),
    make_layer(20, 20, ReLU()),
    make_layer(20,  1)
])

# PD controller
pd = PD(kp=0.01, kd=0.05)
pd_controller = BlackBoxController(pd)
pd_preprocessing = NNA.FunctionPreprocessing(x -> @view x[SVector(1,3)])



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
H = HybridSystem(; automaton, modes, resetmaps=identity.(resetmaps))


## Initial Conditions
x0 = Any[
    0.0,        # rx
    120..150,   # ry
    0 ± 20,     # vx
    -45 ± 5,    # vy
    15.0,       # m_prop
    70 ± 3,     # Isp
    60 ± 5,     # m_dry
    900..950,   # T
    0.0,        # θ
] |> to_set |> concretize

ic = [(1, x0)]


##
ivp = InitialValueProblem(H, ic)
postprocessing = first
vars_idx = Dict(:state_vars=>1:8, :control_vars=>9)
period = 0.01


## Run single PD controlled sim
pd_plant = ControlledPlant(ivp, pd_controller, vars_idx, period;
    preprocessing = pd_preprocessing,
    postprocessing,
)
alg = TMJets(orderT=8, orderQ=1)
alg_nn = BlackBoxSolver()

ReachabilityAnalysis._check_dim(sys) = true
LazySets.dim(x::Vector{<:Tuple}) = LazySets.dim(only(x)[2])

sol = solve(pd_plant; T=15.0, alg, alg_nn)
reach_sol = sol[1];


## Optimize PD controller



##
nn_plant = ControlledPlant(ivp, nn_controller, vars_idx, period; postprocessing)

alg = TMJets(orderT=8, orderQ=1)
alg_nn = BlackBoxSolver()

sol = solve(nn_plant; T=15.0, alg, alg_nn)
reach_sol = sol[1];