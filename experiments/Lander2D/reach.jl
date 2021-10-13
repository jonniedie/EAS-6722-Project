
using ComponentArrays
using LinearAlgebra
using NeuralNetworkAnalysis
using ReachabilityAnalysis
using Symbolics
using UnPack


## Constants and Symobics
const g = 9.80665
vars = @variables rx, ry, vx, vy, m_prop, Isp, m_dry, T, t


## General Utilities
# Convert from Intervals and numbers to LazySets
to_set(a::AbstractArray) = reduce(×, to_set.(a))
to_set(i::IntervalArithmetic.Interval) = Interval(i)
to_set(s::LazySet) = s
to_set(x) = Singleton([x])

# Make a ComponentArray of the layers of a neural network
function to_componentarray(network)
    layers = network.layers
    pairs = map(enumerate(layers)) do (i, layer)
        s = Symbol("L", i)
        s => (W=layer.weights, b=layer.bias)
    end
    return ComponentArray(NamedTuple(pairs))
end

function set_parameters!(network, values)
    for (i, key) in enumerate(valkeys(values))
        layer = values[key]
        network.layers[i].weights .= layer.W
        network.layers[i].bias .= layer.b
    end
end


## ODE Functions
function make_ode_functions(controls)
    @unpack kp, kd = controls

    function lander_thrust_on!(dx, x, p, t)
        # Unpack variables
        rx, ry, vx, vy, m_prop, Isp, m_dry, T, t = x

        # Intermediate parameters
        m = m_dry + m_prop
        θ = -kd*vx - kp*rx
        Fx, Fy = T .* (sin(θ), cos(θ))

        # State derivatives
        dx[1] = vx              # rx
        dx[2] = vy              # ry
        dx[3] = Fx/m            # vx
        dx[4] = -g + Fy/m       # vy
        dx[5] = -T/(Isp * g)    # m_prop
        dx[6:8] .= zero(x[1])   # Isp, m_dry, T
        dx[9] = one(x[1])      # t
        return nothing
    end

    function lander_thrust_off!(dx, x, p, t)
        # Unpack variables
        rx, ry, vx, vy, m_prop, Isp, m_dry, T, t = x

        # State derivatives
        dx .= zero(x[1])
        dx[1] = vx              # rx
        dx[2] = vy              # ry
        dx[4] = -g              # vy
        dx[end] = one(x[1])     # t
        return nothing
    end

    function lander_terminated!(dx, x, p, t)
        dx .= zero(x[1])
        return nothing
    end

    return lander_thrust_on!, lander_thrust_off!, lander_terminated!
end

## Constraints
angle_sat = deg2rad(45)
# u_constraints = HPolyhedron([α ≥ -angle_sat, α ≤ angle_sat], controls)
above_ground = HalfSpace(ry > 0, vars)
below_ground = HalfSpace(ry ≤ 0, vars)
burnt_out = HalfSpace(m_prop ≤ 0, vars)


## Modes
controls = ComponentArray(kp=0.01, kd=0.05)
f_on!, f_off!, f_terminated! = make_ode_functions(controls)

# Mode 1: Thrust on
thrust_on = @system(x'=f_on!(x), x∈above_ground, dims=9)

# Mode 2: Thrust off
thrust_off = @system(x'=f_off!(x), x∈above_ground, dims=9)

# Mode 3: Terminated (hit ground)
terminated = @system(x'=f_terminated!(x), x∈below_ground, dims=9)

modes = [thrust_on, thrust_off, terminated]


## Mode Transitions
# State automoaton
automaton = LightAutomaton(3)

hits_ground = @map(x -> x, dim:9, x∈below_ground) #ConstrainedIdentityMap(length(vars), below_ground)
burns_out = @map(x -> x, dim:9, x∈burnt_out) #ConstrainedIdentityMap(length(vars), burnt_out)
resetmaps = typeof(hits_ground)[]

# Transition 1: thrust on -> thrust off
add_transition!(automaton, 1, 2, 1)
push!(resetmaps, burns_out)

# Transition 2: thrust on -> terminated
add_transition!(automaton, 1, 3, 2)
push!(resetmaps, hits_ground)

# Transition 3: thrust off -> terminated
add_transition!(automaton, 2, 3, 3)
push!(resetmaps, hits_ground)


## System Definition
H = HybridSystem(; automaton, modes, resetmaps)

## Initial Conditions
x0 = Any[
    0.0,        # rx
    120..150,   # ry
    0 ± 20,     # vx
    -45 ± 5,    # vy
    20.0,       # m_prop
    70 ± 3,     # Isp
    60 ± 5,     # m_dry
    1400..1450, # T
    0.0,        # t
] |> to_set |> concretize

ic = [(1, x0)]

prob = InitialValueProblem(H, ic)


##
boxdirs = BoxDirections(9)
alg = TMJets(; orderT=9, orderQ=1, disjointness=BoxEnclosure())
sol = solve(prob, alg;
    T = 10.0,
    max_jumps = 2,
    intersect_source_invariant = false,
    intersection_method = TemplateHullIntersection(boxdirs),
    clustering_method = LazyClustering(1),
    disjointness_method = BoxEnclosure(),
)