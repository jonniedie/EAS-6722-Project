
include("initialize.jl")
using Plots

##
# Make the ODEProblem
PD_prob = let 
	## Inputs
    # Model constants
    g = 9.80665

    # Control Signal
    saturation_lims = (-50, 50)
    control, u = PD_controller(
        kp = 1000f0, # Proportional gain
        kd = 300f0, # Derivative gain
    )
    # control, u = dense_neural_controller(50, Flux.relu, only)

	# Simulation parameters
	tf = 5.0 # Stop time (s)
	
	# Model parameters
	m = 100 # Rod mass (kg)
	L = 0.5 # Rod length (m)
	c = 0.1 # Damping coefficient
	
	# Initial conditions
	θ₀ = deg2rad(-20) # Initial angle
	ω₀ = deg2rad(45) # Initial angular velocity

    p = (; g, m, L, c, I=(1/3)*m*L^2, control)
	ic = ComponentArray{Float32}(θ=θ₀, ω=ω₀)

    u_sat = Saturated(u, saturation_lims...)
	ODEProblem(pendulum_with_control(u_sat), ic, Float32.((0.0, tf)), p)
end

neural_prob = let prob=deepcopy(PD_prob)
    params, u = dense_neural_controller(50, Flux.relu, only)
    f = prob.f.f
    @set! f.u.f = u
    p = ComponentArray(; NamedTuple(prob.p)..., control=params)
    ODEProblem(f, prob.u0, prob.tspan, p; prob.kwargs...)
end

PD_run_func = make_run_func(PD_prob; saveat=0.1)
neural_run_func = make_run_func(neural_prob; saveat=0.1)


## Solve the plain ODEProblem
PD_sol = solve(PD_prob)
neural_sol = solve(neural_prob)


## Plot stuff
plot(plot(PD_sol, title="PD"), plot(neural_sol, title="Neural"), layout=(2,1), size=(700,500))


## Buiild optimization problems
evaluate_sol = sol -> sum(abs2, Array(sol))
PD_obj = make_objective(PD_prob, evaluate_sol; sensealg=InterpolatingAdjoint())


## Solve optimization problem'
x0 = ComponentArray{Float64}(kp=1000, kd=100)
res1 = DiffEqFlux.sciml_train(PD_obj, x0, ADAM(0.05), maxiters=500)
# cb(res1.minimizer, loss_n_ode(res1.minimizer)...; doplot=true)

res2 = DiffEqFlux.sciml_train(PD_obj, res1.minimizer, Newton())
# cb(res2.minimizer, loss_n_ode(res2.minimizer)...; doplot=true)