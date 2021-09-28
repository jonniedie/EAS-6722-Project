## Functions
# Pendulum equations of motion
function pendulum!(dx, x, p, t; M=0)
	θ, ω = x
	g, m, L, c, I = p
	
	dx[1] = ω
	dx[2] = (M + m*g*sin(θ)*L/2 - c*ω) / I
	return nothing
end
function pendulum!(D, vars::ComponentArray, params, t; M=0)
	@unpack θ, ω = vars
	@unpack g, m, L, c, I = params
	
	D.θ = ω
	D.ω = (M + m*g*sin(θ)*L/2 - c*ω) / I
	return nothing
end

pendulum_with_control(u) = (dx,x,p,t) -> pendulum!(dx,x,p,t; M=u(x,p.control,t))
