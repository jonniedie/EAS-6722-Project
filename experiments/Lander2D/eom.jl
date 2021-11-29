const g = 9.80665

# ODE function for 2D lander problem
function lander_2D!(D, vars, (params, θ), t)
    @unpack r, v, m_prop = vars
    @unpack Isp, m_dry, T = params

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

function lander_2D(vars, (params, θ), t)
    @unpack r, v, m_prop = vars
    @unpack Isp, m_dry, T = params

    m = m_dry + m_prop
    Fx = T*sin(θ)
    Fy = T*cos(θ)

    return [
        v.x
        v.y
        Fx/m
        -g + Fy/m
        -T / (Isp * g)
    ]
end