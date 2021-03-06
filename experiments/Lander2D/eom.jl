const g = 9.80665

# ODE Functions
@taylorize function lander_thrust_on!(dx, x, p, t)
    # Unpack variables
    rx, ry, vx, vy, m_prop, Isp, m_dry, T, θ = x

    # Intermediate parameters
    m = m_dry + m_prop
    Fx, Fy = T .* (sin(θ), cos(θ))

    # Needed because numbers can't be converted into Taylor?
    _0 = zero(x[1])

    # State derivatives
    dx[1] = vx              # rx
    dx[2] = vy              # ry
    dx[3] = Fx/m            # vx
    dx[4] = -g + Fy/m       # vy
    dx[5] = -T/(Isp * g)    # m_prop
    dx[6] = _0              # Isp
    dx[7] = _0              # m_dry
    dx[8] = _0              # T
    dx[9] = _0              # θ
    return nothing
end

@taylorize function lander_thrust_off!(dx, x, p, t)
    # Unpack variables
    rx, ry, vx, vy, m_prop, Isp, m_dry, T = x

    # Needed because numbers can't be converted into Taylor?
    _1 = one(x[1])
    _0 = zero(x[1])

    # State derivatives
    dx[1] = vx      # rx
    dx[2] = vy      # ry
    dx[3] = _0      # vx
    dx[4] = -_1*g   # vy
    dx[5] = _0      # m_prop
    dx[6] = _0      # Isp
    dx[7] = _0      # m_dry
    dx[8] = _0      # T
    dx[9] = _0      # θ
    return nothing
end

@taylorize function lander_terminated!(dx, x, p, t)
    _0 = zero(x[1])
    dx[1] = _0  # rx
    dx[2] = _0  # ry
    dx[3] = _0  # vx
    dx[4] = _0  # vy
    dx[5] = _0  # m_prop
    dx[6] = _0  # Isp
    dx[7] = _0  # m_dry
    dx[8] = _0  # T
    dx[9] = _0  # θ
    return nothing
end