## Types
@concrete struct Saturated <: Function
    f
    lb
    ub
end
(f::Saturated)(args...; kwargs...) = clamp(f.f(args...; kwargs...), f.lb, f.ub)

# Proportional-derivative controller
function PD_controller(;kp, kd)
    gains = ComponentArray(;kp, kd)
    f = (x, p, t) -> -p.kp*x.θ - p.kd*x.ω
    return gains, f
end

# Single-layer dense neural controller
function dense_neural_controller(layer_size, act_fun=identity, out_fun=only)
    layers = ComponentArray(;
        L1 = dense_layer(2, layer_size),
        L2 = dense_layer(layer_size, 1),
    )

    f = function (x, p, t)
        @unpack L1, L2 = p
        out = act_fun.(L2.W * act_fun.(L1.W * x + L1.b) + L2.b)
        return out_fun(out)
    end

    return layers, f
end