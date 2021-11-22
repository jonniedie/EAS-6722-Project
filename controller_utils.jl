# Make a dense neural network layer
make_layer(num_in, num_out, activation=Id()) = Layer(rand(num_out, num_in), rand(num_out), activation)

collect_params(layer::Layer) = vcat(vec(layer.weights), layer.bias)
collect_params(network::Network) = mapreduce(collect_params, vcat, network.layers)

function set_params!(layer::Layer, values)
    n_weights = length(layer.weights)
    layer.weights[:] .= @view values[1:n_weights]
    layer.bias .= @view values[n_weights+1:end]
    return layer
end
function set_params!(network::Network, values)
    i_begin = 1
    for layer in network.layers
        i_end = i_begin + length(layer.weights) + length(layer.bias) - 1
        set_params!(layer, @view(values[i_begin:i_end]))
        i_begin = i_end+1
    end
    return network
end

Base.@kwdef mutable struct PD{T}
    kp::T = 0.0
    kd::T = 0.0
end

(controller::PD)(x) = controller.kp*x[1] + controller.kd*x[2]

collect_params(controller::PD) = [controller.kp, controller.kd]
function set_params!(controller::PD, values)
    controller.kp = values[1]
    controller.kd = values[2]
    return controller
end