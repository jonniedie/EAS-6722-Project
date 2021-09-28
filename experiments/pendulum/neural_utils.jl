# Make the parameter structure of a dense layer
dense_layer(n_in, n_out) = ComponentArray(W=Flux.glorot_uniform(n_out, n_in), b=zeros(Float32, n_out))

#
function make_run_func(prob; kwargs...)
    return function (control)
        p = copy(prob.p)
        p.control .= control
        new_prob = remake(prob; p=p)
        solve(new_prob, Tsit5(); kwargs...)
    end
end

# # Don't think we'll actually use this
# function run_neural(x, layers, act_fun, out_fun=identity)
#     out =  reduce(valkeys(layers); init=x) do y, key
#         @unpack W, b = layers[key]
#         return act_fun.(W*y .+ b)
#     end
#     return out_fun(out)
# end