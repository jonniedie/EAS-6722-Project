using ConcreteStructs


# Controller definitions
Base.@kwdef @concrete struct WithController <: Function
    ode_fun<:Function
    controller<:Function = (x,p,t) -> 0
    params = nothing
end

(sys::WithController)(dx, x, p, t) = sys.ode_fun(dx, x, (p, sys.controller(x, sys.params, t)), t)
(sys::WithController)(x, p, t) = sys.ode_fun(x, (p, sys.controller(x, sys.params, t)), t)


# Neural network stuff
dense_layer(n_in, n_out, activation=identity; eltype=Float32) = (; params=ComponentArray{eltype}(W=glorot_uniform(n_out, n_in), b=zeros(n_out)), activation)

function chain(; layers...)
    layers = NamedTuple(layers)
    params = ComponentArray(; map(first, layers)...)
    activations = map(last, layers)
    k = valkeys(params)
    f = (x,p,t) -> reduce((x,(f,p))->f.(p.W*x .+ p.b), zip(activations, view.(Ref(p), k)); init=x)
    return (; params, f)
end
chain(layer1, layers...) = chain(; NamedTuple{ntuple(i->Symbol(:layer, i), length(layers)+1)}((layer1, layers...))...)
