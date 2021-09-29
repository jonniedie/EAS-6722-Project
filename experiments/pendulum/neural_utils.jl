# Make the parameter structure of a dense layer
dense_layer(n_in, n_out) = ComponentArray(W=Flux.glorot_uniform(n_out, n_in), b=zeros(Float32, n_out))

# Returns a function that runs the simulation for a given control parameters
function make_run_func(prob; kwargs...)
    return function (control)
        p = deepcopy(prob.p)
        p = (; p..., control)
        u0 = eltype(control).(prob.u0)
        new_prob = remake(prob; p=p, u0=u0)
        return solve(new_prob, Tsit5(); kwargs...)
    end
end

function make_objective(prob, out_fun; kwargs...)
    run_func = make_run_func(prob; kwargs...)
    return function (x)
        sol = run_func(x)
        return out_fun(sol), sol
    end
end

function make_callback(; plot_every=1, doplot=false)
    # plt = plot()
    all_loss = Float64[]
    i = 0
    return function (θ, loss, pred; doplot=doplot)
        display(loss)
        if doplot
            if i % plot_every == 0
                push!(all_loss, loss)
                display(plot((0:length(all_loss)-1)*plot_every, all_loss))
            end
        end
        i += 1
        return false
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