# using SparseArrays
# using QuantumOperators

# export exactevolve exported already in Exact.jl

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

function exactevolve(state0, initial_id::Integer, U, dt::Real, t::Real, observables...
    ; effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)

    steps = length(0:dt:t)
    initial_args = (Vector.(deepcopy(state0)), initial_id) # id identifies the current subspace
    time_step_funcs = [] # functions to run in a single timestep

    take_time_step! = take_exact_time_step_subspace_function!(U)
    push!(time_step_funcs, take_time_step!)
    push!(time_step_funcs, :calc_obs) # calculating observables is told with a keyword :calc_obs

    if !isa(effect!, Nothing)
        if save_before_effect
            push!(time_step_funcs, effect!)
        else
            insert!(time_step_funcs, 2, effect!)
        end
    end
        
    return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
end

function take_exact_time_step_subspace_function!(U)
    function take_time_step!(state, id)
        state[id] .= U[id] * state[id]
        normalize!(state[id])
        return state, id
    end
    return take_time_step!
end
