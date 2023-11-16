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

function exactevolve(state0::AbstractVector{<:Number}, U, find_subspace::Function, dt::Real, t::Real, observables...; 
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)

    steps = length(0:dt:t)
    function initialize(state0)
        id, indices = find_subspace(state0)
        return Vector(copy(state0)), id, indices # the arguments are for subspace_id, and subspace_range
    end
    time_step_funcs = [] # functions to run in a single timestep

    function current_subspace(state, id, indices)
        new_id, new_indices = find_subspace(state)
        return state, new_id, new_indices
    end
    push!(time_step_funcs, current_subspace)

    function take_time_step(state, id, subspace_indices)
        @views state[subspace_indices] .= U[id] * state[subspace_indices]
        normalize!(state)
        return state, id, subspace_indices
    end
    push!(time_step_funcs, take_time_step)

    if !isa(effect!, Nothing)
        if save_before_effect
            push!(time_step_funcs, :calc_obs)
            push!(time_step_funcs, effect!)
        else
            push!(time_step_funcs, effect!)
            push!(time_step_funcs, :calc_obs)
        end
    else # no effect
        push!(time_step_funcs, :calc_obs)
    end
        
    return timeevolve!(state0, initialize, time_step_funcs, steps, observables...; save_only_last)
end

function exact_time_step_subspace!(state, U, id, subspace_indices)
    @views state[subspace_indices] .= U[id] * state[subspace_indices]
    normalize!(state)
    return state, id, subspace_indices
end
