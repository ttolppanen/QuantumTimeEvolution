# using SparseArrays
# using QuantumOperators

export exactevolve

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real, observables...; 
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)

    steps = length(0:dt:t)
    initialize(state0) = return (copy(state0), )
    time_step_funcs = [] # functions to run in a single timestep

    function take_time_step(state)
        state .= U * state
        normalize!(state)
        return (state, )
    end
    push!(time_step_funcs, take_time_step)

    if !isa(effect!, Nothing)
        function do_effect(state)
            effect!(state)
            return (state, )
        end
        if save_before_effect
            push!(time_step_funcs, :calc_obs)
            push!(time_step_funcs, do_effect)
        else
            push!(time_step_funcs, do_effect)
            push!(time_step_funcs, :calc_obs)
        end
    else # no effect
        push!(time_step_funcs, :calc_obs)
    end
        
    return timeevolve!(state0, initialize, time_step_funcs, steps, observables...; save_only_last)
end
