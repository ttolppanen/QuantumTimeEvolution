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

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real, observables...; kwargs...)
    work_vector = copy(state0)
    exactevolve(state0, work_vector, U, dt, t, observables...; kwargs...)
end
function exactevolve(state0::AbstractVector{<:Number}, work_vector::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real, observables...; 
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)

    steps = length(0:dt:t)
    work_vector .= state0
    initial_args = work_vector
    time_step_funcs = [] # functions to run in a single timestep

    take_time_step! = take_exact_time_step_function(U)
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

function take_exact_time_step_function(U)
    function take_time_step!(state)
        state .= U * state
        normalize!(state)
        return state
    end
    return take_time_step!
end