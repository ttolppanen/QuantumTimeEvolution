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

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real, observables...; effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)
    steps = length(0:dt:t)
    evolve_time_step!(state) = exact_time_step!(state, U)
    state = copy(state0)
    return timeevolve!(state, evolve_time_step!, steps, observables...; effect!, save_before_effect, save_only_last)
end

function exact_time_step!(state, U)
    state .= U * state
    normalize!(state)
end