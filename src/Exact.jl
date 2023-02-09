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

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real, observables; effect! = nothing, save_before_effect::Bool = false)
    apply_effect_first = !isa(effect!, Nothing) && !save_before_effect
    apply_effect_last = !isa(effect!, Nothing) && save_before_effect
    state = copy(state0)
    out = zeros(length(observables), length(0:dt:t))
    out[:, 1] .= [obs(state) for obs in observables]
    for i in 2:length(0:dt:t)
        state .= U * state
        if apply_effect_first effect!(state) end
        out[:, i] .= [obs(state) for obs in observables]
        if apply_effect_last effect!(state) end
    end
    return out
end