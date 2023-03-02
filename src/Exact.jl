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
    apply_effect_first = !isa(effect!, Nothing) && !save_before_effect
    apply_effect_last = !isa(effect!, Nothing) && save_before_effect
    state = copy(state0)
    out = save_only_last ? zeros(length(observables), 1) : zeros(length(observables), length(0:dt:t))
    out[:, 1] .= [obs(state) for obs in observables]
    steps = length(0:dt:t)
    for i in 2:steps
        state .= U * state
        normalize!(state)
        if apply_effect_first effect!(state) end
        if save_only_last
            if i == steps
                out[:, 1] .= [obs(state) for obs in observables]
            end
        else
            out[:, i] .= [obs(state) for obs in observables]
        end
        if apply_effect_last effect!(state) end
    end
    return out
end