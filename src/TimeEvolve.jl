# using SparseArrays
# using QuantumOperators

# state0 : initial state;
# evolve_time_step! : a function that evolves the current state for one timestep;
# steps : number of steps in the time evolution, where the 0 time is counted as a step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

function timeevolve!(state0, evolve_time_step!::Function, steps::Int, observables...; effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)
    apply_effect_first = !isa(effect!, Nothing) && !save_before_effect
    apply_effect_last = !isa(effect!, Nothing) && save_before_effect
    out = save_only_last ? zeros(length(observables), 1) : zeros(length(observables), steps)
    state = deepcopy(state0)
    out[:, 1] .= [obs(state) for obs in observables]
    for i in 2:steps
        evolve_time_step!(state)
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