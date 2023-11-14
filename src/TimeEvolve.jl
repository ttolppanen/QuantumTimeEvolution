# using SparseArrays
# using QuantumOperators

# state0 : initial state;
# evolve_time_step! : a function that evolves the current state for one timestep;
# steps : number of steps in the time evolution, where the 0 time is counted as a step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect
# find_subspace : a function that finds the indeces of the current subspace; This should be a function that takes the state as an argument, and uses it to find the
#                                                                            indeces of the current subspace as an array e.g. find_subspace(state) -> [1,2,4,5,10,12,...]

function timeevolve!(state0, evolve_time_step!::Function, steps::Int, observables...;
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, find_subspace = nothing)
    
    apply_effect_first = !isa(effect!, Nothing) && !save_before_effect
    apply_effect_last = !isa(effect!, Nothing) && save_before_effect
    out = save_only_last ? zeros(length(observables), 1) : zeros(length(observables), steps)
    state = deepcopy(state0)
    out[:, 1] .= [obs(state) for obs in observables]
    if isa(find_subspace, Nothing) # No subspace
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
    else # In subspace
        for i in 2:steps
            subspace_indeces = find_subspace(state)
            evolve_time_step!(state, subspace_indeces)
            if apply_effect_first effect!(state, subspace_indeces) end
            if save_only_last
                if i == steps
                    out[:, 1] .= [obs(state, subspace_indeces) for obs in observables]
                end
            else
                out[:, i] .= [obs(state, subspace_indeces) for obs in observables]
            end
            if apply_effect_last effect!(state, subspace_indeces) end
        end
        return out
    end
end