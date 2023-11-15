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
    if isa(find_subspace, Nothing) # No subspace
        for (j, obs) in pairs(observables)
            out[j, 1] = obs(state)
        end
        for i in 2:steps
            evolve_time_step!(state)
            if apply_effect_first effect!(state) end
            if save_only_last
                if i == steps
                    for (j, obs) in pairs(observables)
                        out[j, 1] = obs(state)
                    end
                end
            else
                for (j, obs) in pairs(observables)
                    out[j, i] = obs(state)
                end
            end
            if apply_effect_last effect!(state) end
        end
        return out
    else # In subspace
        subspace_indices = find_subspace(state)
        for (j, obs) in pairs(observables)
            out[j, 1] = obs(state, subspace_indices)
        end
        for i in 2:steps
            subspace_indices = find_subspace(state)
            evolve_time_step!(state, subspace_indices)
            if apply_effect_first effect!(state, subspace_indices) end
            if save_only_last
                if i == steps
                    for (j, obs) in pairs(observables)
                        out[j, 1] = obs(state, subspace_indices)
                    end
                end
            else
                for (j, obs) in pairs(observables)
                    out[j, i] = obs(state, subspace_indices)
                end
            end
            if apply_effect_last effect!(state, subspace_indices) end
        end
        return out
    end
end