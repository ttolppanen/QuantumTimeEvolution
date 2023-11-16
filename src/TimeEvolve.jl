# using SparseArrays
# using QuantumOperators

# state0 : initial state;
# initial_args : a function that calculates the initial arguments; A function that takes the initial state
#                                                                  as an argument, and returns the arguments the other functions
#                                                                  that do the time evolution need.
# time_step_funcs : list of functions that the timestep consists of; These functions take in the arguments that initial_args returns, they can change these,
#                                                                    and they should return these so that the next function has the updated arguments, if needed.
#                                                                    These functions build the time evolution, so the list is something like
#                                                                    time_step_funcs = [take_time_step, do_effects, :calc_obs]. The :calc_obs is reverved for
#                                                                    calculating the observables.
# steps : number of steps in the time evolution, where the 0 time is counted as a step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.                                                                   indeces of the current subspace as an array e.g. find_subspace(state) -> [1,2,4,5,10,12,...]


function timeevolve!(state0, initial_args::Function, time_step_funcs, steps::Int, observables...; save_only_last::Bool = false)
    out = save_only_last ? zeros(length(observables), 1) : zeros(length(observables), steps)
    up_out = generate_calc_obs_func(out, observables)
    args = initial_args(state0) # initial arguments that are passed on to other functions
    up_out(1, args...) # state0 observables values
    for i in 2:steps
        for f in time_step_funcs
            if f == :calc_obs
                if save_only_last
                    if i == steps 
                        up_out(1, args...) 
                    end
                else
                    up_out(i, args...)
                end
            else
                args = f(args...)
            end
        end
    end
    return out
end

function generate_calc_obs_func(out, observables)
    function up_out(i, args...)
        for (j, obs) in pairs(observables)
            out[j, i] = obs(args...)
        end
    end
    return up_out
end