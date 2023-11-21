# using SparseArrays
# using QuantumOperators

# state0 : initial state;
# initial_args : the initial running argument of the time evolution; Arguments that change during the time evolution. For the most basic case, initial_args could be
#                                                                    a copy of the state
# time_step_funcs : list of functions that the timestep consists of; These functions take in the arguments that initial_args returns, they can change these,
#                                                                    and they should return these so that the next function has the updated arguments, if needed.
#                                                                    These functions build the time evolution, so the list is something like
#                                                                    time_step_funcs = [take_time_step, do_effects, :calc_obs]. The :calc_obs is reverved for
#                                                                    calculating the observables.
# steps : number of steps in the time evolution, where the 0 time is counted as a step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.                                                                   indeces of the current subspace as an array e.g. find_subspace(state) -> [1,2,4,5,10,12,...]


function timeevolve!(initial_args, time_step_funcs, steps::Int, observables...; save_only_last::Bool = false)
    out = save_only_last ? zeros(length(observables), 1) : zeros(length(observables), steps) # matrix with dimensions of N(observables) x N(steps)
    up_out = generate_calc_obs_func(out, observables)
    args = initial_args # initial arguments that are passed on to time evolution
    up_out(1, args...) # initial_args observables values
    for i in 2:steps
        for f in time_step_funcs
            if f != :calc_obs
                args = f(args...)
            else
                if save_only_last
                    if i == steps 
                        up_out(1, args...) 
                    end
                else
                    up_out(i, args...)
                end
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
