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


function timeevolve!(initial_args, time_step_funcs, steps::Int, observables...; kwargs...)
    out = kwargs[:save_only_last] ? zeros(length(observables), 1) : zeros(length(observables), steps) # matrix with dimensions of N(observables) x N(steps)
    return timeevolve!(initial_args, time_step_funcs, steps, out, observables...; kwargs...)
end

function timeevolve!(initial_args, time_step_funcs, steps::Int, out::Matrix{Float64}, observables...; save_only_last::Bool = false)
    up_out = generate_calc_obs_func(out, observables)
    # typeof IMPORTANT
    args::typeof(initial_args) = initial_args # initial arguments that are passed on to time evolution
    is_args_a_tuple = isa(args, Tuple)
    is_args_a_tuple ? up_out(1, args...) : up_out(1, args) # initial_args observables values
    try
        for i in 2:steps
            for f in time_step_funcs
                if f != :calc_obs
                    if is_args_a_tuple
                        args = f(args...)
                    else
                        args = f(args)
                    end
                else
                    if save_only_last
                        if i == steps
                            is_args_a_tuple ? up_out(1, args...) : up_out(1, args)
                        end
                    else
                        is_args_a_tuple ? up_out(i, args...) : up_out(i, args)
                    end
                end
            end
        end
    catch e
        if isa(args, Nothing)
            println("
            A function in the time evolution is returning nothing. 
            Check that your effect! and your timestep are returning proper values. 
            For the normal case, all functions should return the state. 
            In the subspace case, all functions should return the state and the current subspace id -> (state, id) 
            In your own custom time evolution they should all return what is changing during the time evolution.\n")
        end
        throw(e)
    end
    return out
end

function generate_calc_obs_func(out, observables)
    out::typeof(out) = out
    function up_out(i, args...)
        for (j, obs) in pairs(observables)
            out[j, i] = obs(args...)
        end
    end
    return up_out
end
