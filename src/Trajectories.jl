# using ITensors
# using LinearAlgebra
# using Distributed

export solvetrajectories
export solvetrajectories_channel
export traj_channels

#f : function; a function that takes no arguments and returns the time-evolution
#traj : number of trajectories

function traj_channels(out, traj::Integer, args...)
    n_threads = Threads.nthreads()
    pa_out = Channel{typeof(out)}(traj)
    pa_args = Channel{typeof(args)}(n_threads)
    for _ in 1:traj
        put!(pa_out, deepcopy(out))
    end
    for _ in 1:n_threads
        put!(pa_args, deepcopy(args))
    end
    return pa_args, pa_out
end

function solvetrajectories_channel(f::Function, traj::Integer, pa_args, pa_out)
    @sync for _ in 1:traj
        Threads.@spawn begin
            out = take!(pa_out)
            args = take!(pa_args)
            f(out, args...)
            put!(pa_out, out)
            put!(pa_args, args)
        end
    end
end

function solvetrajectories(f::Function, traj::Integer; paral::Symbol = :none)
    out = Array{Any}(undef, traj)
    if paral == :threads
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        Threads.@threads for i in 1:traj
            out[i] = f()
        end
        BLAS.set_num_threads(blas_threads)
    elseif paral == :distributed
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        p_f(x) = f()
        return pmap(p_f, 1:traj)
        BLAS.set_num_threads(blas_threads)
    elseif paral == :none
        for i in 1:traj
            out[i] = f()
        end
    else
        throw(ArgumentError("possible keywords for paral {:none, :threads, :distributed}"))
    end
    return out
end