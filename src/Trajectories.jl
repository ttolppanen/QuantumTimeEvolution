# using ITensors
# using LinearAlgebra
# using Distributed
# using ChunkSplitters

export solvetrajectories
export pre_alloc_threads

#f : function; a function that takes no arguments and returns the time-evolution
#traj : number of trajectories

function pre_alloc_threads(out, traj::Integer, args...)
    n_threads = Threads.nthreads()
    pa_out = [copy(out) for _ in 1:traj]
    pa_args = Channel{typeof(args)}(n_threads)
    for _ in 1:n_threads
        put!(pa_args, deepcopy(args))
    end
    return pa_args, pa_out
end

function solvetrajectories(f::Function, traj::Integer, pa_args, pa_out; paral::Symbol = :none)
    if paral == :none
        args = take!(pa_args)
        for i in 1:traj
            f(pa_out[i], args...)
        end
        put!(pa_args, args)
    elseif paral == :threads
        solve_traj_paral_pre_alloc(f, traj, pa_args, pa_out)
    else
        throw(ArgumentError("possible keywords for paral {:none, :threads"))
    end
end

function solve_traj_paral_pre_alloc(f, traj, pa_args, pa_out)
    @sync for (traj_chunk, _) in chunks(1:traj, Threads.nthreads())
        Threads.@spawn begin
            args = take!(pa_args)
            for i in traj_chunk
                f(pa_out[i], args...)
            end
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