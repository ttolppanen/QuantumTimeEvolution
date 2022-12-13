# using ITensors
# using LinearAlgebra
# using Distributed

export solvetrajectories

#f : function; a function that takes no arguments and retursn the time-evolution
#traj : number of trajectories

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