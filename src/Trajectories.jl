# using ITensors
# using LinearAlgebra

export solvetrajectories

#f : function; a function that takes no arguments and retursn the time-evolution
#traj : number of trajectories

function solvetrajectories(f::Function, traj::Integer; use_threads::Bool = true)
    out = Array{Any}(undef, traj)
    if use_threads
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        Threads.@threads for i in 1:traj
            out[i] = f()
        end
        BLAS.set_num_threads(blas_threads)
    else
        for i in 1:traj
            out[i] = f()
        end
    end
    return out
end