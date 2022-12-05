# using ITensors

export solvetrajectories

#f : function; a function that takes no arguments and retursn the time-evolution
#traj : number of trajectories

function solvetrajectories(f::Function, traj::Integer; use_threads::Bool = true)
    out = Array{Any}(undef, traj)
    if use_threads
        Threads.@threads for i in 1:traj
            out[i] = f()
        end
    else
        for i in 1:traj
            out[i] = f()
        end
    end
    return out
end