# using ITensors

export solvetrajectories

#f : function; a function that takes no arguments and retursn the time-evolution
#traj : number of trajectories

function solvetrajectories(f::Function, traj::Integer; use_threads::Bool = true)
    out = [f()]
    for _ in 2:traj push!(out, deepcopy(out[1])) end
    if use_threads
        Threads.@threads for i in 2:traj
            f(out[i])
        end
    else
        for i in 2:traj
            f(out[i])
        end
    end
    return out
end