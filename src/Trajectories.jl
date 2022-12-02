export solvetrajectories

#f : function; a function that takes no arguments and retursn the time-evolution
#traj : number of trajectories

function solvetrajectories(f::Function, traj::Integer)
    out = [[] for _ in 1:Threads.nthreads()]
    Threads.@threads for _ in 1:traj
        push!(out[Threads.threadid()], f())
    end
    return reduce(vcat, out)
end