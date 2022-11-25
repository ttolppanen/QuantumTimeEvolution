export solvetrajectories
export measured_bh

#f : function; a function that takes no arguments and retursn the time-evolution
#traj : number of trajectories

function solvetrajectories(f::Function, traj::Integer)
    out = []
    for _ in 1:traj
        push!(out, f())
    end
    return out
end