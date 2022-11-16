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

function measured_bh(mps0::MPS, dt::Real, t::Real; traj::Integer = 1, effect!::Function, kwargs...) #keyword arguments for bosehubbard, mpsevolve_bosehubbard
    f() = mpsevolve_bosehubbard(mps0::MPS, dt::Real, t::Real; effect!, kwargs...)
    return solvetrajectories(f, traj)
end