using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using ITensors

function mipt(d, L, dt, t, traj, prob)
    mps0 = zeroonemps(d, L)
    gates = bosehubbardgates(siteinds(mps0), dt)
    @show siteinds(mps0)
    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, siteinds(mps0))
    res = []
    for p in prob
        meffect!(state) = measuresitesrandomly!(state, msrop, p)
        r_exact() = mpsevolve(mps0, gates, dt, t; effect! = meffect!, savelast = true)
        r_traj = solvetrajectories(r_exact, traj)
        r_mean = trajmean(r_traj, s->entanglement(s, Int(floor(L/2))))
        push!(res, r_mean[1])
    end
    return prob, res
end

@profview @time mipt(2, 4, 0.1, 10.0, 5, 0.01:0.01:0.3);