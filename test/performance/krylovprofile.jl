using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using Plots

function mipt(d, L, dt, t, traj, prob)
    state0 = zeroone(d, L)
    H = bosehubbard(d, L)
    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, L)
    res = []
    for p in prob
        meffect!(state) = measuresitesrandomly!(state, msrop, p)
        r_exact(args...) = krylovevolve(state0, H, dt, t, 3, args...; effect! = meffect!, savelast = true)
        r_traj = solvetrajectories(r_exact, traj; use_threads = true)
        r_mean = trajmean(r_traj, s->entanglement(d, L, s, Int(floor(L/2))))
        push!(res, r_mean[1])
    end
    return prob, res
end

@time res = mipt(2, 4, 0.1, 10.0, 1000, 0.01:0.01:0.3);