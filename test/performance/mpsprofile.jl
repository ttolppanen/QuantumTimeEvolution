using Distributed
if(length(procs()) == 1)
    addprocs(7)
    @show workers()
    println("Added a process")
end
# @everywhere begin
#     using Pkg
#     Pkg.activate(@__DIR__)
#     Pkg.instantiate()
#     Pkg.precompile()
# end

@everywhere begin
    using QuantumStates
    using QuantumOperators
    using QuantumTimeEvolution
    using ITensors
    using LinearAlgebra

    function mipt(d, L, dt, t, traj, prob)
        mps0 = zeroonemps(d, L)
        gates = bosehubbardgates(siteinds(mps0), dt)
        op_to_msr = nop(d)
        msrop = measurementoperators(op_to_msr, siteinds(mps0))
        res = []
        for p in prob
            meffect!(state) = measuresitesrandomly!(state, msrop, p)
            r_exact() = mpsevolve(mps0, gates, dt, t; savelast = true)
            r_traj = solvetrajectories(r_exact, traj; paral = :distributed)
            r_mean = trajmean(r_traj, s->entanglement(s, Int(floor(L/2))))
            push!(res, r_mean[1])
        end
        return prob, res
    end
end

@time mipt(2, 4, 0.6, 10.0, 100, 0.1:0.1:0.1);
# rmprocs(workers())
