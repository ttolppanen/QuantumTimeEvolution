using Distributed
if(length(procs()) == 1)
    addprocs(0)
    println("Added a process")
end
@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere begin
    using QuantumStates
    using QuantumOperators
    using QuantumTimeEvolution
    using Plots
    using BenchmarkTools
    using LinearAlgebra

    function mipt(d, L, dt, t, traj, prob)
        k = 3
        state0 = zeroone(d, L)
        H = bosehubbard(d, L)
        op_to_msr = nop(d)
        msrop = measurementoperators(op_to_msr, L)
        res = []
        for p in prob
            meffect!(state) = measuresitesrandomly!(state, msrop, p)
            r_exact() = krylovevolve(state0, H, dt, t, k; effect! = meffect!, savelast = true)
            r_traj = solvetrajectories(r_exact, traj; paral = :distributed)
            r_mean = trajmean(r_traj, s->entanglement(d, L, s, Int(floor(L/2))))
            push!(res, r_mean[1])
        end
        return prob, res
    end
end

# PA_krylov(length(state0), k)
BLAS.set_num_threads(1)
@time res = mipt(2, 4, 0.1, 10.0, 1000, 0.01:0.05:0.3);
rmprocs(workers())
plot(res)
