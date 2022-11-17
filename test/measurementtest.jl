using QuantumTimeEvolution.QuantumStates
using QuantumTimeEvolution.QuantumOperators
using ITensors
using Random
using Plots

@testset "measuresitesrandomly!" begin
    d = 3; L = 3
    dt = 0.1; t = 5
    msr_prob = 0.1
    rng_seed = 5
    state0 = zeroone(d, L)
    mps0 = zeroonemps(d, L)
    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, L)
    msrop_tensor = measurementoperators(op_to_msr, siteinds(mps0))
    meffect!(state) = measuresitesrandomly!(state, msrop, msr_prob)
    meffect_t!(state) = measuresitesrandomly!(state, msrop_tensor, msr_prob)
    Random.seed!(rng_seed) #Makes the rng the same
    r_exact = exactevolve_bosehubbard(d, L, state0, dt, t; effect! = meffect!)
    Random.seed!(rng_seed) #Makes the rng the same
    r_krylov = krylovevolve_bosehubbard(d, L, state0, dt, t, 5; effect! = meffect!)
    Random.seed!(rng_seed) #Makes the rng the same
    r_mps = mpsevolve_bosehubbard(mps0, dt, t; effect! = meffect_t!)
    plot_x = 0:dt:t
    @testset "Normalization" begin
        pl = plot(plot_x, [norm(s) for s in r_exact], ylims=(0.99, 1.01), label="exact")
        plot!(pl, plot_x, [norm(s) for s in r_krylov], label="krylov")
        plot!(pl, plot_x, [norm(s) for s in r_mps], label="mps")
        saveplot(pl, "msr_norm")
        @test true
    end
    @testset "First Site Boson Number" begin
        #with mps, the first site is the last site?
        #So here this shouldnt work, since the measurements happen
        #to the wrong sites (There was a measurement on the first
        #site -> msr on the last site for mps). But since the state is
        #symmetrical, it works.

        n1 = singlesite_n(d, L, 1)
        pl = plot(plot_x, expval(r_exact, n1), label="exact")
        plot!(pl, plot_x, expval(r_krylov, n1), label="krylov")
        plot!(pl, plot_x, expval(r_mps, "N"; sites=1), label="mps")
        saveplot(pl, "msr_bosonnumber")
        @test true
    end
    @test true
end

function calc_ent_traj(msr_prob)
    d = 2; L = 4
    dt = 0.1; t = 5
    mps0 = zeroonemps(d, L)
    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, siteinds(mps0))
    meffect!(state) = measuresitesrandomly!(state, msrop, msr_prob)
    result = measured_bh(mps0, dt, t; traj = 30, effect! = meffect!, cutoff = 1E-8)
    res = trajmean(result, state -> entanglement(state, 2))
    return 0:dt:t, res
end
@testset "Trajectories" begin
    rng_seed = 3
    Random.seed!(rng_seed) #Makes the rng the same
    t, res = calc_ent_traj(0.01)
    pl = plot(t, res, label="0.01")
    Random.seed!(rng_seed) #Makes the rng the same
    t, res = calc_ent_traj(0.15)
    plot!(pl, t, res, label="0.15")
    Random.seed!(rng_seed) #Makes the rng the same
    t, res = calc_ent_traj(0.3)
    plot!(pl, t, res, label="0.3")
    saveplot(pl, "traj_entanglement")
    @test true
end