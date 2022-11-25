# using QuantumStates
# using QuantumOperators
# using ITensors
# using Random
# using Plots

@testset "Measurements" begin
    
@testset "measuresitesrandomly!" begin
    d = 3; L = 3
    dt = 0.1; t = 5
    msr_prob = 0.1
    rng_seed = 5
    state0 = zeroone(d, L)
    mps0 = zeroonemps(d, L)
    H = bosehubbard(d, L)
    U_op = exp(-im * dt * Matrix(H))
    gates = bosehubbardgates(siteinds(mps0), dt)
    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, L)
    msrop_tensor = measurementoperators(op_to_msr, siteinds(mps0))
    meffect!(state) = measuresitesrandomly!(state, msrop, msr_prob)
    meffect_t!(state) = measuresitesrandomly!(state, msrop_tensor, msr_prob)
    Random.seed!(rng_seed) #Makes the rng the same
    r_exact = exactevolve(state0, U_op, dt, t; effect! = meffect!)
    Random.seed!(rng_seed) #Makes the rng the same
    r_krylov = krylovevolve(state0, H, dt, t, 5; effect! = meffect!)
    Random.seed!(rng_seed) #Makes the rng the same
    r_mps = mpsevolve(mps0, gates, dt, t; effect! = meffect_t!)
    plot_x = 0:dt:t
    @testset "Normalization" begin
        @test all([norm(state) ≈ 1.0 for state in r_exact]) #norm should be one
        @test all([norm(state) ≈ 1.0 for state in r_krylov])
        @test all([norm(state) ≈ 1.0 for state in r_mps])
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
    gates = bosehubbardgates(siteinds(mps0), dt)
    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, siteinds(mps0))
    meffect!(state) = measuresitesrandomly!(state, msrop, msr_prob)
    r_f() = mpsevolve(mps0, gates, dt, t; effect! = meffect!, cutoff = 1E-8)
    result = solvetrajectories(r_f, 30)
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

end # testset