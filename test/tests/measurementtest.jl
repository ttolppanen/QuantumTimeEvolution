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
    mps0 = onezeromps(d, L) # reverse order for MPS
    H = bosehubbard(d, L)
    U_op = exp(-im * dt * Matrix(H))
    gates = bosehubbardgates(siteinds(mps0), dt)

    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, L)
    msrop_tensor = measurementoperators(op_to_msr, siteinds(mps0))
    meffect!(state) = random_measurement!(state, msrop, msr_prob)
    meffect_t!(state) = random_measurement!(state, msrop_tensor, msr_prob)

    n1 = singlesite_n(d, L, 1)
    observables = [norm, state -> expval(state, n1)]
    observables_mps = [norm, state -> expval(state, "N"; sites=1)]
    Random.seed!(rng_seed) # Makes the rng the same
    r_exact = exactevolve(state0, U_op, dt, t, observables...; effect! = meffect!)
    Random.seed!(rng_seed) # Makes the rng the same
    r_krylov = krylovevolve(state0, H, dt, t, 4, observables...; effect! = meffect!)
    Random.seed!(rng_seed) # Makes the rng the same
    r_mps = mpsevolve(mps0, gates, dt, t, observables_mps...; effect! = meffect_t!)

    plot_x = 0:dt:t
    @testset "Normalization" begin
        for r in [r_exact, r_krylov, r_mps]
            @test all([val ≈ 1.0 for val in r[1, :]]) # norm should be on
        end
    end
    @testset "First Site Boson Number" begin
        # with mps, the first site is the last site?
        # So here this shouldnt work, since the measurements happen
        # to the wrong sites (There was a measurement on the first
        # site -> msr on the last site for mps). But since the state is
        # symmetrical, it works.

        pl = plot(plot_x, r_exact[2, :], label="exact")
        plot!(pl, plot_x, r_krylov[2, :], label="krylov")
        plot!(pl, plot_x, r_mps[2, :], label="mps")
        saveplot(pl, "msr_bosonnumber")
        @test true
    end
    @test true
end

function make_projections_to_zeroes(d, L)
    out = []
    m = n_bosons_projector(d, 0)
    for i in 1:L
        push!(out, singlesite(m, L, i))
    end
    return out
end

@testset "random_predetermined_measurement!" begin
    d = 3; L = 3
    dt = 0.1; t = 5
    msr_prob = 0.1
    rng_seed = 4
    state0 = zeroone(d, L)
    H = bosehubbard(d, L)

    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, L)
    proj_op = make_projections_to_zeroes(d, L)

    n = nall(d, L)
    observables = [norm, state -> expval(state, n)]
    Random.seed!(rng_seed) # Makes the rng the same
    proj_prob = 0.0
    r_krylov_1 = krylovevolve(state0, H, dt, t, 4, observables...; effect! = state -> random_measurement_random_feedback!(state, msrop, msr_prob, proj_op, proj_prob))
    
    Random.seed!(rng_seed) # Makes the rng the same
    proj_prob = 1.0
    r_krylov_2 = krylovevolve(state0, H, dt, t, 4, observables...; effect! = state -> random_measurement_random_feedback!(state, msrop, msr_prob, proj_op, proj_prob))

    plot_x = 0:dt:t
    @testset "Normalization" begin
        for r in [r_krylov_1, r_krylov_2]
            @test all([val ≈ 1.0 for val in r[1, :]]) # norm should be on
        end
    end
    @testset "Total boson number" begin
        pl = plot(plot_x, r_krylov_1[2, :], label="r = 0")
        plot!(pl, plot_x, r_krylov_2[2, :], label="r = 1")
        saveplot(pl, "random_predetermined_bosonnumber")
        @test true
    end
    @test true
end

function traj_mean(result)
    out = zeros(size(result[1]))
    for traj in result
        out .+= traj
    end
    return out ./ length(result)
end
function calc_ent_traj(msr_prob)
    d = 2; L = 4
    dt = 0.1; t = 5
    traj = 30
    mps0 = zeroonemps(d, L)
    gates = bosehubbardgates(siteinds(mps0), dt)
    meffect! = random_measurement_function(siteinds(mps0), nop(d), msr_prob)
    observables = [state -> entanglement(state, 2)]
    r_f() = mpsevolve(mps0, gates, dt, t, observables...; effect! = meffect!, cutoff = 1E-8)
    result = solvetrajectories(r_f, traj)
    res = traj_mean(result)
    return 0:dt:t, res[1, :]
end
@testset "Trajectories" begin
    rng_seed = 3
    Random.seed!(rng_seed) # Makes the rng the same
    t, res = calc_ent_traj(0.01)
    pl = plot(t, res, label="0.01")
    Random.seed!(rng_seed) # Makes the rng the same
    t, res = calc_ent_traj(0.15)
    plot!(pl, t, res, label="0.15")
    Random.seed!(rng_seed) # Makes the rng the same
    t, res = calc_ent_traj(0.3)
    plot!(pl, t, res, label="0.3")
    saveplot(pl, "traj_entanglement")
    @test true
end
@testset "Trajectories with mipt" begin
    d = 2; L = 4
    dt = 0.1; t = 5
    prob = [0.01, 0.15, 0.3]
    traj = 50
    state0 = zeroone(d, L)
    H = bosehubbard(d, L)
    msrop = measurementoperators(nop(d), L)
    meffect!(state, msr_prob) = random_measurement!(state, msrop, msr_prob)
    calc_ent(state) = entanglement(d, L, state, 2)
    res1 = mipt(state0, H, 6, meffect!, dt, t, prob, traj, calc_ent; paral = :threads)
    pl = plot(prob, res1)
    mps0 = onezeromps(d, L)
    gates = bosehubbardgates(siteinds(mps0), dt)
    msrop = measurementoperators(nop(d), siteinds(mps0))
    meffect2!(state, msr_prob) = random_measurement!(state, msrop, msr_prob)
    calc_ent2(state) = entanglement(state, 2)
    res2 = mipt(mps0, gates, meffect2!, dt, t, prob, traj, calc_ent2; paral = :threads)
    plot!(pl, prob, res2)
    saveplot(pl, "mipt_test")
    @test all([abs(res1[i] - res2[i]) < 0.2 for i in 1:length(prob)]) # close enough...
end
@testset "measuresitesrandomly overload" begin
    d = 3; L = 4; half = Int(floor(L/2))
    dt = 0.1; t = 1.0
    rng_seed = 5; msr_prob = 0.5
    state = zeroone(d, L)
    H = bosehubbard(d, L)
    msrop = measurementoperators(nop(d), L)
    effect!(state) = random_measurement!(state, msrop, msr_prob)
    observables = [state -> entanglement(d, L, state, half)]
    Random.seed!(rng_seed)
    r_k = krylovevolve(state, H, dt, t, 5, observables...; effect!)
    r1 = r_k[1, :]
    effect! = random_measurement_function(L, nop(d), msr_prob)
    Random.seed!(rng_seed)
    r_k = krylovevolve(state, H, dt, t, 5, observables...; effect!)
    r2 = r_k[1, :]
    @test norm(r2 - r1) + 1 ≈ 1.0
end

end # testset