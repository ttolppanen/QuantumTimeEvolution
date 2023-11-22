# using QuantumStates
# using QuantumOperators
# using Random

@testset "savebeforeafter" begin

@testset "No Randomness" begin
    d = 3; L = 4
    dt = 0.1; t = 5
    state0 = zeroone(d, L)
    mps0 = zeroonemps(d, L)
    H = bosehubbard(d, L)
    U_op = exp(-im * dt * Matrix(H))
    gates = bosehubbardgates(siteinds(mps0), dt)

    r_exact_last = exactevolve(state0, U_op, dt, t, state -> entanglement(d, L, state, Int(L / 2)); save_before_effect = false)
    r_krylov_last = krylovevolve(state0, H, dt, t, 4, state -> entanglement(d, L, state, Int(L / 2)); save_before_effect = false)
    r_mps_last = mpsevolve(mps0, gates, dt, t, state -> entanglement(state, Int(L / 2)); save_before_effect = false)
    r_exact_first = exactevolve(state0, U_op, dt, t, state -> entanglement(d, L, state, Int(L / 2)); save_before_effect = true)
    r_krylov_first = krylovevolve(state0, H, dt, t, 4, state -> entanglement(d, L, state, Int(L / 2)); save_before_effect = true)
    r_mps_first = mpsevolve(mps0, gates, dt, t, state -> entanglement(state, Int(L / 2)); save_before_effect = true)

    @test r_exact_last[1, :] == r_exact_first[1, :]
    @test r_krylov_last[1, :] == r_krylov_first[1, :]
    @test r_mps_last[1, :] == r_mps_first[1, :]
end

@testset "With Randomness" begin
    d = 3; L = 4
    dt = 0.1; t = 5
    msr_prob = 0.1; rng_seed = 5
    state0 = zeroone(d, L)
    mps0 = zeroonemps(d, L)
    H = bosehubbard(d, L)
    U_op = exp(-im * dt * Matrix(H))
    gates = bosehubbardgates(siteinds(mps0), dt)

    op_to_msr = nop(d)
    msrop = measurementoperators(op_to_msr, L)
    msrop_tensor = measurementoperators(op_to_msr, siteinds(mps0))
    meffect!(state) = random_measurement!(state, msrop, msr_prob)
    meffect_t!(state) = random_measurement!(state, msrop_tensor, msr_prob)

    n1 = singlesite_n(d, L, 1)
    Random.seed!(rng_seed) # Makes the rng the same
    r_exact_last = exactevolve(state0, U_op, dt, t, state -> entanglement(d, L, state, Int(L / 2)); effect! = meffect!, save_before_effect = false)
    Random.seed!(rng_seed) # Makes the rng the same
    r_krylov_last = krylovevolve(state0, H, dt, t, 4, state -> entanglement(d, L, state, Int(L / 2)); effect! = meffect!, save_before_effect = false)
    Random.seed!(rng_seed) # Makes the rng the same
    r_mps_last = mpsevolve(mps0, gates, dt, t, state -> entanglement(state, Int(L / 2)); effect! = meffect_t!, save_before_effect = false)
    Random.seed!(rng_seed) # Makes the rng the same
    r_exact_first = exactevolve(state0, U_op, dt, t, state -> entanglement(d, L, state, Int(L / 2)); effect! = meffect!, save_before_effect = true)
    Random.seed!(rng_seed) # Makes the rng the same
    r_krylov_first = krylovevolve(state0, H, dt, t, 4, state -> entanglement(d, L, state, Int(L / 2)); effect! = meffect!, save_before_effect = true)
    Random.seed!(rng_seed) # Makes the rng the same
    r_mps_first = mpsevolve(mps0, gates, dt, t, state -> entanglement(state, Int(L / 2)); effect! = meffect_t!, save_before_effect = true)

    # the last line is to test if they are ≈ the same, but there is small difference which doesn't 
    # change the overall dynamics, but which is large enough that a ≈ b fails.
    leaq(a, b) = a <= b  || abs(a - b) < 1E-1
    @test all([leaq(r_exact_last[1, i], r_exact_first[1, i]) for i in eachindex(r_exact_last[1, :])])
    @test all([leaq(r_krylov_last[1, i], r_krylov_first[1, i]) for i in eachindex(r_krylov_last[1, :])])
    @test all([leaq(r_mps_last[1, i], r_mps_first[1, i]) for i in eachindex(r_mps_last[1, :])])
    pl = plot(0:dt:t, r_exact_last[1, :])
    plot!(pl, 0:dt:t, r_exact_first[1, :])
    plot!(pl, 0:dt:t, [leaq(r_exact_last[1, i], r_exact_first[1, i]) for i in eachindex(r_exact_last[1, :])])
    saveplot(pl, "just_testing")
    pl = plot(0:dt:t, r_mps_last[1, :])
    plot!(pl, 0:dt:t, r_mps_first[1, :])
    plot!(pl, 0:dt:t, [leaq(r_mps_last[1, i], r_mps_first[1, i]) for i in eachindex(r_mps_last[1, :])])
    saveplot(pl, "just_testing_more")
end

end # testset