# using QuantumStates
# using QuantumOperators

# These tests just check that code runs.
# It feels too difficult to figure how to test
# effect that have random outcomes...

@testset "Effects" begin

d = 2; L = 16;
dt = 0.02; t = 30.0; k = 6
p = 0.01

@testset "Full Space" begin
    state = allone(d, L)
    H = bosehubbard(d, L)
    n = nall(d, L)
    n1 = singlesite_n(d, L, 1)
    observables = [state -> expval(state, n), state -> expval(state, n1)]


    msrop = measurementoperators(nop(d), L)
    feedback = [singlesite(n_bosons_projector(d, 0), L, i) for i in 1:L]

    effect!(state) = random_measurement!(state, msrop, p)
    krylovevolve(state, H, dt, t, k, observables...; effect!)
    
    effect!(state) = random_measurement_feedback!(state, msrop, p, feedback)
    krylovevolve(state, H, dt, t, k, observables...; effect!)
    @test true
end

@testset "SubSpace" begin
    state = allone(d, L)
    indices = total_boson_number_subspace_indices(d, L)
    ranges, perm_mat = total_boson_number_subspace_tools(d, L)
    initial_id = find_subspace(state, indices)

    state = subspace_split(state, ranges, perm_mat)
    H = subspace_split(H, ranges, perm_mat)
    n = subspace_split(n, ranges, perm_mat)
    n1 = subspace_split(n1, ranges, perm_mat)
    observables = [
        (state, id) -> expval(state[id], n[id]),
        (state, id) -> expval(state[id], n1[id])]

    msrop = measurementoperators(nop(d), L)
    msrop = measurement_subspace(msrop, ranges, perm_mat)
    feedback = [singlesite(n_bosons_projector(d, 0), L, i) for i in 1:L]
    feedback = feedback_measurement_subspace(feedback, msrop, indices; digit_error = 10, id_relative_guess = -1)

    effect!(state, id) = random_measurement!(state, id, msrop, p)
    krylovevolve(state, initial_id, H, dt, t, k, observables...; effect!)

    effect!(state, id) = random_measurement_feedback!(state, id, msrop, p, feedback; skip_subspaces = 1)
    krylovevolve(state, initial_id, H, dt, t, k, observables...; effect!)
    @test true
end

end # test set