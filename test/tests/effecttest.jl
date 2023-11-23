# using QuantumStates
# using QuantumOperators
# using Random

# These tests just check that code runs, and that the subspace
# results are the same as in the normal case, with measurement and feedback.

@testset "Effects" begin

d = 2; L = 4;
dt = 0.02; t = 30.0; k = 6
p = 0.01
rng_seed = 7

@testset "Subspace and Total Space Give Same Result" begin
    state = allone(d, L)
    H = bosehubbard(d, L)
    n = nall(d, L)
    n1 = singlesite_n(d, L, 1)
    observables = [state -> expval(state, n), state -> expval(state, n1)]


    msrop = measurementoperators(nop(d), L)
    feedback = [singlesite(n_bosons_projector(d, 0), L, i) for i in 1:L]

    effect!(state) = random_measurement!(state, msrop, p)
    krylovevolve(state, H, dt, t, k, observables...; effect!)
    
    effect_subspace!(state) = random_measurement_feedback!(state, msrop, p, feedback)
    Random.seed!(rng_seed) # Makes the rng the same
    k_r = krylovevolve(state, H, dt, t, k, observables...; effect! = effect_subspace!)

    # in subspace
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

    effect_subspace!(state, id) = random_measurement_feedback!(state, id, msrop, p, feedback; skip_subspaces = 1)
    Random.seed!(rng_seed) # Makes the rng the same
    sb_r = krylovevolve(state, initial_id, H, dt, t, k, observables...; effect! = effect_subspace!)
    
    @test round(norm(sb_r .- k_r), digits = 13) == 0.0 # testing if the results from no subspace and subspace are the same
end

end # test set