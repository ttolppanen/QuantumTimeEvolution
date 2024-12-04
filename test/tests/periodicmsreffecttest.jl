# using QuantumStates
# using QuantumOperators
# using ITensors
# using LinearAlgebra
# using Plots

@testset "Periodic Measurement With Feedback" begin

d = 2; L = 2
dt = 0.1; t = 5.0; msr_rate = t / 5;
one_boson_site = 1

state = singleone(d, L, one_boson_site)
H = bosehubbard(d, L)
U_op = exp(-im * dt * Matrix(H))
ntot = nall(d, L)
n = singlesite_n(d, L, one_boson_site)
observables = [state -> expval(state, ntot), state -> expval(state, n)]

msr_op = measurementoperators(nop(d), L)
feedback = [1 0; 0 1]
fb_op = [singlesite(feedback, L, i) for i in 1:L]

Random.seed!(1) # Makes the rng the same
effect! = periodic_msr_feedback!_function(msr_op, dt, msr_rate, fb_op, [one_boson_site])
r_exact = exactevolve(state, U_op, dt, t, observables...; effect!)

# same in subspace
indices = total_boson_number_subspace_indices(d, L)
ranges, perm_mat = total_boson_number_subspace_tools(d, L)
split(x) = subspace_split(x, ranges, perm_mat)

msr_op = measurement_subspace(msr_op, ranges, perm_mat)
feedback = feedback_measurement_subspace(fb_op, msr_op, indices; digit_error = 10, id_relative_guess = 0)

initial_id = find_subspace(state, indices)
state = split(state)
U_op = split(U_op)

n = split(n)
ntot = split(ntot)
observables = [(state, id) -> expval(state[id], ntot[id]), (state, id) -> expval(state[id], n[id])]

Random.seed!(1) # Makes the rng the same
effect! = periodic_sbspc_msr_feedback!_function(msr_op, dt, msr_rate, feedback, [one_boson_site])
r_exact_subspace = exactevolve(state, initial_id, U_op, dt, t, observables...; effect!)

# results
r_all = [r_exact, r_exact_subspace]

@testset "Total Boson Number" begin
    for r in r_all
        @test all([val ≈ 1.0 for val in r[1, :]])
    end
end

@testset "First Site Boson Number" begin
    x_t = (0:dt:t)
    n_res = [r_exact[2, :], r_exact_subspace[2, :]]
    pl = plot(x_t, n_res[1], title = "msr_rate = t/5, expect ~5 quantum jumps, with 1[time] in between")
    plot!(pl, x_t, n_res[2], linestyle=:dash, label="in subspace")

    saveplot(pl, "periodic measurements")
    @test true
end

end # testset