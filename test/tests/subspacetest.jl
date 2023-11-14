# using QuantumStates
# using QuantumOperators
# using ITensors
# using LinearAlgebra
# using Plots

@testset "Subspace Two Qubits" begin

d = 2; L = 2; k = 5
dt = 0.1; t = 5.0
one_boson_site = 1
state = singleone(d, L, one_boson_site)
H = bosehubbard(d, L)
U_op = exp(-im * dt * Matrix(H))
ntot = nall(d, L)
n = singlesite_n(d, L, one_boson_site)
find_subspace = generate_total_boson_number_subspace_finder(d, L)
observables = [(state, subspace_indices) -> norm(@view(state[subspace_indices])), (state, subspace_indices) -> expval(state, ntot, subspace_indices), (state, subspace_indices) -> expval(state, n, subspace_indices)]
r_exact = exactevolve(state, U_op, dt, t, observables...; find_subspace)
r_krylov = krylovevolve(state, H, dt, t, k, observables...; find_subspace)
r_all = [r_exact, r_krylov]

@testset "Initial State Unaffected" begin
    @test state == singleone(d, L, one_boson_site) #evolution doesn't change initial state
end

@testset "Normalization" begin
    for r in r_all
        @test all([r_i ≈ 1.0 for r_i in r[1, :]]) #norm should be one
    end
end

@testset "Total Boson Number" begin
    for r in r_all
        @test all([val ≈ 1.0 for val in r[2, :]])
    end
end

@testset "First Site Boson Number" begin
    x_t = (0:dt:t) / pi
    n_res = [r_exact[3, :], r_krylov[3, :]]
    pl = plot(x_t, n_res[1], label="exact", title = "J = 1")
    plot!(pl, x_t, n_res[2], linestyle=:dash, label="krylov")
    plot!(pl, x_t, x->cos(x * pi)^2, label="analytical")
    saveplot(pl, "subspace first site boson number")
    @test true
end

end # testset