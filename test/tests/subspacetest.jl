# using QuantumStates
# using QuantumOperators
# using ITensors
# using LinearAlgebra
# using Plots

@testset "Subspace Two Qubits" begin

d = 2; L = 2; k = 5
dt = 0.1; t = 5.0
one_boson_site = 1
indices, perm_mat, ranges = total_boson_number_subspace(d, L)
finder(state) = find_subspace(state, ranges)
state = perm_mat * singleone(d, L, one_boson_site)
H = perm_mat * bosehubbard(d, L) * perm_mat'
U_op = perm_mat * exp(-im * dt * Matrix(H)) * perm_mat'
ntot = perm_mat * nall(d, L) * perm_mat'
n = perm_mat * singlesite_n(d, L, one_boson_site) * perm_mat'
observables = [(state, subspace_indices) -> norm(@view(state[subspace_indices])), (state, subspace_indices) -> expval(state, ntot, subspace_indices), (state, subspace_indices) -> expval(state, n, subspace_indices)]
r_exact = exactevolve(state, U_op, dt, t, observables...; find_subspace = finder)
r_krylov = krylovevolve(state, H, dt, t, k, observables...; find_subspace = finder)
r_all = [r_exact, r_krylov]

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