# using QuantumStates
# using QuantumOperators
# using ITensors
# using LinearAlgebra
# using Plots

@testset "Subspace Two Qubits" begin

d = 2; L = 2; k = 5
dt = 0.1; t = 5.0
one_boson_site = 1
perm_mat, ranges = total_boson_number_subspace_tools(d, L)
finder(state) = find_subspace(state, ranges)
state = perm_mat * singleone(d, L, one_boson_site)

H = bosehubbard(d, L)
sub_H = split_operator(H, perm_mat, ranges)
U_op = exp(-im * dt * Matrix(H))
sub_U = split_operator(U_op, perm_mat, ranges)
ntot = nall(d, L)
sub_ntot = split_operator(ntot, perm_mat, ranges)
n = singlesite_n(d, L, one_boson_site)
sub_n = split_operator(n, perm_mat, ranges)

observables = [
    (state, id, range) -> norm(@view(state[range])),
    (state, id, range) -> expval(@view(state[range]), sub_ntot[id]),
    (state, id, range) -> expval(@view(state[range]), sub_n[id])]

r_exact = exactevolve(state, sub_U, finder, dt, t, observables...)
r_krylov = krylovevolve(state, sub_H, finder, dt, t, k, observables...)
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