# using QuantumStates
# using QuantumOperators
# using ITensors
# using LinearAlgebra
# using Plots

@testset "Subspace Two Qubits" begin

d = 2; L = 2; k = 5
dt = 0.1; t = 5.0
one_boson_site = 1
indices = total_boson_number_subspace_indices(d, L)
state = singleone(d, L, one_boson_site)
initial_id = find_subspace(state, indices)

state = subspace_split(state, indices)
H = bosehubbard(d, L)
U = subspace_split(exp(-im * dt * Matrix(H)), indices)
H = subspace_split(H, indices)
ntot = subspace_split(nall(d, L), indices)
n = subspace_split(singlesite_n(d, L, one_boson_site), indices)

observables = [
    (state, id) -> norm(state[id]),
    (state, id) -> expval(state[id], ntot[id]),
    (state, id) -> expval(state[id], n[id])]

r_exact = exactevolve(state, initial_id, U, dt, t, observables...)
r_krylov = krylovevolve(state, initial_id, H, dt, t, k, observables...)
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