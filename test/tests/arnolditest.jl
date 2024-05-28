# using QuantumStates
# using QuantumOperators
# using LinearAlgebra
# using Plots

@testset "Arnoldi Two Qubits" begin

d = 2; L = 4; k = 5
dt = 0.1; t = 5.0
one_boson_site = 1
state = singleone(d, L, one_boson_site)
H = bosehubbard(d, L)
ntot = nall(d, L)
n = singlesite_n(d, L, one_boson_site)
observables = [norm, state -> expval(state, ntot), state -> expval(state, n)]
r_krylov_lancoz = krylovevolve(state, H, dt, t, k, observables...; krylov_alg = :lancoz)
r_krylov_arnoldi = krylovevolve(state, H, dt, t, k, observables...; krylov_alg = :arnoldi)
normalize_state = (state) -> normalize!(state)
r_krylov_lancoz_non_h = krylovevolve(state, H .+ 1im, dt, t, k, observables...; effect! = normalize_state, krylov_alg = :lancoz)
r_krylov_arnoldi_non_h = krylovevolve(state, H .+ 1im, dt, t, k, observables...; effect! = normalize_state, krylov_alg = :arnoldi)
r_all = [r_krylov_lancoz, r_krylov_arnoldi]

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
    n_res = [r_krylov_lancoz[3, :], r_krylov_arnoldi[3, :]]
    pl = plot(x_t, n_res[1], label="lancoz", title = "lancoz vs arnoldi in the krylov algorithm")
    plot!(pl, x_t, n_res[2], linestyle=:dash, label="arnoldi")
    saveplot(pl, "first site boson number - krylov arnoldi")

    @test true
end

@testset "First Site Boson Number - non-hermitian" begin
    x_t = (0:dt:t) / pi
    n_res = [r_krylov_lancoz_non_h[3, :], r_krylov_arnoldi_non_h[3, :]]
    pl = plot(x_t, n_res[1], label="lancoz", titlefontsize = 9, title = "lancoz vs arnoldi in the krylov algorithm for non-hermitian hamiltonian.")
    plot!(pl, x_t, n_res[2], linestyle=:dash, label="arnoldi")
    saveplot(pl, "first site boson number - krylov arnoldi non-hermitian")

    @test true
end

end # testset