# using QuantumStates
# using QuantumOperators
# using ITensors
# using LinearAlgebra
# using Plots

@testset "Two Qubits" begin

d = 2; L = 2; k = 2
dt = 0.1; t = 5.0
one_boson_site = 1
state = singleone(d, L, one_boson_site)
indices = siteinds("Boson", L; dim = d)
mps0 = MPS(Vector(state), indices)
r_exact = exactevolve_bosehubbard(d, L, state, dt, t)
r_krylov = krylovevolve_bosehubbard(d, L, state, dt, t, k)
r_mps = mpsevolve_bosehubbard(mps0, dt, t)
r_all = [r_exact, r_krylov, r_mps]

@testset "Initial State Unaffected" begin
    @test state == singleone(d, L, one_boson_site) #evolution doesn't change initial state
    @test inner(mps0, MPS(Vector(state), indices)) == 1.0
end

@testset "Normalization" begin
    for r in r_all
        @test all([norm(state) ≈ 1.0 for state in r]) #norm should be one
    end
end

@testset "Total Boson Number" begin
    ntot = nall(d, L)
    @test all([val ≈ 1.0 for val in expval(r_exact, ntot)])
    @test all([val ≈ 1.0 for val in expval(r_krylov, ntot)])
    @test all([sum(val) ≈ 1.0 for val in expval(r_mps, "N")])
end

@testset "First Site Boson Number" begin
    n = singlesite_n(d, L, one_boson_site)
    x_t = (0:dt:t) / pi
    n_res = [expval(r_exact, n), expval(r_krylov, n), expval(r_mps, "N"; sites=2)]
    pl = plot(x_t, n_res[1], label="exact", title = "J = 1")
    plot!(pl, x_t, n_res[2], linestyle=:dash, label="krylov")
    plot!(pl, x_t, n_res[3], linestyle=:dashdot, label="MPS")
    plot!(pl, x_t, x->cos(x * pi)^2, label="analytical")
    saveplot(pl, "first site boson number")

    pl = plot(x_t, [(n_res[1][i] - cos(x_t[i] * pi)^2) for i in eachindex(n_res[1])], title = "Difference to Analytical - exact, dt = $(dt)")
    saveplot(pl, "diff_analytical_exact")
    pl = plot(x_t, [(n_res[2][i] - cos(x_t[i] * pi)^2) for i in eachindex(n_res[2])], title = "Difference to Analytical - krylov, dt = $(dt)")
    saveplot(pl, "diff_analytical_krylov")
    pl = plot(x_t, [(n_res[3][i] - cos(x_t[i] * pi)^2) for i in eachindex(n_res[3])], title = "Difference to Analytical - mps, dt = $(dt)")
    saveplot(pl, "diff_analytical_mps")
    @test true
end

end # testset