@testset "Boson Stack" begin
    
d = 4; L = 4; N = 3
U = 50
J = 2
J_e = N / factorial(N - 1) * (J/U)^(N - 1) * J
dt = 1; t = 2 * pi * U
target = 2
state0 = bosonstack(N, L, target)
indices = siteinds("Boson", L; dim = d)
mps0 = MPS(Vector(state0), indices)

r_exact = exactevolve_bosehubbard(d, L, state0, dt, t; U, J)
r_krylov = krylovevolve_bosehubbard(d, L, state0, dt, t, 10; U, J)
r_mps = mpsevolve_bosehubbard(mps0, dt, t; U, J)

n = singlesite_n(d, L, target)
x_t = (0:dt:t) / (pi / J_e)
pl = plot(x_t, expval(r_exact, n), label="exact", title = "U = $(U), J = $(J) (MPS fails because dt too small)")
plot!(pl, x_t, expval(r_krylov, n), label="krylov")
plot!(pl, x_t, expval(r_mps, "N"; sites=3), label="mps")
plot!(pl, x_t, x -> N * (cos(x * pi)^2), label="analytical", linestyle = :dash)
saveplot(pl, "bosonstack")
@test true

end #test set