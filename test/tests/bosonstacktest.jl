# using QuantumStates
# using QuantumOperators
# using ITensors
# using Plots

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
H = bosehubbard(d, L; U, J)
U_op = exp(-im * dt * Matrix(H))
gates = bosehubbardgates(siteinds(mps0), dt; U, J)

r_exact = exactevolve(state0, U_op, dt, t)
r_krylov = krylovevolve(state0, H, dt, t, 10)
r_mps = mpsevolve(mps0, gates, dt, t)

n = singlesite_n(d, L, target)
x_t = (0:dt:t) / (pi / J_e)
pl = plot(x_t, expval(r_exact, n), label="exact", title = "U = $(U), J = $(J) (MPS fails because dt too small)")
plot!(pl, x_t, expval(r_krylov, n), label="krylov")
plot!(pl, x_t, expval(r_mps, "N"; sites=3), label="mps")
plot!(pl, x_t, x -> N * (cos(x * pi)^2), label="analytical", linestyle = :dash)
saveplot(pl, "bosonstack")
@test true

end #test set