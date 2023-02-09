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

n = singlesite_n(d, L, target)
observables = [state -> expval(state, n)]
observables_mps = [state -> expval(state, "N"; sites=3)]
r_exact = exactevolve(state0, U_op, dt, t, observables)
r_krylov = krylovevolve(state0, H, dt, t, 10, observables)
r_mps = mpsevolve(mps0, gates, dt, t, observables_mps)

x_t = (0:dt:t) / (pi / J_e)
pl = plot(x_t, r_exact[1, :], label="exact", title = "U = $(U), J = $(J) (MPS fails because dt too small)")
plot!(pl, x_t, r_krylov[1, :], label="krylov")
plot!(pl, x_t, r_mps[1, :], label="mps")
plot!(pl, x_t, x -> N * (cos(x * pi)^2), label="analytical", linestyle = :dash)
saveplot(pl, "bosonstack")
@test true

end #test set