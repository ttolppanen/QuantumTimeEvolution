#using QuantumStates
#using QuantumOperators

@testset "savelast-argument" begin
    
function lasttest(full_evolution, last_evolution)
    @test length(last_evolution) == 1
    @test full_evolution[end] ≈ last_evolution[1]
end

d = 4; L = 4
dt = 0.1; t = 1 + rand()
state0 = onezero(d, L)
mps0 = onezeromps(d, L)
H = bosehubbard(d, L)
U_op = exp(-im * dt * Matrix(H))
gates = bosehubbardgates(siteinds(mps0), dt)

r_exact = exactevolve(state0, U_op, dt, t, state -> entanglement(d, L, state, Int(L / 2)))
r_krylov = krylovevolve(state0, H, dt, t, 8, state -> entanglement(d, L, state, Int(L / 2)))
r_mps = mpsevolve(mps0, gates, dt, t, state -> entanglement(state, Int(L / 2)))

r_last_exact = exactevolve(state0, U_op, dt, t, state -> entanglement(d, L, state, Int(L / 2)); save_only_last=true)
r_last_krylov = krylovevolve(state0, H, dt, t, 8, state -> entanglement(d, L, state, Int(L / 2)); save_only_last=true)
r_last_mps = mpsevolve(mps0, gates, dt, t, state -> entanglement(state, Int(L / 2)); save_only_last=true)

lasttest(r_exact[1, :], r_last_exact[1, :])
lasttest(r_krylov[1, :], r_last_krylov[1, :])
lasttest(r_mps[1, :], r_last_mps[1, :])

end # testset