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
gates = bosehubbardgates(siteinds(mps0), dt)

r_exact = exactevolve(state0, H, dt, t)
r_krylov = krylovevolve(state0, H, dt, t, 8)
r_mps = mpsevolve(mps0, gates, dt, t)

r_last_exact = exactevolve(state0, H, dt, t; savelast=true)
r_last_krylov = krylovevolve(state0, H, dt, t, 8; savelast=true)
r_last_mps = mpsevolve(mps0, gates, dt, t; savelast=true)

lasttest(r_exact, r_last_exact)
lasttest(r_krylov, r_last_krylov)
lasttest(r_mps, r_last_mps)

end # testset