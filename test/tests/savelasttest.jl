#using QuantumStates
#using QuantumOperators

@testset "savelast-argument" begin
    
function lasttest(full_evolution, last_evolution)
    @test length(last_evolution) == 1
    @test full_evolution[end] ≈ last_evolution[1]
end

d = 4; L = 4
dt = 0.1; t = 2.0
state0 = onezero(d, L)
mps0 = onezeromps(d, L)

r_exact = exactevolve_bosehubbard(d, L, state0, dt, t)
r_krylov = krylovevolve_bosehubbard(d, L, state0, dt, t, 8)
r_mps = mpsevolve_bosehubbard(mps0, dt, t)

r_last_exact = exactevolve_bosehubbard(d, L, state0, dt, t; savelast=true)
r_last_krylov = krylovevolve_bosehubbard(d, L, state0, dt, t, 8; savelast=true)
r_last_mps = mpsevolve_bosehubbard(mps0, dt, t; savelast=true)

lasttest(r_exact, r_last_exact)
lasttest(r_krylov, r_last_krylov)
lasttest(r_mps, r_last_mps)

end # testset