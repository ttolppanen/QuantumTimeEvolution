# using QuantumStates
# using QuantumOperators
# using ITensors

@testset "Trotter Order" begin

d = 2; L = 2
mps0 = zeroonemps(d, L)
dt = 0.001; t=0.1
gates = bosehubbardgates(siteinds(mps0), dt; k=1)
observables = [state -> expval(state, "N"; sites=2)]
r_mps = mpsevolve(mps0, gates, dt, t, observables)
res1 = copy(r_mps[1, :])
for k in 2:4
    gates = bosehubbardgates(siteinds(mps0), dt; k=k)
    r_mps = mpsevolve(mps0, gates, dt, t, observables)
    res2 = copy(r_mps[1, :])
    dif = maximum(abs.(res1 .- res2))
    @test dif < 1E-9 # should be at least this small
end

end