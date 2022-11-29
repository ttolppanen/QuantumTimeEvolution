# using QuantumStates
# using ITensors

function testtype(state)
    @test isa(state, AbstractVector{<:Number})
    @test typeof(state) == typeof(complex(state))
end

@testset "Types" begin
    d = 4; L = 4
    dt = 0.1; t = 1.0
    state = zeroone(d, L)
    indices = siteinds("Boson", L; dim = d)
    mps0 = MPS(Vector(state), indices)
    H = bosehubbard(d, L)
    U_op = exp(-im * dt * Matrix(H))
    gates = bosehubbardgates(siteinds(mps0), dt)
    result = exactevolve(state, U_op, dt, t)
    testtype(result[8])
    k = 4
    result = krylovevolve(state, H, dt, t, k)
    testtype(result[6])
    result = mpsevolve(mps0, gates, dt, t)
    @test isa(result, Vector{MPS})
end