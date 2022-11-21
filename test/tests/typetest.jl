function testtype(state)
    @test isa(state, AbstractVector{<:Number})
    @test issparse(state)
    @test typeof(state) == typeof(complex(state))
end

@testset "Types" begin
    d = 4; L = 4
    dt = 0.1; t = 1.0
    state = zeroone(d, L)
    indices = siteinds("Boson", L; dim = d)
    mps0 = MPS(Vector(state), indices)
    result = exactevolve_bosehubbard(d, L, state, dt, t)
    testtype(result[8])
    k = 4
    result = krylovevolve_bosehubbard(d, L, state, dt, t, k)
    testtype(result[6])
    result = mpsevolve_bosehubbard(mps0, dt, t)
    @test isa(result, Vector{MPS})
end