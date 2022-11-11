using QuantumTimeEvolution, Test
using Plots
using SparseArrays
using LinearAlgebra
using QuantumTimeEvolution.QuantumStates
using QuantumTimeEvolution.QuantumOperators

function waitforkeypress()
    println("press enter here to continue")
    readline()
end
function saveplot(pl, name)
    savefig(pl, "./plots/" * name * ".png")
end
@testset "Plots" begin
    d = 3; L = 4
    dt = 0.1; t = 10.0
    state = zeroone(d, L)
    result = exactevolve_bosehubbard(d, L, state, dt, t; J=4.0)
    @test state == zeroone(d, L)

    ntot = nall(d, L)
    res = [expval(s, ntot) for s in result]
    pl = plot(0:dt:t, res, ylims=(1.8, 2.2))
    saveplot(pl, "total boson number")

    res = [norm(s) for s in result]
    pl = plot(0:dt:t, res, ylims=(0.9, 1.1))
    saveplot(pl, "state norm")
    
    n = singlesite_n(d, L, 1)
    res = [expval(s, n) for s in result]
    pl = plot(0:dt:t, res, label="exact")
    
    k = 4
    result = krylovevolve_bosehubbard(d, L, state, dt, t, k)
    n = singlesite_n(d, L, 1)
    res = [expval(s, n) for s in result]
    pl = plot!(pl, 0:dt:t, res, linestyle=:dash, label="krylov")
    saveplot(pl, "first site boson number")
    @test true
end

function testtype(state)
    @test isa(state, AbstractVector{<:Number})
    @test issparse(state)
    @test typeof(state) == typeof(complex(state))
end

@testset "Types" begin
    d = 4; L = 4
    dt = 0.1; t = 1.0
    state = zeroone(d, L)
    result = exactevolve_bosehubbard(d, L, state, dt, t)
    testtype(result[8])
    k = 4
    result = krylovevolve_bosehubbard(d, L, state, dt, t, k)
    testtype(result[6])
end