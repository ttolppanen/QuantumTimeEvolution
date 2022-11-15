using QuantumTimeEvolution, Test
using Plots
using SparseArrays
using LinearAlgebra
using ITensors
using QuantumTimeEvolution.QuantumStates
using QuantumTimeEvolution.QuantumOperators

function saveplot(pl, name)
    savefig(pl, "./plots/" * name * ".png")
end

include("measurementtest.jl")

@testset "Plots" begin
    d = 3; L = 4
    dt = 0.1; t = 5.0
    state = zeroone(d, L)
    result = exactevolve_bosehubbard(d, L, state, dt, t)
    @test state == zeroone(d, L)

    ntot = nall(d, L)
    res = expval(result, ntot)
    pl = plot(0:dt:t, res, ylims=(1.8, 2.2))
    saveplot(pl, "total boson number")

    res = [norm(s) for s in result]
    pl = plot(0:dt:t, res, ylims=(0.9, 1.1))
    saveplot(pl, "state norm")
    
    n = singlesite_n(d, L, 1)
    res = expval(result, n)
    pl = plot(0:dt:t, res, label="exact")
    
    k = 4
    result = krylovevolve_bosehubbard(d, L, state, dt, t, k)
    n = singlesite_n(d, L, 1)
    res = expval(result, n)
    pl = plot!(pl, 0:dt:t, res, linestyle=:dash, label="krylov")

    indices = siteinds("Boson", L; dim = d)
    mps0 = MPS(Vector(state), indices)
    result = mpsevolve_bosehubbard(mps0, dt, t)
    res = expval(result, "N"; sites=L)
    pl = plot!(pl, 0:dt:t, res, linestyle=:dashdot, label="MPS")
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
