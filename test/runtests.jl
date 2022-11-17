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
    d = 3; L = 5
    dt = 0.01; t = 5.0
    center_site = 3
    state = singleone(d, L, center_site)
    result = exactevolve_bosehubbard(d, L, state, dt, t)
    @test state == singleone(d, L, center_site)

    ntot = nall(d, L)
    res = expval(result, ntot)
    pl = plot(0:dt:t, res, ylims=(0.99, 1.01), label="exact")
    saveplot(pl, "total boson number")

    res = [norm(s) for s in result]
    pl = plot(0:dt:t, res, ylims=(0.99, 1.01), label="exact")
    saveplot(pl, "state norm")
    
    n = singlesite_n(d, L, center_site)
    res = expval(result, n)
    pl = plot(0:dt:t, res, label="exact")
    
    k = 4
    result = krylovevolve_bosehubbard(d, L, state, dt, t, k)
    res = expval(result, n)
    pl = plot!(pl, 0:dt:t, res, linestyle=:dash, label="krylov")

    indices = siteinds("Boson", L; dim = d)
    mps0 = MPS(Vector(state), indices)
    result = mpsevolve_bosehubbard(mps0, dt, t)
    res = expval(result, "N"; sites=center_site)
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
