using QuantumTimeEvolution, Test
using Plots
using SparseArrays
using LinearAlgebra
using ITensors
using Random
using QuantumStates
using QuantumOperators

function saveplot(pl, name)
    savefig(pl, "./plots/" * name * ".png")
end

include("tests/twoqubitstest.jl")
include("tests/bosonstacktest.jl")
include("tests/measurementtest.jl")
include("tests/typetest.jl")
include("tests/savelasttest.jl")
