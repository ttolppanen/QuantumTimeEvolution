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

include("./tests/twoqubitstest.jl")
include("./tests/bosonstacktest.jl")
include("./tests/measurementtest.jl")
include("./tests/typetest.jl")
