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

include("twoqubitstest.jl")
include("measurementtest.jl")
include("typetest.jl")
