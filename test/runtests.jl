using QuantumTimeEvolution, Test
using Plots
using SparseArrays
using LinearAlgebra
using ITensors
using Random
using QuantumStates
using QuantumOperators

function saveplot(pl, name)
    mkpath("./plots/")
    savefig(pl, "./plots/" * name * ".png")
end


include("tests/twoqubitstest.jl")
include("tests/bosonstacktest.jl")
include("tests/trotterodertest.jl")
include("tests/measurementtest.jl")
include("tests/savebeforeaftertest.jl")
include("tests/savelasttest.jl")
# include("tests/typetest.jl") states are not returned anymore, so testing this doesn't make sense anymore.
