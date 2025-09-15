using QuantumTimeEvolution, Test
using Plots
using SparseArrays
using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using QuantumStates
using QuantumOperators
using JLD2

function saveplot(pl, name)
    mkpath("./plots/")
    savefig(pl, "./plots/" * name * ".png")
end

include("tests/twoqubitstest.jl")
include("tests/arnolditest.jl")
include("tests/bosonstacktest.jl")
include("tests/trotterodertest.jl")
include("tests/savebeforeaftertest.jl")
include("tests/subspacetest.jl")
include("tests/effecttest.jl")
include("tests/savelasttest.jl")
include("tests/paraltest.jl")
include("tests/dissdecotest.jl")
include("tests/periodicmsreffecttest.jl")
include("tests/measurementtest.jl") # run this last...

# include("tests/typetest.jl") states are not returned anymore, so testing this doesn't make sense anymore.
