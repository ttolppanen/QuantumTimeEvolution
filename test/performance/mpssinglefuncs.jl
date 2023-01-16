using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using ITensors
using Plots

function mps(d, L, dt, t; addplot=false)
    steps = length(0:dt:t) - 1
    state = zeroonemps(d, L)
    linkinds(state)
    gates = bosehubbardgates(siteinds(state), dt; k=2)
    ent(s) = entanglement(s, Int(floor(L/2)))
    r = mpsevolvefunc(state, gates, steps, ent; cutoff = 1E-8, maxdim = 20)
    r = r[:, 1]
    ts = 0:dt:t
    return addplot ? plot!(ts, r) : plot(ts, r)
end

@time mps(3, 12, 0.025, 10; addplot=false)
