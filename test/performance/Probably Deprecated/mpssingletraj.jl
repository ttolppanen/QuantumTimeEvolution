using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using ITensors
using Plots

function f(L)
    d = 2;
    dt = 1.0; t=10
    state = zeroonemps(d, L)
    gates = bosehubbardgates(siteinds(state), dt; k=2)
    r = mpsevolve(state, gates, dt, t)
    t = 0:dt:t
    res = expval(r, "N";sites=1)
    plot(t, x->cos(x)^2)
    plot!(t, res)
end

f(2)