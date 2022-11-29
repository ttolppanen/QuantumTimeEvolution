using QuantumStates
using QuantumOperators
using QuantumTimeEvolution

function f(L)
    d = 2;
    dt = 0.02; t=20
    state = zeroone(d, L)
    H = bosehubbard(d, L)
    @time krylovevolve(state, H, dt, t, 6; savelast=true)
end

f(4);
f(12);