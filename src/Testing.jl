using TimeEvolution
using Plots

include("Operators.jl")

function f()
    d = 3; L = 4
    dt = 0.1; t = 10
    dn = [1.0 + 0im, 0.0, 0.0]
    up = [0.0, 0.0, 1.0]
    state = kron(dn, up, dn, up)
    result = exactevolvebosehubbard(d, L, state, dt, t)
    n = numberOperator(d) ⊗ Matrix(I, d^(L-1), d^(L-1))
    result = [real(Ψ' * n * Ψ) for Ψ in result]
    plot(0:dt:t, result)
end

f()