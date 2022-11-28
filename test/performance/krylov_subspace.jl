using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using LinearAlgebra

function old_lanczos!(H, state, sdim, V, h, w)
    #V .= zero(ComplexF64)
    #h .= zero(ComplexF64)
    V[:, 1] .= state
    mul!(w, H, state)
    h[1, 1] = w' * state
    w .-= h[1, 1] .* state
    for j in 2:sdim
        beta = norm(w)
        V[:, j] .= w ./ beta
        @views mul!(w, H, V[:, j])
        @views h[j, j] = w' * V[:, j]
        @views w .-= (h[j, j] .* V[:, j] .+ beta .* V[:, j - 1])
        h[j - 1, j] = beta
        h[j, j - 1] = beta
    end
end

function f()
    d = 3; L = 4
    state = zeroone(d, L)
    k = 6
    H = bosehubbard(d, L)
    w = deepcopy(state)
    pa_k = QuantumTimeEvolution.PA_krylov(length(state), k)
    old_lanczos!(H, state, k, pa_k.U, pa_k.H_k, w);
    old_lanczos!(H, state, k, pa_k.U, pa_k.H_k, w);
    @time QuantumTimeEvolution.krylovsubspace!(state, H, k, pa_k.H_k, pa_k.U, pa_k.z)
    @code_warntype QuantumTimeEvolution.krylovsubspace!(state, H, k, pa_k)
end

f();