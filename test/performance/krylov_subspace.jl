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
    H_k, U, z = QuantumTimeEvolution.krylov_prealloc_Hk_U(length(state), k)
    old_lanczos!(H, state, k, U, H_k, w);
    old_lanczos!(H, state, k, U, H_k, w);
    @time QuantumTimeEvolution.krylovsubspace!(state, H, k, H_k, U, z)
    @time krylovevolve(state, H, 0.1, 1.0, k, H_k, U, z)
end

f();