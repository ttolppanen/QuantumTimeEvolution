using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using PlotAndSave
using LinearAlgebra

function measurement_and_feedback!(state::AbstractVector{<:Number}, msr_op, msr_prob::Real, fb_op)
    for i in 1:length(msr_op)
        if rand(Float64) < msr_prob
            measuresite!(state, msr_op, i)
            state .= fb_op[i] * state
            normalize!(state)
        end
    end
end

function f()
    d = 3; L = 7;
    dt = 0.02; t = 30.0; k = 6
    state = allone(d, L)
    H = bosehubbard(d, L)
    n = nall(d, L)
    n1 = singlesite_n(d, L, 1)
    observables = [state -> expval(state, n), state -> expval(state, n1)]

    p = 0.01
    msrop = measurementoperators(nop(d), L)
    feedback = [singlesite(n_bosons_projector(d, 0), L, i) for i in 1:L]
    effect!(state) = measurement_and_feedback!(state, msrop, p, feedback)
    
    lines = []

    r = krylovevolve(state, H, dt, t, k, observables...; effect!)
    @time r = krylovevolve(state, H, dt, t, k, observables...; effect!)
    push!(lines, LineInfo(0:dt:t, r[1, :], 1, "no_subspace, n"))
    push!(lines, LineInfo(0:dt:t, r[2, :], 1, "no_subspace, n1"))

    perm_mat, ranges = total_boson_number_subspace_tools(d, L)
    finder(state) = find_subspace(state, ranges)
    state .= perm_mat * state
    H = split_operator(H, perm_mat, ranges)
    n = split_operator(n, perm_mat, ranges)
    n1 = split_operator(n1, perm_mat, ranges)
    observables = [
        (state, id, range) -> expval(@view(state[range]), n[id]),
        (state, id, range) -> expval(@view(state[range]), n1[id])]

    msrop = measurementoperators(nop(d), L)
    for L in msrop
        for op in L
            op .= perm_mat * op * perm_mat'
        end
    end
    feedback = [perm_mat * singlesite(n_bosons_projector(d, 0), L, i) * perm_mat' for i in 1:L]
    function effect!(state, id, range)
        measurement_and_feedback!(state, msrop, p, feedback)
        id, range = finder(state)
        return state, id, range
    end

    r = krylovevolve(state, H, finder, dt, t, k, observables...; effect!)
    @time r = krylovevolve(state, H, finder, dt, t, k, observables...; effect!)
    push!(lines, LineInfo(0:dt:t, r[1, :], 1, "in_subspace, n"))
    push!(lines, LineInfo(0:dt:t, r[2, :], 1, "in_subspace, n1"))

    path = joinpath(@__DIR__, "KrylovSubspace")
    makeplot(path, lines...; xlabel = "t", ylabel = "")
end

f()
