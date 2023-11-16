using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using PlotAndSave

function f()
    d = 3; L = 8;
    dt = 0.02; t = 1.0; k = 6
    state = allone(d, L)
    n = nall(d, L)
    n1 = singlesite_n(d, L, 1)
    observables = [state -> expval(state, n), state -> expval(state, n1)]
    lines = []

    r = krylovevolve(state, H, dt, t, k, observables...)
    @time r = krylovevolve(state, H, dt, t, k, observables...)
    # push!(lines, LineInfo(0:dt:t, r[1, :], 1, "no_subspace, n"))
    # push!(lines, LineInfo(0:dt:t, r[2, :], 1, "no_subspace, n1"))

    perm_mat, ranges = total_boson_number_subspace_tools(d, L)
    state .= perm_mat * state
    H = split_operator(H, perm_mat, ranges)
    n = split_operator(n, perm_mat, ranges)
    n1 = split_operator(n1, perm_mat, ranges)
    observables = [
        (state, id, range) -> expval(@view(state[range]), n[id]),
        (state, id, range) -> expval(@view(state[range]), n1[id])]

    finder(state) = find_subspace(state, ranges)
    r = krylovevolve(state, H, finder, dt, t, k, observables...)
    @time r = krylovevolve(state, H, finder, dt, t, k, observables...)
    # push!(lines, LineInfo(0:dt:t, r[1, :], 1, "in_subspace, n"))
    # push!(lines, LineInfo(0:dt:t, r[2, :], 1, "in_subspace, n1"))

    # path = joinpath(@__DIR__, "KrylovSubspace")
    # makeplot(path, lines...; xlabel = "t", ylabel = "")
end

@profview f()
