using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using PlotAndSave

function f()
    d = 3; L = 7;
    dt = 0.02; t = 1.0; k = 6
    state = allone(d, L)
    H = bosehubbard(d, L)
    n = nall(d, L)
    n1 = singlesite_n(d, L, 1)
    observables = [state -> expval(state, n), state -> expval(state, n1)]
    lines = []

    @profview r = krylovevolve(state, H, dt, t, k, observables...)
    # push!(lines, LineInfo(0:dt:t, r[1, :], 1, "no_subspace, n"))
    # push!(lines, LineInfo(0:dt:t, r[2, :], 1, "no_subspace, n1"))

    indices, perm_mat, ranges = total_boson_number_subspace(d, L)
    state = perm_mat * state
    H = perm_mat * H * perm_mat'
    n = perm_mat * n * perm_mat'
    n1 = perm_mat * n1 * perm_mat'
    observables = [(state, indices) -> expval(state, n, indices), (state, indices) -> expval(state, n1, indices)]
    finder(state) = find_subspace(state, ranges)
    @profview r = krylovevolve(state, H, dt, t, k, observables...; find_subspace = finder)
    # push!(lines, LineInfo(0:dt:t, r[1, :], 1, "in_subspace, n"))
    # push!(lines, LineInfo(0:dt:t, r[2, :], 1, "in_subspace, n1"))

    # path = joinpath(@__DIR__, "KrylovSubspace")
    # makeplot(path, lines...; xlabel = "t", ylabel = "")
end

f()
