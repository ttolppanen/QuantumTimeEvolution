using Distributed
using LinearAlgebra
@everywhere using QuantumStates
@everywhere using QuantumOperators
@everywhere using QuantumTimeEvolution
using PlotAndSave
using OrderedCollections
using FastClosures

function traj_mean(result)
    out = zeros(size(result[1]))
    for traj in result
        out .+= traj
    end
    return out ./ length(result)
end

function initial_parameters()
    param = OrderedDict()
    param[:dt] = 0.02
    param[:t] = 30
    param[:d] = 3
    param[:L] = 10
    param[:state] = ValueInfo(productstate(param[:d], [isodd(i) ? 2 : 0 for i in 1:param[:L]]), "|2020...>")
    param[:alg] = "krylov BN-subspace"
    param[:k] = 6
    param[:H] = "BH"
    param[:w] = 0
    param[:J] = 1

    d = param[:d]; L = param[:L]
    msrop = measurementoperators(nop(d), L)
    feedback = [1 1 0; 0 0 0; 0 0 1]
    fb_op = [singlesite(feedback, L, i) for i in 1:L]
    
    param[:effect] = ValueInfo((msr = msrop, fb = fb_op), "msr_n -> fb: 0 -> |0>, 1 -> |0>, 2 -> |2>")
    return param
end

function f()
    param = initial_parameters()

    # -----------
    traj = 100
    ps = 0.01:0.01:0.1
    Us = [7, 9, 10, 11, 13]
    # -----------

    d = param[:d]; L = param[:L];
    indices = total_boson_number_subspace_indices(d, L)
    ranges, perm_mat = total_boson_number_subspace_tools(d, L)
    initial_id = find_subspace(param[:state].val, indices)
    state = subspace_split(param[:state].val, ranges, perm_mat)

    msrop = measurement_subspace(param[:effect].val.msr, ranges, perm_mat)
    feedback = feedback_measurement_subspace(param[:effect].val.fb, msrop, indices; digit_error = 10, id_relative_guess = -1)

    n = subspace_split(nall(d, L) ./ L, ranges, perm_mat)
    obs = @closure((state, id) -> expval(state[id], n[id]))

    H = subspace_split(bosehubbard(d, L; w = param[:w], J = param[:J], U = Us[1]), ranges, perm_mat)

    dt = param[:dt]; t = param[:t];
    pa_k = PA_krylov_sub(param[:k], H)
    pa_args, pa_out = pre_alloc_threads(zeros(1, length(0:dt:t)), traj, pa_k)

    @time for U in Us
        H = subspace_split(bosehubbard(d, L; w = param[:w], J = param[:J], U), ranges, perm_mat)
        plot_lines = []
        @time for p in ps
            effect! = @closure((state, id) -> random_measurement_feedback!(state, id, msrop, p, feedback; skip_subspaces = 1))
            
            dt = param[:dt]; t = param[:t]; k = param[:k]
            r_f = ((out, pa_k) -> krylovevolve(state, initial_id, H, dt, t, k, pa_k, obs; effect!, out))
            solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)
            r = traj_mean(pa_out)
            push!(plot_lines, LineInfo(0:dt:t, r[1, :], traj, "p = $p"))
        end
    end
end

@profview f()