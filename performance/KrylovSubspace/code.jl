using QuantumStates
using QuantumOperators
using QuantumTimeEvolution
using PlotAndSave
using LinearAlgebra
using Profile
using Random
using JET
using FastClosures

function measurement_and_feedback!(state::AbstractVector{<:Number}, msr_op, msr_prob::Real, fb_op)
    for i in 1:length(msr_op)
        if rand(Float64) < msr_prob
            measuresite!(state, msr_op, i)
            state .= fb_op[i] * state
            normalize!(state)
        end
    end
    return state
end

function traj_mean(result)
    out = zeros(size(result[1]))
    for traj in result
        out .+= traj
    end
    return out ./ length(result)
end

# function measurement_and_feedback!(state, msr_op, msr_prob::Real, fb_op, id)
#     id_after_measurement = id
#     for L in 1:length(msr_op[id])
#         if rand(Float64) < msr_prob
#             i = measuresite!(state[id], msr_op[id], L)
#             new_id, op = fb_op[id][L][i]
#             if new_id == id
#                 state[new_id] .= op * state[id]
#             else
#                 mul!(state[new_id], op, state[id])
#             end
#             normalize!(state[new_id])
#             id_after_measurement = new_id
#         end
#     end
#     return id_after_measurement
# end

function take_krylov_time_step_subspace_function(H, k, dt, pa_k)
    function take_time_step!(state, id)
        krylovsubspace!(state[id], H[id], k, pa_k.H_k, pa_k.U[id], pa_k.z[id])
        if !all(isfinite, pa_k.H_k)
            throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) 
        end
        @views mul!(state[id], pa_k.U[id], (exp(-1im * dt * pa_k.H_k)[:, 1]))
        normalize!(state[id])
        return state, id
    end
    return take_time_step!
end

function f()
    d = 2; L = 16;
    dt = 0.02; t = 30; k = 6
    rng_seed = 2
    state = zeroone(d, L)
    H = bosehubbard(d, L)
    n = nall(d, L)
    n1 = singlesite_n(d, L, 1)
    observables = [state -> expval(state, n), state -> expval(state, n1)]

    p = 0.1
    msrop = measurementoperators(nop(d), L)
    feedback = [singlesite(n_bosons_projector(d, 0), L, i) for i in 1:L]
    effect!(state) = measurement_and_feedback!(state, msrop, p, feedback)
    
    lines = []

    # r = krylovevolve(state, H, dt, t, k, observables...; effect!)
    Random.seed!(rng_seed)
    # @time r = krylovevolve(state, H, dt, t, k, observables...; effect!)
    # push!(lines, LineInfo(0:dt:t, r[1, :], 1, "no_subspace, n"))
    # push!(lines, LineInfo(0:dt:t, r[2, :], 1, "no_subspace, n1"))


    # in subspace
    indices = total_boson_number_subspace_indices(d, L)
    ranges, perm_mat = total_boson_number_subspace_tools(d, L)
    initial_id = find_subspace(state, indices)

    state = subspace_split(state, ranges, perm_mat)
    H = subspace_split(H, ranges, perm_mat)
    n = subspace_split(n, ranges, perm_mat)
    n1 = subspace_split(n1, ranges, perm_mat)
    observables = [
        @closure((state, id) -> expval(state[id], n[id])),
        @closure((state, id) -> expval(state[id], n1[id]))]

    msrop = measurementoperators(nop(d), L)
    msrop = measurement_subspace(msrop, ranges, perm_mat)
    feedback = [singlesite(n_bosons_projector(d, 0), L, i) for i in 1:L]
    feedback = feedback_measurement_subspace(feedback, msrop, indices; digit_error = 10, id_relative_guess = -1)

    effect! = @closure((state, id) -> random_measurement_feedback!(state, id, msrop, p, feedback; skip_subspaces = 1))

    # krylovevolve(state, initial_id, H, dt, t, k, observables...; effect!)
    # @time krylovevolve(state, initial_id, H, dt, t, k, observables...; effect!)
    Random.seed!(rng_seed)

    pa_k = PA_krylov_sub(state, k)
    traj = 1000
    pa_args, pa_out = pre_alloc_threads(zeros(2, length(0:dt:t)), traj, pa_k)

    r_f = @closure((out, pa_k) -> krylovevolve(state, initial_id, H, dt, t, k, pa_k, observables...; out, effect!))
    solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)
    @profview solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)

    r = traj_mean(pa_out)
    push!(lines, LineInfo(0:dt:t, r[1, :], 1, "in_subspace, n"))
    push!(lines, LineInfo(0:dt:t, r[2, :], 1, "in_subspace, n1"))

    path = joinpath(@__DIR__, "results")
    makeplot(path, lines...; xlabel = "t", ylabel = "")
end

f();
