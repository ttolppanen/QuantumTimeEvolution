# using ITensors
# include("Krylov.jl")
# include("MPS.jl")
# include("Trajectories.jl")

export mipt

function mipt(mps0::MPS, gates, meffect!, dt, t, prob, traj, calc_res...; kwargs...) # kwargs for mpsevolve and solvetraj
    mpsevolve_kwargs, solvetraj_kwargs = splitkwargs(kwargs, [:cutoff, :maxdim, :mindim], [:paral])
    num_obs = length(calc_res)
    traj_f(p) = solvetrajectories(() -> mpsevolve(mps0, gates, dt, t, calc_res...; effect! = state -> meffect!(state, p), mpsevolve_kwargs...), traj; solvetraj_kwargs...)
    return mipt_abstract(traj_f, prob, num_obs, traj)
end
function mipt(state0::AbstractVector, H, k::Integer, meffect!, dt, t, prob, traj, calc_res...; kwargs...) # kwargs for solvetraj
    num_obs = length(calc_res)
    traj_f(p) = solvetrajectories(() -> krylovevolve(state0, H, dt, t, k, calc_res...; effect! = state -> meffect!(state, p)), traj; kwargs...)
    return mipt_abstract(traj_f, prob, num_obs, traj)
end
function mipt_abstract(traj_f, prob, num_obs, traj)
    out = zeros(num_obs, length(prob))
    for p_i in eachindex(prob)
        r_traj = traj_f(prob[p_i])
        last_val = [traj[:, end] for traj in r_traj]
        out[:, p_i] = sum(last_val) ./ traj
    end
    if num_obs == 1
        return vec(out)
    end
    return out
end