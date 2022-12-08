# using ITensors
# include("Krylov.jl")
# include("MPS.jl")
# include("Trajectories.jl")

export mipt

function mipt(mps0::MPS, gates, meffect!, dt, t, prob, traj, calc_res...)
    traj_f(p) = solvetrajectories(() -> mpsevolve(mps0, gates, dt, t; effect! = state -> meffect!(state, p), savelast = true), traj)
    return mipt_abstract(traj_f, prob, calc_res...)
end
function mipt(state0::AbstractVector, H, k::Integer, meffect!, dt, t, prob, traj, calc_res...)
    traj_f(p) = solvetrajectories(() -> krylovevolve(state0, H, dt, t, k; effect! = state -> meffect!(state, p), savelast = true), traj)
    return mipt_abstract(traj_f, prob, calc_res...)
end
function mipt_abstract(traj_f, prob, calc_res...)
    res = [[] for _ in calc_res]
    for p in prob
        r_traj = traj_f(p)
        for i in eachindex(calc_res)
            push!(res[i], calc_res[i](r_traj)[1])
        end
    end
    if length(calc_res) == 1
        return res[1]
    end
    return res
end