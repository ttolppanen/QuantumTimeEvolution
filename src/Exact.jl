using SparseArrays
using QuantumOperators

export exactevolve
export exactevolve_bosehubbard

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;

function exactevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real)
    if issparse(H)
        U = exp(-im * dt * Matrix(H))
    else
        U = exp(-im * dt * H)
    end
    out = [deepcopy(state0)]
    for _ in dt:dt:t
        push!(out, U * out[end])
    end
    return out
end

function exactevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector{<:Number}, dt::Real, t::Real; kwargs...)#keyword arguments for bosehubbard
    H = bosehubbard(d, L; kwargs...) #kwargs can be {w, U, J}
    return exactevolve(state0, H, dt, t)
end
