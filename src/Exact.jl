using SparseArrays
using QuantumOperators

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;

export exactevolve
export exactevolve_bosehubbard

function exactevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, t::Real, dt::Real)
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

function exactevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector{<:Number}, dt::Real, t::Real; w = 1, U = 1, J = 1)
    H = bosehubbard(d, L; w = w, U = U, J = J)
    return exactevolve(state0, H, t, dt)
end