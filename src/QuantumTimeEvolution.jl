module QuantumTimeEvolution

using LinearAlgebra
using SparseArrays
using QuantumOperators
using QuantumStates

export exactevolve
export exactevolve_bosehubbard
export krylovevolve
export krylovevolve_bosehubbard

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;

function exactevolve(state0::AbstractVector, H::AbstractMatrix, t::Real, dt::Real)
    if issparse(H)
        M = exp(-im * dt * Matrix(H))
    else
        M = exp(-im * dt * H)
    end
    out = [deepcopy(state0)]
    for _ in dt:dt:t
        push!(out, M * out[end])
    end
    return out
end

function exactevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector, dt::Real, t::Real; w = 1, U = 1, J = 1)
    H = bosehubbard(d, L; w = w, U = U, J = J)
    return exactevolve(state0, H, t, dt)
end

function krylovevolve(state0::AbstractVector, H::AbstractMatrix, t::Real, dt::Real, k::Integer)
    out = [deepcopy(state0)]
    for i in dt:dt:t
        Hₖ, U = krylovsubspace(out[end], H, k)
        try
            push!(out, normalize(U * exp(-1im * dt * Hₖ)[:, 1]))
        catch error
            throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, or there is no time evolution H * state0 = 0."))
        end
    end
    return out
end

function krylovevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector, dt::Real, t::Real, k::Integer; w = 1, U = 1, J = 1)
    H = bosehubbard(d, L; w = w, U = U, J = J)
    return krylovevolve(state0, H, t, dt, k)
end

function krylovsubspace(state::AbstractVector, H::AbstractMatrix, k::Integer)
    #doesnt check if HΨ = 0
    Hₖ = complex(zeros(k, k))
    U = complex(zeros(length(state), k))

    U[:, 1] = normalize(state)
    z = H * U[:, 1]
    for i in 1:k-1
        a = U[:, i]' * z
        z .-= a * U[:, i]
        b = norm(z)
        Hₖ[i, i] = a
        Hₖ[i, i + 1] = b
        Hₖ[i + 1, i] = b
        U[:, i + 1] .= z / b
        z = H * U[:, i + 1] .- b * U[:, i]
    end
    Hₖ[k, k] = U[:, k]' * z
    return Hₖ, U
end

#NOT USED
#=
function seriesEvolve(Ψ, H, t, dt, k)
    result = [deepcopy(Ψ)]
    M = -im * dt * H
    for _ in dt:dt:t
        Ψtemp = result[end]
        Ψ .= Ψtemp
        for i in 1:k-1
            Ψtemp = M * Ψtemp
            Ψ .+= Ψtemp ./ factorial(i)
        end
        push!(result, deepcopy(Ψ))
    end
    return result
end
=#

end # module
