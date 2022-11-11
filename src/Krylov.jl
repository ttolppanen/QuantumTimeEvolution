using LinearAlgebra
using QuantumOperators

export krylovevolve
export krylovevolve_bosehubbard

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;

function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer)
    if k < 2
        throw(ArgumentError("k <= 1"))
    end
    out = [deepcopy(state0)]
    for i in dt:dt:t
        Hₖ, U = krylovsubspace(out[end], H, k)
        try #This is just to get a more descriptive error message
            push!(out, normalize(U * exp(-1im * dt * Hₖ)[:, 1]))
        catch error
            if isa(error, ArgumentError)
                throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, or there is no time evolution H * state0 = 0."))
            end
        end
    end
    return out
end

function krylovevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector{<:Number}, dt::Real, t::Real, k::Integer; kwargs...) #key value arguments for bosehubbard
    H = bosehubbard(d, L; kwargs...) #kwargs can be {w, U, J}
    return krylovevolve(state0, H, dt, t, k)
end

function krylovsubspace(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer)
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