# using LinearAlgebra
# using QuantumOperators

export krylovevolve
export krylovevolve_bosehubbard

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# effect! : function with one argument, the state; something to do to the state after each timestep
# savelast : set true if you only need the last value of the time-evolution

function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer; effect! = nothing, savelast::Bool = false)
    if k < 2
        throw(ArgumentError("k <= 1"))
    end
    out = [deepcopy(state0)]
    for _ in dt:dt:t
        Hₖ, U = krylovsubspace(out[end], H, k)
        try #This is just to get a more descriptive error message
            if savelast
                out[1] .= normalize(U * exp(-1im * dt * Hₖ)[:, 1])
            else
                push!(out, normalize(U * exp(-1im * dt * Hₖ)[:, 1]))
            end
        catch error
            if isa(error, ArgumentError)
                throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, or there is no time evolution H * state0 = 0."))
            else
                throw(error)
            end
        end
        !isa(effect!, Nothing) ? effect!(out[end]) : nothing
    end
    return out
end

function krylovevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector{<:Number}, dt::Real, t::Real, k::Integer; kwargs...) #keyword arguments for bosehubbard, krylovevolve
    bhkwargs, kekwargs = splitkwargs(kwargs, [:w, :U, :J], [:effect!, :savelast]) #bhkwargs ∈  {w, U, J}, kekwargs ∈ {effect!, savelast}
    H = bosehubbard(d, L; bhkwargs...)
    return krylovevolve(state0, H, dt, t, k; kekwargs...)
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