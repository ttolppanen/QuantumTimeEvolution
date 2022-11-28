# using LinearAlgebra
# using QuantumOperators
# using StaticArrays

export krylovevolve

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# pa_k : PA_krylov; pre-allocated matrices/vectors needed in the algorithm
# effect! : function with one argument, the state; something to do to the state after each timestep
# savelast : set true if you only need the last value of the time-evolution

# d : statevector dimension;
struct PA_krylov{T}
    H_k::T
    U::Matrix{ComplexF64}
    z::Vector{ComplexF64}
    function PA_krylov(d::Integer, k::Integer)
        H_k = complex(zeros(MMatrix{k, k}))
        U = complex(zeros(d, k))
        z = complex(zeros(d))
        new{typeof(H_k)}(H_k, U, z)
    end
end

function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer; kwargs...)
    pa_k = PA_krylov(length(state0), k)
    return krylovevolve(state0, H, dt, t, k, pa_k; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, pa_k::PA_krylov; kwargs...)
    return krylovevolve(state0, H, dt, t, k, pa_k.H_k, pa_k.U, pa_k.z; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, H_k::MMatrix, U::Matrix{ComplexF64}, z::Vector{ComplexF64}; effect! = nothing, savelast::Bool = false)
    if k < 2
        throw(ArgumentError("k <= 1"))
    end
    out = [deepcopy(state0)]
    for _ in dt:dt:t
        krylovsubspace!(out[end], H, k, H_k, U, z) # makes changes in to pa_k
        try # This is just to get a more descriptive error message
            if savelast
                out[1] .= sparse(normalize(U * exp(-1im * dt * H_k)[:, 1]))
            else
                push!(out, normalize(U * exp(-1im * dt * H_k)[:, 1]))
            end
        catch error
            if isa(error, ArgumentError)
                throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too largle or there is no time evolution H * state0 = 0."))
            else
                throw(error)
            end
        end
        !isa(effect!, Nothing) ? effect!(out[end]) : nothing
    end
    return out
end

# here H_k, U and z are pre-allocated
function krylovsubspace!(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, H_k::MMatrix, U::AbstractMatrix{<:Number}, z::AbstractVector{<:Number})
    # doesnt check if HΨ = 0
    U[:, 1] .= Vector(state) # Here should be normalization, but the state should always be normalized?
    @views mul!(z, H, U[:, 1])
    for i in 1:k-1
        @views a = U[:, i]' * z
        @views z .-= a .* U[:, i]
        b = norm(z)
        H_k[i, i] = a
        H_k[i, i + 1] = b
        H_k[i + 1, i] = b
        @views U[:, i + 1] .= z ./ b
        @views mul!(z, H, U[:, i + 1])
        @views z .-= b .* U[:, i]
    end
    @views H_k[k, k] = U[:, k]' * z
end
function krylovsubspace!(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, pa_k::PA_krylov)
    krylovsubspace!(state, H, k, pa_k.H_k, pa_k.U, pa_k.z)
end
function krylovsubspace(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer)
    pa_k = PA_krylov(length(state0), k)
    krylovsubspace!(state, H, k, pa_k)
    return pa_k
end