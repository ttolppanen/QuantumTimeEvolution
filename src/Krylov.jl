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
    out = kwargs[:savelast] ? [complex(zeros(size(state0)))] : [complex(zeros(size(state0))) for _ in 0:dt:t]
    return krylovevolve(state0, H, dt, t, k, out; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, out::Vector{<:AbstractVector{<:Number}}; kwargs...)
    pa_k = PA_krylov(length(state0), k)
    return krylovevolve(state0, H, dt, t, k, pa_k, out; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, pa_k::PA_krylov, out::Vector{<:AbstractVector{<:Number}}; effect! = nothing, savelast::Bool = false)
    if k < 2 throw(ArgumentError("k <= 1")) end
    apply_effect = !isa(effect!, Nothing)
    out[1] .= deepcopy(Vector(state0))
    for i in 2:length(dt:dt:t)
        state = savelast ? out[1] : out[i]
        krylovsubspace!(state, H, k, pa_k) # makes changes into pa_k
        if !all(isfinite, pa_k.H_k) throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) end
        mul!(state, pa_k.U, @view(exp(-1im * dt * pa_k.H_k)[:, 1]))
        normalize!(state)
        if apply_effect effect!(state) end
    end
    return out
end

# here H_k, U and z are pre-allocated
function krylovsubspace!(state::Vector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, H_k::MMatrix, U::AbstractMatrix{<:Number}, z::AbstractVector{<:Number})
    # doesnt check if HΨ = 0
    U[:, 1] .= state
    mul!(z, H, state)
    H_k[1, 1] = z' * state
    z .-= H_k[1, 1] .* state
    for j in 2:k
        beta = norm(z)
        U[:, j] .= z ./ beta
        @views mul!(z, H, U[:, j])
        @views H_k[j, j] = z' * U[:, j]
        @views z .-= (H_k[j, j] .* U[:, j] .+ beta .* U[:, j - 1])
        H_k[j - 1, j] = beta
        H_k[j, j - 1] = beta
    end
end
function krylovsubspace!(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, pa_k::PA_krylov)
    krylovsubspace!(state, H, k, pa_k.H_k, pa_k.U, pa_k.z)
end
function krylovsubspace(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer)
    pa_k = PA_krylov(length(state0), k)
    krylovsubspace!(state, H, k, pa_k)
    return pa_k
end