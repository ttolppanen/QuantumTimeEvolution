# using LinearAlgebra
# using QuantumOperators
# using StaticArrays

export krylovevolve

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
    H_k, U = krylov_prealloc_Hk_U(length(state0), k)
    for _ in dt:dt:t
        krylovsubspace!(out[end], H, k, H_k, U) # returns H_k, U
        try #This is just to get a more descriptive error message
            if savelast
                out[1] .= sparse(normalize(U * exp(-1im * dt * H_k)[:, 1]))
            else
                push!(out, normalize(U * exp(-1im * dt * H_k)[:, 1]))
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

function krylovsubspace(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer)
    H_k, U = krylov_prealloc_Hk_U(length(state0), k)
    krylovsubspace!(state, H, k, H_k, U)
    return H_k, U
end

#here H_k and U are pre-allocated
function krylovsubspace!(state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, H_k::MMatrix, U::AbstractMatrix{<:Number})
    #doesnt check if HΨ = 0
    @views U[:, 1] .= normalize(Vector(state))
    @views z = H * U[:, 1]
    for i in 1:k-1
        @views a = U[:, i]' * z
        @views z .-= a * U[:, i]
        b = norm(z)
        H_k[i, i] = a
        H_k[i, i + 1] = b
        H_k[i + 1, i] = b
        @views U[:, i + 1] .= z / b
        @views z .= H * U[:, i + 1] .- b * U[:, i]
    end
    H_k[k, k] = U[:, k]' * z
end

# d : statevector dimension;
function krylov_prealloc_Hk_U(d::Integer, k::Integer)
    H_k = complex(@MMatrix zeros(k, k))
    U = complex(zeros(d, k))
    return H_k, U
end