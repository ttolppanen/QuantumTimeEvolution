# using LinearAlgebra
# using QuantumOperators
# using StaticArrays

export krylovevolve
export PA_krylov
export krylov_error_estimate

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# k : krylov subdimension;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# pa_k : PA_krylov; pre-allocated matrices/vectors needed in the algorithm
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

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

function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov(length(state0), k)
    return krylovevolve(state0, H, dt, t, k, pa_k, observables...; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, pa_k::PA_krylov, observables...; effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)
    if k < 2 throw(ArgumentError("k <= 1")) end
    apply_effect_first = !isa(effect!, Nothing) && !save_before_effect
    apply_effect_last = !isa(effect!, Nothing) && save_before_effect
    state = Vector(copy(state0))
    out = save_only_last ? zeros(length(observables), 1) : zeros(length(observables), length(0:dt:t))
    out[:, 1] .= [obs(state) for obs in observables]
    steps = length(0:dt:t)
    for i in 2:steps
        krylovsubspace!(state, H, k, pa_k) # makes changes into pa_k
        if !all(isfinite, pa_k.H_k) 
            throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) 
        end
        mul!(state, pa_k.U, @view(exp(-1im * dt * pa_k.H_k)[:, 1]))
        normalize!(state)
        if apply_effect_first effect!(state) end
        if save_only_last
            if i == steps
                out[:, 1] .= [obs(state) for obs in observables]
            end
        else
            out[:, i] .= [obs(state) for obs in observables]
        end
        if apply_effect_last effect!(state) end
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
        if beta == 0.0
            set_rest_of_Hk_to_zero(H_k, j, k)
            break;
        end
        U[:, j] .= z ./ beta
        @views mul!(z, H, U[:, j])
        @views H_k[j, j] = z' * U[:, j]
        @views z .-= (H_k[j, j] .* U[:, j] .+ beta .* U[:, j - 1])
        H_k[j - 1, j] = beta
        H_k[j, j - 1] = beta
    end
end
function krylovsubspace!(state::Vector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, pa_k::PA_krylov)
    krylovsubspace!(state, H, k, pa_k.H_k, pa_k.U, pa_k.z)
end
function krylovsubspace(state::Vector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer)
    pa_k = PA_krylov(length(state), k)
    krylovsubspace!(state, H, k, pa_k)
    return pa_k
end

function set_rest_of_Hk_to_zero(H_k, j::Integer, k::Integer)
    for j_j = 1:k
        if j_j < j
            for i_j = j:k
                H_k[i_j, j_j] = 0.0
            end
        else
            for i_j = 1:k
                H_k[i_j, j_j] = 0.0
            end
        end
    end
end

function krylov_error_estimate(dt::Real, k::Integer, state::AbstractVector{<:Number}, H::AbstractMatrix{<:Number})
    err = norm(Matrix(1.0im * dt .* H))
    err *= exp(err)
    pa_k = krylovsubspace(Vector(state), H, k)
    krylov_error = norm(Matrix(-1.0im * dt .* @view pa_k.H_k[1:k, 1:k]))
    krylov_error = (krylov_error^k * exp(krylov_error) * norm(@view pa_k.U[:, 1:k]) + err) / factorial(big(k))
    return krylov_error
end