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
struct PA_krylov
    ks::KrylovSubspace
    cache::ExpvCache
    work_vector::Vector{ComplexF64}
    function PA_krylov(state, k)
        ks = KrylovSubspace{ComplexF64, Float64}(size(state, 1), k)
        cache = ExpvCache{Float64}(k)
        work_vector = Vector(deepcopy(state))
        new(ks, cache, work_vector)
    end
end

function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov(state0, k)
    return krylovevolve(state0, H, dt, t, k, pa_k, observables...; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real, k::Integer, pa_k::PA_krylov, observables...;
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)
    
    if k < 2 throw(ArgumentError("k <= 1")) end
    steps = length(0:dt:t)
    pa_k.work_vector .= state0
    initial_args = pa_k.work_vector
    time_step_funcs = [] # functions to run in a single timestep

    take_time_step! = take_krylov_time_step_function(H, dt, pa_k)
    push!(time_step_funcs, take_time_step!)
    push!(time_step_funcs, :calc_obs)

    if !isa(effect!, Nothing)
        if save_before_effect
            push!(time_step_funcs, effect!)
        else
            insert!(time_step_funcs, 2, effect!)
        end
    end
        
    return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
end

function take_krylov_time_step_function(H::AbstractMatrix{<:Number}, dt::Real, pa_k::PA_krylov)
    function take_time_step!(state)
        lanczos!(pa_k.ks, H, state)
        expv!(state, dt, pa_k.ks; pa_k.cache)
        normalize!(state)
        return state
    end
    return take_time_step!
end

# here H_k, U and z are pre-allocated
function krylovsubspace!(state::AbstractArray{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, H_k::MMatrix, U::AbstractMatrix{<:Number}, z::AbstractVector{<:Number})
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
function krylovsubspace!(state::AbstractArray{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, pa_k::PA_krylov)
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