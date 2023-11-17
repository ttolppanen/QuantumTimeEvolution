# using LinearAlgebra
# using QuantumOperators
# using StaticArrays

# The functions here are already exported in Krylov.jl
# export krylovevolve
export PA_krylov_sub

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
struct PA_krylov_sub{T}
    H_k::T
    U::Vector{Matrix{ComplexF64}}
    z::Vector{ComplexF64}
    function PA_krylov_sub(d::Integer, k::Integer, H)
        H_k = complex(zeros(MMatrix{k, k}))
        U = []
        for H_i in H
            d_i = size(H_i)[1]
            push!(U, complex(zeros(d_i, k)))
        end
        z = complex(zeros(d))
        new{typeof(H_k)}(H_k, U, z)
    end
end

function krylovevolve(state0::AbstractVector{<:Number}, H, find_subspace::Function, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov_sub(length(state0), k, H)
    return krylovevolve(state0, H, find_subspace, dt, t, k, pa_k, observables...; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H, find_subspace::Function, dt::Real, t::Real, k::Integer, pa_k::PA_krylov_sub, observables...;
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)
    
    if k < 2 throw(ArgumentError("k <= 1")) end
    steps = length(0:dt:t)
    function initialize(state0) 
        id, indices = find_subspace(state0)
        return Vector(copy(state0)), id, indices
    end
    time_step_funcs = [] # functions to run in a single timestep

    function take_time_step(state, id, indices)
        dim = length(indices)
        @views krylovsubspace!(state[indices], H[id], k, pa_k.H_k, pa_k.U[id], pa_k.z[1:dim])
        if !all(isfinite, pa_k.H_k)
            throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) 
        end
        @views mul!(state[indices], pa_k.U[id], (exp(-1im * dt * pa_k.H_k)[:, 1]))
        normalize!(@view(state[indices]))
        return state, id, indices
    end
    push!(time_step_funcs, take_time_step)

    if !isa(effect!, Nothing)
        if save_before_effect
            push!(time_step_funcs, :calc_obs)
            push!(time_step_funcs, effect!)
        else
            push!(time_step_funcs, effect!)
            push!(time_step_funcs, :calc_obs)
        end
    else # no effect
        push!(time_step_funcs, :calc_obs)
    end
    return timeevolve!(state0, initialize, time_step_funcs, steps, observables...; save_only_last)
end

function krylov_time_step!(state, H, k, pa_k, dt)
    krylovsubspace!(state, H, k, pa_k) # makes changes into pa_k
    if !all(isfinite, pa_k.H_k) 
        throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) 
    end
    mul!(state, pa_k.U, @view(exp(-1im * dt * pa_k.H_k)[:, 1]))
    normalize!(state)
end