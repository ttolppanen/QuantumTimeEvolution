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
    z::Vector{Vector{ComplexF64}}
    work_vector::Vector{Vector{ComplexF64}}
    function PA_krylov_sub(k::Integer, H)
        H_k = complex(zeros(MMatrix{k, k}))
        U = []
        z = []
        for H_i in H
            d_i = size(H_i)[1]
            push!(U, complex(zeros(d_i, k)))
            push!(z, complex(zeros(d_i)))
        end
        new{typeof(H_k)}(H_k, U, z, deepcopy(z))
    end
end

function krylovevolve(state0, initial_subspace_id, H, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov_sub(k, H)
    return krylovevolve(state0, initial_subspace_id, H, dt, t, k, pa_k, observables...; kwargs...)
end
function krylovevolve(state0, initial_subspace_id::Integer, H, dt::Real, t::Real, k::Integer, pa_k::PA_krylov_sub, observables...;
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, out = nothing)
    
    if k < 2 throw(ArgumentError("k <= 1")) end
    steps = length(0:dt:t)
    for i in eachindex(state0)
        pa_k.work_vector[i] .= state0[i]
    end
    initial_args = (pa_k.work_vector, initial_subspace_id)

    take_time_step! = take_krylov_time_step_subspace_function(H, k, dt, pa_k)
    
    if !isa(effect!, Nothing)
        if save_before_effect
            time_step_funcs = (
                take_time_step!,
                :calc_obs,
                effect!
            )
        else
            time_step_funcs = (
                take_time_step!,
                effect!,
                :calc_obs
            )
        end
    else
        time_step_funcs = (
            take_time_step!,
            :calc_obs
        )
    end
    if isa(out, Nothing)
        return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
    end
    return timeevolve!(initial_args, time_step_funcs, steps, out, observables...; save_only_last) 
end

function take_krylov_time_step_subspace_function(H, k, dt, pa_k)
    function take_time_step!(state, id)
        krylovsubspace!(state[id], H[id], k, pa_k.H_k, pa_k.U[id], pa_k.z[id])
        if !all(isfinite, pa_k.H_k)
            throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) 
        end
        @views mul!(state[id], pa_k.U[id], (exp(-1im * dt * pa_k.H_k)[:, 1]))
        normalize!(state[id])
        return state, id
    end
    return take_time_step!
end