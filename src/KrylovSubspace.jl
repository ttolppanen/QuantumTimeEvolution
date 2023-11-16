# using LinearAlgebra
# using QuantumOperators
# using StaticArrays

# The functions here are already exported in Krylov.jl
# export krylovevolve

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

function krylovevolve(state0::AbstractVector{<:Number}, H, find_subspace::Function, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov(length(state0), k)
    return krylovevolve(state0, H, find_subspace, dt, t, k, pa_k, observables...; kwargs...)
end
function krylovevolve(state0::AbstractVector{<:Number}, H, find_subspace::Function, dt::Real, t::Real, k::Integer, pa_k::PA_krylov, observables...;
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false)
    
    if k < 2 throw(ArgumentError("k <= 1")) end
    steps = length(0:dt:t)
    function initialize(state0) 
        id, indices = find_subspace(state0)
        return Vector(copy(state0)), id, indices
    end
    time_step_funcs = [] # functions to run in a single timestep

    function current_subspace(state, id, indices)
        new_id, new_indices = find_subspace(state)
        return state, new_id, new_indices
    end
    push!(time_step_funcs, current_subspace)

    function take_time_step(state, id, indices)
        dim = length(indices)
        @views krylovsubspace!(state[indices], H[id], k, pa_k.H_k, pa_k.U[1:dim, :], pa_k.z[1:dim])
        if !all(isfinite, pa_k.H_k)
            throw(ArgumentError("Hₖ contains Infs or NaNs. This is is usually because k is too small, too large or there is no time evolution H * state0 = 0.")) 
        end
        @views mul!(state[indices], pa_k.U[1:dim, :], (exp(-1im * dt * pa_k.H_k)[:, 1]))
        normalize!(state)
        return state, id, indices
    end
    push!(time_step_funcs, take_time_step)

    if !isa(effect!, Nothing)
        function do_effect(state)
            effect!(state)
            return (state, )
        end
        if save_before_effect
            push!(time_step_funcs, :calc_obs)
            push!(time_step_funcs, do_effect)
        else
            push!(time_step_funcs, do_effect)
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