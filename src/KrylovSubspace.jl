# using LinearAlgebra
# using QuantumOperators
# using StaticArrays
# using ExponentialUtilities

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
struct PA_krylov_sub
    ks::Vector{KrylovSubspace}
    cache::ExpvCache
    work_vector::Vector{Vector{ComplexF64}}
    function PA_krylov_sub(state, k)
        ks = []
        for subdim_vec in state
            push!(ks, KrylovSubspace{ComplexF64, Float64}(size(subdim_vec, 1), k))
        end
        cache = ExpvCache{Float64}(k)
        work_vector = Vector.(deepcopy(state))
        new(ks, cache, work_vector)
    end
end

function krylovevolve(state0, initial_subspace_id, H, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov_sub(state0, k)
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

    time_step_funcs = [] # functions to run in a single timestep
    take_time_step! = take_krylov_time_step_subspace_function(H, -1.0im * dt, pa_k)
    push!(time_step_funcs, take_time_step!)
    push!(time_step_funcs, :calc_obs) # calculating observables is told with a keyword :calc_obs

    if !isa(effect!, Nothing)
        if save_before_effect
            push!(time_step_funcs, effect!)
        else
            insert!(time_step_funcs, 2, effect!)
        end
    end
    if isa(out, Nothing)
        return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
    end
    return timeevolve!(initial_args, time_step_funcs, steps, out, observables...; save_only_last) 
end

function take_krylov_time_step_subspace_function(H::Vector{SparseMatrixCSC{ComplexF64, Int64}}, dt::ComplexF64, pa_k::PA_krylov_sub)
    function take_time_step!(state::Vector{Vector{ComplexF64}}, id::Int64)
        lanczos!(pa_k.ks[id], H[id], state[id])
        expv!(state[id], dt, pa_k.ks[id]; pa_k.cache)
        normalize!(state[id])
        return state::Vector{Vector{ComplexF64}}, id::Int64
    end
    return take_time_step!
end