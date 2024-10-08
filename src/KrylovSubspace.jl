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
# krylov_alg : the algorith used to calcutate the krylov subspace; defaults to :lancoz, use :arnoldi for non-hermitian Hamiltonians. 

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
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, out = nothing, krylov_alg = :lancoz)
    
    if k < 2 throw(ArgumentError("k <= 1")) end
    steps = length(0:dt:t)
    for i in eachindex(state0)
        pa_k.work_vector[i] .= state0[i]
    end
    initial_args = (pa_k.work_vector, initial_subspace_id)

    take_time_step! = take_krylov_time_step_subspace_function(H, -1.0im * dt, pa_k; alg = krylov_alg)
    time_step_funcs = make_time_step_list(take_time_step!, effect!, save_before_effect) # defined in TimeEvolve.jl

    if isa(out, Nothing)
        return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
    end
    return timeevolve!(initial_args, time_step_funcs, steps, out, observables...; save_only_last) 
end

function take_krylov_time_step_subspace_function(H::Vector{SparseMatrixCSC{ComplexF64, Int64}}, dt::ComplexF64, pa_k::PA_krylov_sub; kwargs...)
    return take_krylov_time_step_subspace_function(H, dt, pa_k.ks, pa_k.cache; alg = :lancoz)
end
function take_krylov_time_step_subspace_function(H::Vector{SparseMatrixCSC{ComplexF64, Int64}}, dt::ComplexF64, ks::Vector{KrylovSubspace}, ks_cache::ExpvCache; alg = :lancoz)
    take_time_step! = @closure((state, id) -> begin
        if alg == :lancoz
            lanczos!(ks[id], H[id], state[id])
        elseif alg == :arnoldi
            arnoldi!(ks[id], H[id], state[id])
        end
        expv!(state[id], dt, ks[id]; cache = ks_cache)
        # normalize!(state[id])
        return state, id
    end)
    return take_time_step!
end