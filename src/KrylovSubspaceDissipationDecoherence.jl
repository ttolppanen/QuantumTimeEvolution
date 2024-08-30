# using LinearAlgebra
# using QuantumOperators
# using StaticArrays

export PA_krylov_sub_dd

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
struct PA_krylov_sub_dd
    ks::Vector{KrylovSubspace}
    cache::ExpvCache
    work_vector::Vector{Vector{ComplexF64}}
    prev_work_vector::Vector{Vector{ComplexF64}}
    function PA_krylov_sub_dd(state, k)
        ks = []
        for subdim_vec in state
            push!(ks, KrylovSubspace{ComplexF64, Float64}(size(subdim_vec, 1), k))
        end
        cache = ExpvCache{Float64}(k)
        work_vector = Vector.(deepcopy(state))
        prev_work_vector = Vector.(deepcopy(state))
        new(ks, cache, work_vector, prev_work_vector)
    end
end

function krylovevolve(state0, initial_subspace_id::Integer, H, diss_op, deco_op, dt::Real, t::Real, k::Integer, observables...; kwargs...)
    pa_k = PA_krylov_sub_dd(state0, k)
    return krylovevolve(state0, initial_subspace_id, H, diss_op, deco_op, dt, t, k, pa_k, observables...; kwargs...)
end
function krylovevolve(state0, initial_subspace_id::Integer, H, diss_op, deco_op,
                      dt::Real, t::Real, k::Integer, pa_k::PA_krylov_sub_dd, observables...;
                      effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, out = nothing)
    
    if k < 2 throw(ArgumentError("k <= 1")) end
    steps = length(0:dt:t)
    for i in eachindex(state0)
        pa_k.work_vector[i] .= state0[i]
    end
    initial_args = (pa_k.work_vector, initial_subspace_id)

    take_time_step! = take_krylov_sub_dd_time_step_function(H, diss_op, deco_op, -1.0im * dt, pa_k)
    time_step_funcs = make_time_step_list(take_time_step!, effect!, save_before_effect) # defined in TimeEvolve.jl
        
    if isa(out, Nothing)
        return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
    end
    return timeevolve!(initial_args, time_step_funcs, steps, out, observables...; save_only_last) 
end

function take_krylov_sub_dd_time_step_function(H, diss_op, deco_op, dt, pa_k::PA_krylov_sub_dd)
    krylov_subspace_time_step! = take_krylov_time_step_subspace_function(H, dt, pa_k.ks, pa_k.cache; alg = :arnoldi)

    take_time_step! = @closure((state, id) -> begin
        pa_k.prev_work_vector[id] .= state[id]
        state, id = krylov_subspace_time_step!(state, id)
        if norm(state[id])^2 < rand() #norm^2 = <psi|psi>.
            id, result = apply_diss_deco!(pa_k.prev_work_vector, id, diss_op, deco_op; normalize = false)
            state[id] .= pa_k.prev_work_vector[id]
        end
        normalize!(state[id])
        return state, id
    end)
    return take_time_step!
end