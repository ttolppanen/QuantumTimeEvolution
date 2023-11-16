# using SparseArrays
# using QuantumOperators

export exactevolve

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real, observables...; 
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, find_subspace = nothing)

    steps = length(0:dt:t)
    evolve_time_step!(state) = exact_time_step!(state, U)
    if !isa(find_subspace, Nothing)
        evolve_time_step!(state, subspace_indeces) = exact_time_step_subspace!(state, U, subspace_indeces)
    end
    state = copy(state0)
    return timeevolve!(state, evolve_time_step!, steps, observables...; effect!, save_before_effect, save_only_last, find_subspace)
end

function initial_args(state0)
    return state0
end
function exact_time_step!(state, U)
    state .= U * state
    normalize!(state)
end
function exact_time_step_subspace!(state, U, subspace_indices)
    @views state[subspace_indices] .= U[subspace_indices, subspace_indices] * state[subspace_indices]
    normalize!(@view(state[subspace_indices]))
end