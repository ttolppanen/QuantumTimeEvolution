# using SparseArrays
# using QuantumOperators

export PA_exact_dd

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number.
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

struct PA_exact_dd
    work_vector::Vector{ComplexF64}
    prev_work_vector::Vector{ComplexF64}
    function PA_exact_dd(state)
        work_vector = Vector(deepcopy(state))
        prev_work_vector = Vector(deepcopy(state))
        new(work_vector, prev_work_vector)
    end
end

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dd_op, dt::Real, t::Real, observables...; kwargs...)
    pa_dd = PA_exact_dd(state0)
    exactevolve(state0, U, dd_op, dt, t, pa_dd, observables...; kwargs...)
end
function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dd_op, dt::Real, t::Real, pa_exact_dd::PA_exact_dd, observables...; 
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, out = nothing)

    steps = length(0:dt:t)
    pa_exact_dd.work_vector .= state0
    initial_args = pa_exact_dd.work_vector

    take_time_step! = take_exact_dd_time_step_function(U, dd_op, pa_exact_dd)
    time_step_funcs = make_time_step_list(take_time_step!, effect!, save_before_effect) # defined in TimeEvolve.jl
        
    if isa(out, Nothing)
        return timeevolve!(initial_args, time_step_funcs, steps, observables...; save_only_last)
    end
    return timeevolve!(initial_args, time_step_funcs, steps, out, observables...; save_only_last) 
end

function take_exact_dd_time_step_function(U, dd_op, pa_exact_dd::PA_exact_dd)
    return take_exact_dd_time_step_function(U, dd_op, pa_exact_dd.prev_work_vector)
end
function take_exact_dd_time_step_function(U, dd_op, prev_work_vector)
    function take_time_step!(state)
        prev_work_vector .= state
        state .= U * state
        if norm(state)^2 < rand() #norm^2 = <psi|psi>.
            apply_diss_deco!(prev_work_vector, dd_op)
            state .= prev_work_vector
        end
        normalize!(state)
        return state
    end
    return take_time_step!
end