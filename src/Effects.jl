# using QuantumOperators
# using ITensors
# using LinearAlgebra

export random_measurement_function
export random_measurement!
export random_measurement_feedback!
export random_measurement_random_feedback!
export diss_deco_effect!

StateType = Union{AbstractVector{<:Number}, MPS}
MsrOpType = Union{MsrOpMatrixType, MsrOpITensorType}

# msr_prob : measurement probability; the probability that a single site will be measured
# Use this as an effect: msr_effect!(state) = random_measurement!(state, msrop, msr_prob)
function random_measurement_function(L::Union{Integer, Vector{<:Index}}, op::AbstractMatrix, msr_prob::Real; kwargs...) # kwargs for measuresite!
    msr_op = measurementoperators(op, L)
    function effect!(state)
        random_measurement!(state, msr_op, msr_prob; kwargs...)
        return state
    end
    return effect!
end
function random_measurement!(state::StateType, msr_op::MsrOpType, msr_prob::Real; kwargs...) # kwargs for measuresite!
    for site in 1:length(msr_op)
        if rand(Float64) < msr_prob
            measuresite!(state, msr_op, site; kwargs...) # kwargs are for ITensors.apply
        end
    end
    return state
end
function random_measurement!(state::Vector{<:AbstractVector}, subspace_id::Integer, msr_op::Vector, msr_prob::Real)
    for site in 1:length(msr_op[subspace_id])
        if rand(Float64) < msr_prob
            measuresite!(state[subspace_id], msr_op[subspace_id], site)
        end
    end
    return state, subspace_id
end

function random_measurement_feedback!(state::AbstractVector{<:Number}, msr_op::MsrOpMatrixType, msr_prob::Real, feedback::AbstractVector)
    for site in 1:length(msr_op)
        if rand(Float64) < msr_prob
            measuresite!(state, msr_op, site)
            state .= feedback[site] * state
            normalize!(state)
        end
    end
    return state
end
# skip_subspaces : id of the subspaces to skip; e.g. skip_subspaces = 1 skips the 0 boson subspace, skip_subspaces = [1, 3] skips 0 and 2 boson subspaces (in the total boson subspace case)
function random_measurement_feedback!(state::Vector{<:AbstractVector}, subspace_id::Integer, msr_op::Vector, 
    msr_prob::Real, feedback::Vector; skip_subspaces = [])

    if in(subspace_id, skip_subspaces)
        return state, subspace_id::Int64
    end
    current_subspace_id = subspace_id
    for site in 1:length(msr_op[current_subspace_id])
        if rand(Float64) < msr_prob
            msr_outcome = measuresite!(state[current_subspace_id], msr_op[current_subspace_id], site)
            new_id, op = feedback[current_subspace_id][site][msr_outcome]
            if new_id == current_subspace_id
                state[new_id] .= op * state[current_subspace_id]
            else
                mul!(state[new_id], op, state[current_subspace_id])
            end
            normalize!(state[new_id])
            current_subspace_id = new_id
        end
    end
    return state, current_subspace_id::Int64
end

function random_measurement_random_feedback!(state::AbstractVector{<:Number}, msr_op::MsrOpMatrixType, msr_prob::Real, feedback::AbstractVector, feedback_prob::Real)
    for site in 1:length(msr_op)
        if rand(Float64) < msr_prob
            measuresite!(state, msr_op, site)
            if rand(Float64) < feedback_prob
                state .= feedback[site] * state
                normalize!(state)
            end
        end
    end
    return state
end

function diss_deco_effect!(state::AbstractVector{<:Number}, operators::Vector{<:AbstractMatrix})
    if norm(state) < rand()
        apply_diss_deco!(state, operators)
    end
end