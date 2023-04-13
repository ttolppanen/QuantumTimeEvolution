# using QuantumOperators
# using ITensors
# using LinearAlgebra

export measuresitesrandomly!
export random_predetermined_measurement!

StateType = Union{AbstractVector{<:Number}, MPS}
MsrOpType = Union{MsrOpMatrixType, MsrOpITensorType}

# msr_prob : measurement probability; the probability that a single site will be measured
# Use this as an effect: measureeffect!(state) = measuresitesrandomly(state, msrop, msr_prob)
function measuresitesrandomly!(L::Union{Integer, Vector{<:Index}}, op::AbstractMatrix, msr_prob::Real; kwargs...) # kwargs for measuresite!
    msrop = measurementoperators(op, L)
    return effect!(state) = measuresitesrandomly!(state, msrop, msr_prob; kwargs...)
end
function measuresitesrandomly!(state::StateType, msrop::MsrOpType, msr_prob::Real; kwargs...) # kwargs for measuresite!
    for i in 1:length(msrop)
        if rand(Float64) < msr_prob
            measuresite!(state, msrop, i; kwargs...) # kwargs are for ITensors.apply
        end
    end
end

function random_predetermined_measurement!(state::AbstractVector{<:Number}, msr_op::MsrOpMatrixType, msr_prob::Real, proj_op::AbstractVector, proj_prob::Real)
    for i in 1:length(msr_op)
        if rand(Float64) < msr_prob
            measuresite!(state, msr_op, i)
            if rand(Float64) < proj_prob
                state .= proj_op[i] * state
                normalize!(state)
            end
        end
    end
end