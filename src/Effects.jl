# using QuantumOperators
# using ITensors

export measuresitesrandomly!

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