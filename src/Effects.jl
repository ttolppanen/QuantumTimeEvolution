using QuantumOperators
using ITensors

StateType = Union{AbstractVector{<:Number}, MPS}
MsrOpType = Union{MsrOpMatrixType, MsrOpITensorType}

#msr_prob : measurement probability; the probability that a single site will be measured
#Use this as an effect: measureeffect!(state) = measuresitesrandomly(state, msrop, msr_prob)
function measuresitesrandomly!(state::StateType, msrop::MsrOpType, msr_prob::Real)
    L = length(msrop)
    for i in 1:L
        if rand(Float64) < msr_prob
            measuresite!(state, msrop, i)
        end
    end
end