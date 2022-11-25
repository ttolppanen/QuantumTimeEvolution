# using SparseArrays
# using QuantumOperators

export exactevolve

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# effect! : function with one argument, the state; something to do to the state after each timestep
# savelast : set true if you only need the last value of the time-evolution

function exactevolve(state0::AbstractVector{<:Number}, U::AbstractMatrix{<:Number}, dt::Real, t::Real; effect! = nothing, savelast::Bool = false)
    out = [deepcopy(state0)]
    for _ in dt:dt:t
        if savelast
            out[1] .= U * out[1]
        else
            push!(out, U * out[end])
        end
        !isa(effect!, Nothing) ? effect!(out[end]) : nothing
    end
    return out
end