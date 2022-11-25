# using SparseArrays
# using QuantumOperators

export exactevolve
export exactevolve_bosehubbard

# state0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# effect! : function with one argument, the state; something to do to the state after each timestep
# savelast : set true if you only need the last value of the time-evolution

function exactevolve(state0::AbstractVector{<:Number}, H::AbstractMatrix{<:Number}, dt::Real, t::Real; effect! = nothing, savelast::Bool = false)
    if issparse(H)
        U = exp(-im * dt * Matrix(H))
    else
        U = exp(-im * dt * H)
    end
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

function exactevolve_bosehubbard(d::Integer, L::Integer, state0::AbstractVector{<:Number}, dt::Real, t::Real; kwargs...)#keyword arguments for bosehubbard and exactevolve
    bhkwargs, eekwargs = splitkwargs(kwargs, [:w, :U, :J], [:effect!, :savelast]) #bhkwargs ∈  {w, U, J}, eekwargs ∈ {effect!, savelast}
    H = bosehubbard(d, L; bhkwargs...)
    return exactevolve(state0, H, dt, t; eekwargs...)
end
