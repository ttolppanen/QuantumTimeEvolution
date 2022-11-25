using ITensors

include("Utility/SplitKwargs.jl")

export mpsevolve
export mpsevolve_bosehubbard

# mps0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;
# effect! : function with one argument, the state; something to do to the state after each timestep
# savelast : set true if you only need the last value of the time-evolution

function mpsevolve(mps0::MPS, gates::Vector{ITensor}, dt::Real, t::Real; effect! = nothing, savelast::Bool = false, kwargs...) #keyword arguments for ITensors.apply
    out = [deepcopy(mps0)]
    for _ in dt:dt:t
        if savelast
            out[1] .= apply(gates, out[1]; normalize = true, kwargs...)
        else
            push!(out, apply(gates, out[end]; normalize = true, kwargs...))
        end
        !isa(effect!, Nothing) ? effect!(out[end]) : nothing
    end
    return out
end

function mpsevolve_bosehubbard(mps0::MPS, dt::Real, t::Real; kwargs...) #keyword arguments for bosehubbard, mpsevolve
    #bhkwargs ∈  {w, U, J}, mekwargs ∈ {effect!, savelast, cutoff, maxdim, mindim, normalize, method} where everything after effect! is for ITensors.apply
    bhkwargs, mekwargs = splitkwargs(kwargs, [:w, :U, :J], [:effect!, :savelast, :cutoff, :maxdim, :mindim, :normalize, :method]) 
    gates = bosehubbardgates(siteinds(mps0), dt; bhkwargs...)
    return mpsevolve(mps0, gates, dt, t; mekwargs...)
end

function bosehubbardgates(indices, dt; w=1.0, U=1.0, J=1.0)
    L = length(indices)
    out = ITensor[]
    for i in 1:L-1
        s1 = indices[i]
        s2 = indices[i + 1]
        UTermMat = matproduct(op("N", s1), (op("N", s1) - op("I", s1)))
        h = w * op("N", s1) * op("I", s2)
        h += -U/2 * UTermMat * op("I", s2)
        h += J * (op("adag", s1) * op("a", s2) + op("a", s1) * op("adag", s2))
        exph = exp(-im * dt / 2 * h)
        push!(out, exph)
    end
    s1 = indices[end]
    h = w * op("N", s1) - U/2 * matproduct(op("N", s1), (op("N", s1) - op("I", s1)))
    exph = exp(-im * dt / 2 * h)
    push!(out, exph)
    append!(out, reverse(out))
    return out
end

function matproduct(A, B)
    Bind = inds(B)
    replaceind!(B, Bind[1], noprime(Bind[1]))
    replaceind!(B, Bind[2], Bind[2]'')
    out = A * B
    replaceind!(out, Bind[2]'', Bind[2])
    return out
end