using ITensors

include("Utility/SplitKwargs.jl")

export mpsevolve
export mpsevolve_bosehubbard

# mps0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;

function mpsevolve(mps0::MPS, gates::Vector{ITensor}, dt::Real, t::Real; kwargs...) #key-value arguments for apply
    out = [deepcopy(mps0)]
    for _ in dt:dt:t
        push!(out, apply(gates, out[end]; normalize = true, kwargs...))
    end
    return out
end

function mpsevolve_bosehubbard(mps0::MPS, dt::Real, t::Real; kwargs...) #key-value arguments for bosehubbard and apply
    bosehubbardkwargs, applykwargs = splitkwargs(kwargs, [:w, :U, :J], [:cutoff, :maxdim, :mindim, :normalize, :method])
    gates = bosehubbardgates(siteinds(mps0), dt; bosehubbardkwargs)
    return mpsevolve(mps0, gates, dt, t; applykwargs)
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