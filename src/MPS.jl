using ITensors

export mpsevolve
export mpsevolve_bosehubbard

# mps0 : initial state;
# H : the Hamiltonian;
# t : total time the simulation needs to run;
# dt : time step;

function mpsevolve(mps0::MPS, gates::Vector{ITensor}, t::Real, dt::Real; kwargs...) #key-value arguments for apply
    out = [deepcopy(mps0)]
    for _ in dt:dt:t
        push!(out, apply(gates, out[end]; kwargs...))
        normalize!(out[end])
    end
    return out
end

function mpsevolve_bosehubbard(mps0::MPS, t::Real, dt::Real; kwargs...) #key-value arguments for bosehubbard and apply
    bosehubbardkwargs, applykwargs = splitkwargs(kwargs, [:w, :U, :J], [:cutoff, :maxdim])
    gates = bosehubbardgates(siteinds(mps0), dt; bosehubbardkwargs)
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

#=
function getkwargs(f::Function)
    m = code_typed(f)
    args = m[1].first.slotnames
    nonkwargs = Base.method_argnames(m[1].first.parent.def)
    return setdiff(args, nonkwargs)
end

function splitkwargs(kwargs, args...)
    isa([args...], Vector{Function}) ? nothing : throw(ArgumentError)
    allkwargs = [getkwargs(f) for f in args]
    wrongKeywordArguments = setdiff(keys(kwargs), union(allkwargs...))
    if length(wrongKeywordArguments) != 0 
        throw(UndefKeywordError(wrongKeywordArguments[1]))
    end
    out = []
    for kwarg in allkwargs
        push!(out, (;[(key, kwargs[key]) for key in intersect(keys(kwargs), kwarg)]...))
    end
    if length(out) == 1
        return out[1]
    end
    return out
end
=#