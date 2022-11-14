function splitkwargs(kwargs, args...)
    for f in args
        isa(f, Array{Symbol}) ? nothing : throw(ArgumentError("args are not arrays of symbols"))
    end
    wrongKeywordArguments = setdiff(keys(kwargs), union(args...))
    if length(wrongKeywordArguments) != 0 
        throw(UndefKeywordError(wrongKeywordArguments[1]))
    end
    out = []
    for kwarg in args
        push!(out, (;[(key, kwargs[key]) for key in intersect(keys(kwargs), kwarg)]...))
    end
    return length(out) == 1 ? out[1] : out
end

#=
#Split kwargs that takes the argument straight from the function. Doesnt work if the kwargs arent explicitly defined.
function splitkwargs(kwargs, args...)
    for f in args
        isa(f, Function) ? nothing : throw(ArgumentError("args are not functions"))
    end
    allkwargs = []
    for f in args
        fkwargs = Base.kwarg_decl(first(methods((f))))
        length(fkwargs) == 0 ? throw(ArgumentError(String(Symbol(f)) * " has no key-word arguments")) : push!(allkwargs, fkwargs)
    end
    allkwargs = [Base.kwarg_decl(first(methods((f)))) for f in args]
    @show allkwargs
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