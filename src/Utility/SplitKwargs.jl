function splitkwargs(kwargs, args...)
    for f in args
        isa(f, Function) ? nothing : throw(ArgumentError("args are not functions"))
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