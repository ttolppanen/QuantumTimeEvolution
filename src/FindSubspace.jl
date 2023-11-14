# using SparseArrays
# using QuantumOperators

export generate_subspace_indeces
export total_boson_number_subspace
export find_subspace
export generate_total_boson_number_subspace_finder

function generate_subspace_indeces(dim::Int, find_property::Function)
    out = Dict()
    for i in 1:dim
        state = spzeros(dim)
        state[i] = 1.0
        property = find_property(state)
        if haskey(out, property)
            push!(out[property], i)
        else
            out[property] = [i]
        end
    end
    return out
end

function total_boson_number_subspace(d::Int, L::Int)
    op = nall(d, L)
    find_property(state) = expval(state, op)
    return generate_subspace_indeces(d^L, find_property)
end

function find_subspace(state, subspace_indeces::Dict; kwargs...)
    println("whoops doing a wrong thing!")
    return find_subspace(state, collect(values(subspace_indeces)); kwargs...)
end
function find_subspace(state, subspace_indeces; digit_error = 15)
    max_norm, index = findmax(x -> norm(@view state[x]), subspace_indeces)
    if round(max_norm ; digits = digit_error) == 1.0
        return subspace_indeces[index]
    else
        throw(ErrorException("Could not find subspace. Max norm was $max_norm, with index $index. digit_error = $digit_error is too large."))
    end
end

function generate_total_boson_number_subspace_finder(d::Int, L::Int; kwargs...) # kwargs for find_subspace
    indeces = collect(values(total_boson_number_subspace(d, L)))
    return state -> find_subspace(state, indeces; kwargs...)
end