#NOT USED
#=
function seriesEvolve(Ψ, H, t, dt, k)
    result = [deepcopy(Ψ)]
    M = -im * dt * H
    for _ in dt:dt:t
        Ψtemp = result[end]
        Ψ .= Ψtemp
        for i in 1:k-1
            Ψtemp = M * Ψtemp
            Ψ .+= Ψtemp ./ factorial(i)
        end
        push!(result, deepcopy(Ψ))
    end
    return result
end
=#