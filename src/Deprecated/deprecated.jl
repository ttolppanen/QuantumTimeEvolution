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

# here H_k, U and z are pre-allocated
function krylovsubspace!(state::Vector{<:Number}, H::AbstractMatrix{<:Number}, k::Integer, H_k::MMatrix, U::AbstractMatrix{<:Number}, z::AbstractVector{<:Number})
    # doesnt check if HΨ = 0
    U[:, 1] .= state # Here should be normalization, but the state should always be normalized?
    @views mul!(z, H, U[:, 1])
    for i in 1:k-1
        @views a = U[:, i]' * z
        @views z .-= a .* U[:, i]
        b = norm(z)
        H_k[i, i] = a
        H_k[i, i + 1] = b
        H_k[i + 1, i] = b
        @views U[:, i + 1] .= z ./ b
        @views mul!(z, H, U[:, i + 1])
        @views z .-= b .* U[:, i]
    end
    @views H_k[k, k] = U[:, k]' * z
end
=#