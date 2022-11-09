include.(["./Operators.jl", "./TimeEvolution.jl"])
using Plots

function test()
    d = 4
    s = 4
    h = hamiltonian(d, s)
    Ψ = complex(zeros(d^s))
    Ψ[6] = 1.0
    t = MathConstants.pi * 4
    dt = 0.3
    
    @time timeEvolution = exactEvolve(copy(Ψ), h, t, dt)
    #@time chebEvolution = exactEvolveWithCheb(copy(Ψ), h, t, dt)
    @time krylovEvolution = krylovEvolve(copy(Ψ), h, t, dt, 3)
    @time seriesEvolution = seriesEvolve(copy(Ψ), h, t, dt, 8)
    
    n = numberOperator(d) ⊗ Matrix(I, d^(s-1), d^(s-1))
    res = [real(Ψ' * n * Ψ) for Ψ in timeEvolution]
    #res1 = [real(Ψ' * n * Ψ) for Ψ in chebEvolution]
    res2 = [real(Ψ' * n * Ψ) for Ψ in krylovEvolution]
    res3 = [real(Ψ' * n * Ψ) for Ψ in seriesEvolution]

    plot(collect(0:dt:t), res)
    #plot!(collect(0:dt:t), res1)
    plot!(collect(0:dt:t), res2)
    plot!(collect(0:dt:t), res3)
end

test()