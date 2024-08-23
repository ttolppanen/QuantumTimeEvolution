# using QuantumStates
# using QuantumOperators
# using Plots
# using JLD2

@testset "Dissipation and Decoherence" begin

@testset "One qubit dissipation" begin
    d = 2; L = 1;
    dt = 0.02; t = 10;
    k = 6
    state = complex([0.0, 1.0])
    H = bosehubbard(d, L)
    ops = [aop(d)]
    for op in ops
        H .-= 1.0im / 2 .* op' * op
    end
    n = nall(d, L)
    obs(state) = expval(state, n)
    r_f() = krylovevolve(state, H, ops, dt, t, k, obs)
    r = solvetrajectories(r_f, 100)

    function traj_mean(result)
        out = zeros(size(result[1]))
        for traj in result
            out .+= traj
        end
        return out ./ length(result)
    end
    
    r = traj_mean(r)

    x_t = 0:dt:t
    pl = plot(x_t, r[1, :], title = "one qubit dissipation", xlabel = "t", ylabel = "n_all")
    analytical(x) = exp(-x)
    plot!(pl, x_t, analytical.(x_t), label = "analytical")
    saveplot(pl, "diss deco - one qubit dissipation")
    @test true
end

@testset "Two qubit decoherence" begin
    d = 2; L = 2;
    dt = 0.02; t = 10;
    k = 6
    state = onezero(d, L)
    H = bosehubbard(d, L)
    ops = [singlesite_n(d, L, 1), singlesite_n(d, L, 2)]
    for op in ops
        H .-= 1.0im / 2 .* op' * op
    end
    n = singlesite_n(d, L, 1)
    obs(state) = expval(state, n)
    r_f() = krylovevolve(state, H, ops, dt, t, k, obs)
    r = solvetrajectories(r_f, 100)

    function traj_mean(result)
        out = zeros(size(result[1]))
        for traj in result
            out .+= traj
        end
        return out ./ length(result)
    end
    
    r = traj_mean(r)

    x_t = 0:dt:t
    pl = plot(x_t, r[1, :], title = "two qubit decoherence", xlabel = "t", ylabel = "n_all", ylims = [0, 1.0])
    saveplot(pl, "diss deco - two qubit decoherence")
    @test true
end

@testset "Diff Eq Solved Data Comparison" begin
    d = 3; L = 3;
    dt = 0.1; t = 100;
    k = 6
    state = bosonstack(2, L, 1)
    w = 2; U = 13
    H = bosehubbard(d, L; w, U)
    gamma = 0.004; kappa = 0.02
    ops = vcat([sqrt(gamma) * singlesite_a(d, L, i) for i in 1:L], [sqrt(kappa) * singlesite_n(d, L, i) for i in 1:L])
    for op in ops
        H .-= 1.0im / 2 .* op' * op
    end
    n1 = singlesite_n(d, L, 1)
    n2 = singlesite_n(d, L, 2)
    n3 = singlesite_n(d, L, 3)
    n_tot = nall(d, L)
    obs = [state -> expval(state, n1), 
        state -> expval(state, n2), 
        state -> expval(state, n3), 
        state -> expval(state, n_tot)]
    r_f() = krylovevolve(state, H, ops, dt, t, k, obs...)
    r = solvetrajectories(r_f, 5000; paral = :threads)

    function traj_mean(result)
        out = zeros(size(result[1]))
        for traj in result
            out .+= traj
        end
        return out ./ length(result)
    end
    
    r = traj_mean(r)

    # data from solving the master equation with DifferentialEquations package.
    data = load(joinpath(@__DIR__, "diss_deco_test_data.jld2"), "data")

    x_t = 0:dt:t
    pl = plot(x_t, r[1, :], 
    title = "comparison to data solved from the master /n 
    equation by solving the differential equation", 
    xlabel = "t", 
    label = "n1",
    dpi = 300)
    plot!(pl, x_t, r[2, :], label = "n2")
    plot!(pl, x_t, r[3, :], label = "n3")
    plot!(pl, x_t, r[4, :], label = "n_tot")

    plot!(pl, data.t, data.n1, label = "n1 - diff eq", linestyle=:dashdot)
    plot!(pl, data.t, data.n2, label = "n2 - diff eq", linestyle=:dashdot)
    plot!(pl, data.t, data.n3, label = "n3 - diff eq", linestyle=:dashdot)
    plot!(pl, data.t, data.ntot, label = "n_tot - diff eq", linestyle=:dashdot)
    saveplot(pl, "diss deco - diff eq comparison")
    @test true
end

# # this refers to the result in FIG. 4 of PHYSICAL REVIEW RESEARCH 5, 023121 (2023)
# @testset "Paper Comparison" begin
#     d = 4; L = 4;
#     dt = 0.02; t = 80 * 2 * pi;
#     k = 10
#     state = bosonstack(3, L, 2)
#     J = 20
#     U = 230 / J
#     gamma = 0.008 / J
#     kappa = 0.04 / J
#     H = bosehubbard(d, L; J = 1, U)
#     ops = vcat([sqrt(gamma) * singlesite_a(d, L, i) for i in 1:L], [sqrt(kappa) * singlesite_n(d, L, i) for i in 1:L])
#     for op in ops
#         H .-= 1.0im / 2 .* op' * op
#     end
#     ns = [singlesite_n(d, L, i) for i in 1:L]
#     obs = [((state) -> expval(state, n)) for n in ns]
#     r_f() = krylovevolve(state, H, ops, dt, t, k, obs...)
#     r = solvetrajectories(r_f, 100; paral = :threads)

#     function traj_mean(result)
#         out = zeros(size(result[1]))
#         for traj in result
#             out .+= traj
#         end
#         return out ./ length(result)
#     end
    
#     r = traj_mean(r)

#     x_t = 0:dt:t
#     pl = plot(x_t, r[1, :], title = "diss deco paper comparison", xlabel = "t", label = "n1")
#     plot!(pl, x_t, r[2, :], label = "n2")
#     plot!(pl, x_t, r[3, :], label = "n3")
#     plot!(pl, x_t, r[4, :], label = "n4")
#     saveplot(pl, "diss deco - paper comparison")
#     @test true
# end
    
end # testset