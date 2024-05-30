# using QuantumStates
# using QuantumOperators
# using Plots

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
    effect!(state) = begin 
        diss_deco_effect!(state, ops)
        normalize!(state)
        return state
    end
    n = nall(d, L)
    obs(state) = expval(state, n)
    r_f() = krylovevolve(state, H, dt, t, k, obs; effect!, krylov_alg = :arnoldi)
    r = solvetrajectories(r_f, 1000)

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
    effect!(state) = begin 
        diss_deco_effect!(state, ops)
        normalize!(state)
        return state
    end
    n = singlesite_n(d, L, 1)
    obs(state) = expval(state, n)
    r_f() = krylovevolve(state, H, dt, t, k, obs; effect!, krylov_alg = :arnoldi)
    r = solvetrajectories(r_f, 1000)

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

# this refers to the result in FIG. 4 of PHYSICAL REVIEW RESEARCH 5, 023121 (2023)
@testset "Paper Comparison" begin
    d = 4; L = 4;
    dt = 0.2; t = 80;
    k = 6
    state = bosonstack(3, L, 2)
    J = 20
    U = 230 / J
    gamma = 0.008 / J
    kappa = 0.04 / J
    H = bosehubbard(d, L; J = 1, U)
    ops = vcat([sqrt(gamma) * singlesite_a(d, L, i) for i in 1:L], [sqrt(kappa) * singlesite_n(d, L, i) for i in 1:L])
    for op in ops
        H .-= 1.0im / 2 .* op' * op
    end
    effect!(state) = begin 
        diss_deco_effect!(state, ops)
        normalize!(state)
        return state
    end
    ns = [singlesite_n(d, L, i) for i in 1:L]
    obs = [((state) -> expval(state, n)) for n in ns]
    r_f() = krylovevolve(state, H, dt, t, k, obs...; effect!, krylov_alg = :arnoldi)
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
    pl = plot(x_t, r[1, :], title = "diss deco paper comparison", xlabel = "t", label = "n1")
    plot!(pl, x_t, r[2, :], label = "n2")
    plot!(pl, x_t, r[3, :], label = "n3")
    plot!(pl, x_t, r[4, :], label = "n4")
    saveplot(pl, "diss deco - paper comparison")
    @test true
end
    
end # testset