# using QuantumStates
# using QuantumOperators
# using Plots

@testset "Dissipation and Decoherence" begin

@testset "Two qubit dissipation" begin
    d = 2; L = 2;
    dt = 0.02; t = 10;
    k = 6
    state = allone(d, L)
    H = bosehubbard(d, L)
    ops = [singlesite_a(d, L, 1), singlesite_a(d, L, 2)]
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
    pl = plot(x_t, r[1, :], title = "Two qubit dissipation", xlabel = "t", ylabel = "n_all")
    saveplot(pl, "two qubit dissipation")
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
    saveplot(pl, "two qubit decoherence")
    @test true
end
    
end # testset