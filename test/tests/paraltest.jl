@testset "Parallel" begin

d = 2; L = 3;
dt = 0.1; t = 1.0
traj = 4
state = zeroone(d, L)
H = bosehubbard(d, L)
n1 = singlesite_n(d, L, 1)
single_out = zeros(1, length(0:dt:t))

@testset "exact" begin
    U_op = exp(-im * dt * Matrix(H))
    obs = state -> expval(state, n1)
    correct_result = exactevolve(state, U_op, dt, t, obs)
    r_f() = exactevolve(state, U_op, dt, t, obs)

    results = solvetrajectories(r_f, traj)
    @test all([res == correct_result for res in results])
    results = solvetrajectories(r_f, traj; paral = :threads)
    @test all([res == correct_result for res in results])

    pa_args, pa_out = pre_alloc_threads(single_out, traj, state)
    r_f(out, work_vector) = exactevolve(state, work_vector, U_op, dt, t, obs; out)

    solvetrajectories(r_f, traj, pa_args, pa_out)
    @test all([res == correct_result for res in pa_out])
    solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)
    @test all([res == correct_result for res in pa_out])    
end

@testset "krylov" begin
    k = 5
    obs = state -> expval(state, n1)
    correct_result = krylovevolve(state, H, dt, t, k, obs)
    r_f() = krylovevolve(state, H, dt, t, k, obs)

    results = solvetrajectories(r_f, traj)
    @test all([res == correct_result for res in results])
    results = solvetrajectories(r_f, traj; paral = :threads)
    @test all([res == correct_result for res in results])

    pa_args, pa_out = pre_alloc_threads(single_out, traj, PA_krylov(state, k))
    r_f(out, pa_k) = krylovevolve(state, H, dt, t, k, pa_k, obs; out)

    solvetrajectories(r_f, traj, pa_args, pa_out)
    @test all([res == correct_result for res in pa_out])
    solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)
    @test all([res == correct_result for res in pa_out])    
end

@testset "exact subspace" begin
    indices = total_boson_number_subspace_indices(d, L)
    split_state = subspace_split(state, indices)
    initial_id = find_subspace(state, indices)
    U_op = subspace_split(exp(-im * dt * Matrix(H)), indices)
    split_n1 = subspace_split(n1, indices)
    obs = (state, id) -> expval(state[id], split_n1[id])
    correct_result = exactevolve(split_state, initial_id, U_op, dt, t, obs)
    r_f() = exactevolve(split_state, initial_id, U_op, dt, t, obs)

    results = solvetrajectories(r_f, traj)
    @test all([res == correct_result for res in results])
    results = solvetrajectories(r_f, traj; paral = :threads)
    @test all([res == correct_result for res in results])

    pa_args, pa_out = pre_alloc_threads(single_out, traj, split_state)
    r_f(out, work_vector) = exactevolve(split_state, work_vector, initial_id, U_op, dt, t, obs; out)

    solvetrajectories(r_f, traj, pa_args, pa_out)
    @test all([res ≈ correct_result for res in pa_out])
    solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)
    @test all([res ≈ correct_result for res in pa_out])    
end

@testset "krylov subspace" begin
    k = 5
    indices = total_boson_number_subspace_indices(d, L)
    split_state = subspace_split(state, indices)
    initial_id = find_subspace(state, indices)
    split_H = subspace_split(H, indices)
    split_n1 = subspace_split(n1, indices)
    obs = (state, id) -> expval(state[id], split_n1[id])
    correct_result = krylovevolve(split_state, initial_id, split_H, dt, t, k, obs)
    r_f() = krylovevolve(split_state, initial_id, split_H, dt, t, k, obs)

    results = solvetrajectories(r_f, traj)
    @test all([res == correct_result for res in results])
    results = solvetrajectories(r_f, traj; paral = :threads)
    @test all([res == correct_result for res in results])

    pa_args, pa_out = pre_alloc_threads(single_out, traj, PA_krylov_sub(split_state, k))
    r_f(out, pa_k) = krylovevolve(split_state, initial_id, split_H, dt, t, k, pa_k, obs; out)

    solvetrajectories(r_f, traj, pa_args, pa_out)
    @test all([res == correct_result for res in pa_out])
    solvetrajectories(r_f, traj, pa_args, pa_out; paral = :threads)
    @test all([res == correct_result for res in pa_out])    
end

@testset "mps" begin
    indices = siteinds("Boson", L; dim = d)
    mps0 = MPS(Vector(state), indices)
    gates = bosehubbardgates(siteinds(mps0), dt)
    obs = state -> expval(state, "N"; sites = 1)
    correct_result = mpsevolve(mps0, gates, dt, t, obs)
    r_f() = mpsevolve(mps0, gates, dt, t, obs)

    results = solvetrajectories(r_f, traj)
    @test all([res == correct_result for res in results])
    results = solvetrajectories(r_f, traj; paral = :threads)
    @test all([res == correct_result for res in results])

    pa_args_empty, pa_out = pre_alloc_threads(single_out, traj, 1)
    r_f(out, pa_args_empty) = mpsevolve(mps0, gates, dt, t, obs; out)

    solvetrajectories(r_f, traj, pa_args_empty, pa_out)
    @test all([res == correct_result for res in pa_out])
    solvetrajectories(r_f, traj, pa_args_empty, pa_out; paral = :threads)
    @test all([res == correct_result for res in pa_out])   
end

end # testset
