using Distributed
using JLD2
using Plots
using LinearAlgebra
@everywhere using QuantumStates
@everywhere using QuantumOperators
using PlotAndSave
@everywhere using QuantumTimeEvolution
using OrderedCollections

function traj_mean(result)
    out = zeros(size(result[1]))
    for traj in result
        out .+= traj
    end
    return out ./ length(result)
end

function initial_parameters()
    param = OrderedDict()
    param[:dt] = 0.02
    param[:t] = 30
    param[:d] = 3
    param[:L] = 8
    param[:state] = ValueInfo(productstate(param[:d], [isodd(i) ? 2 : 0 for i in 1:param[:L]]), "|2020...>")
    param[:alg] = "krylov BN-subspace"
    param[:k] = 6
    param[:H] = "BH"
    param[:w] = 0
    param[:J] = 1

    d = param[:d]; L = param[:L]
    msrop = measurementoperators(nop(d), L)
    feedback = [1 1 0; 0 0 0; 0 0 1]
    fb_op = [singlesite(feedback, L, i) for i in 1:L]
    
    param[:effect] = ValueInfo((msr = msrop, fb = fb_op), "msr_n -> fb: 0 -> |0>, 1 -> |0>, 2 -> |2>")
    return param
end

function f()
    param = initial_parameters()

    # -----------
    traj = 100
    ps = 0.01:0.01:0.1
    Us = [7, 9, 10, 11, 13]
    # -----------

    d = param[:d]; L = param[:L];
    indices = total_boson_number_subspace_indices(d, L)
    ranges, perm_mat = total_boson_number_subspace_tools(d, L)
    initial_id = find_subspace(param[:state].val, indices)
    state = subspace_split(param[:state].val, ranges, perm_mat)

    msrop = measurement_subspace(param[:effect].val.msr, ranges, perm_mat)
    feedback = feedback_measurement_subspace(param[:effect].val.fb, msrop, indices; digit_error = 10, id_relative_guess = -1)

    n = subspace_split(nall(d, L) ./ L, ranges, perm_mat)
    obs_name = "N / L"
    obs(state, id) = expval(state[id], n[id])

    @time for U in Us
        H = subspace_split(bosehubbard(d, L; w = param[:w], J = param[:J], U), ranges, perm_mat)
        plot_lines = []
        @time for p in ps
            effect!(state, id) = random_measurement_feedback!(state, id, msrop, p, feedback; skip_subspaces = 1)
            
            dt = param[:dt]; t = param[:t]; k = param[:k]
            r_f() = krylovevolve(state, initial_id, H, dt, t, k, obs; effect!)
            r = solvetrajectories(r_f, traj; paral = :distributed)
            r = traj_mean(r)
            push!(plot_lines, LineInfo(0:dt:t, r[1, :], traj, "p = $p"))
        end
        path = joinpath(@__DIR__, "U$U")
        makeplot(path, plot_lines...; xlabel = "t", ylabel = obs_name, U, param...)
        # addtrajectories(path, plot_lines...)
        linearize_plot(path);
    end
    combine_slope_plots();
end

function linearize_plot(path)
    plotinfo = load(joinpath(path, "data.jld2"), "plotinfo")
    traj = collect(values(plotinfo.lines))[1].traj
    pl = plot(yaxis = :log10, 
    title = plotinfo.title * ", traj = $traj", 
    xlabel = "t*gamma", 
    ylabel = plotinfo.ylabel, 
    dpi = 300,
    titlefontsize = 9,
    legend = :outertopright,
    xlims = (0, 30),
    ylims = (10^-1, 1)
    )
    for line in values(plotinfo.lines)
        p = parse(Float64, line.tag[5:end])
        gamma = p / plotinfo.parameters[:dt]
        plot!(pl, line.x .* gamma, line.y, label = line.tag)
    end
    path = joinpath(path, "linearized plot.png")
    savefig(pl, path)
    return pl
end

function lin_reg_slope(x, y) # https://jahoo.github.io/2021/01/11/simplest_linear_regression_example.html
    A = hcat(x, ones(length(x)))
    return (A \ y)[1]
end

function combine_slope_plots()
    path = @__DIR__
    pl = plot(xlabel = "p", 
    ylabel = "Exp decay slopes", 
    dpi = 300,
    titlefontsize = 9,
    legend = :outertopright
    )
    for U in [7, 9, 10, 11, 13]
        data_path = joinpath(path, "U$U")
        plotinfo = load(joinpath(data_path, "data.jld2"), "plotinfo")
        ps = []
        slopes = []
        for line in values(plotinfo.lines)
            p = parse(Float64, line.tag[5:end])
            gamma = p / plotinfo.parameters[:dt]
            x = line.x .* gamma
            y = log10.(line.y)
            line_start_index = Int(floor(length(x) / 4))
            line_end_index = Int(floor(length(x) * 3 / 4))
            slope = lin_reg_slope(x[line_start_index : line_end_index], y[line_start_index : line_end_index])
            push!(ps, p)
            push!(slopes, slope)
        end
        traj = collect(values(plotinfo.lines))[1].traj
        title!(plotinfo.title * ", traj = $traj")
        plot!(pl, ps, slopes, label = "U = $U")
    end
    path = joinpath(path, "combined slope plot test.png")
    savefig(pl, path)
    return pl
end

@time f()