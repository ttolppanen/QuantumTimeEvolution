# using ITensors
# include("Utility/SplitKwargs.jl")

export mpsevolve
export bosehubbardgates

# mps0 : initial state;
# gates : the gates that evolve the system; the unitary operators that a single timestep consists of
# t : total time the simulation needs to run;
# dt : time step;
# observables : Array of observables to calculate; These should be functions with a single argument, the state, and which return a real number
# effect! : function with one argument, the state; something to do to the state after each timestep
# save_before_effect : if you want to calculate observables before effect;

function mpsevolve(mps0::MPS, gates::Vector{ITensor}, dt::Real, t::Real, observables...; 
    effect! = nothing, save_before_effect::Bool = false, save_only_last::Bool = false, kwargs...) #keyword arguments for ITensors.apply
    
    steps = length(0:dt:t)
    initialize(state0) = return (deepcopy(state0), )
    time_step_funcs = [] # functions to run in a single timestep

    function take_time_step(state)
        state .= apply(gates, state; kwargs...) # here the .= was required for this to work, but it doesn't make sense! If there is weird behaviour, check this.
        normalize!(state)
        return (state, )
    end
    push!(time_step_funcs, take_time_step)

    if !isa(effect!, Nothing)
        function do_effect(state)
            effect!(state)
            return (state, )
        end
        if save_before_effect
            push!(time_step_funcs, :calc_obs)
            push!(time_step_funcs, do_effect)
        else
            push!(time_step_funcs, do_effect)
            push!(time_step_funcs, :calc_obs)
        end
    else # no effect
        push!(time_step_funcs, :calc_obs)
    end
    return timeevolve!(mps0, initialize, time_step_funcs, steps, observables...; save_only_last)
end

# The expansion is 2k-th order by default
function bosehubbardgates(indices::Vector{<:Index}, dt::Real; k::Integer = 2, w=1.0, U=1.0, J=1.0)
    k < 1 ? throw(ArgumentError("Trotter order k < 1")) : nothing
    if k == 1
        L = length(indices)
        out = ITensor[]
        for i in 1:L-1
            s1 = indices[i]
            s2 = indices[i + 1]
            UTermMat = matproduct(op("N", s1), (op("N", s1) - op("I", s1)))
            h = w * op("N", s1) * op("I", s2)
            h += -U/2 * UTermMat * op("I", s2)
            h += J * (op("adag", s1) * op("a", s2) + op("a", s1) * op("adag", s2))
            exph = exp(-im * dt / 2 * h)
            push!(out, exph)
        end
        s1 = indices[end]
        h = w * op("N", s1) - U/2 * matproduct(op("N", s1), (op("N", s1) - op("I", s1)))
        exph = exp(-im * dt / 2 * h)
        push!(out, exph)
        append!(out, reverse(out))
        return out
    else
        s_k = 1 / (4 - 4^(1 / (2*k - 1)))
        U_1 = bosehubbardgates(indices, s_k * dt; k = k - 1, w, U, J)
        U_2 = bosehubbardgates(indices, (1 - 4 * s_k) * dt; k = k - 1, w, U, J)
        return vcat(U_1, U_1, U_2, U_1, U_1)
    end
end

# gates with disorder in w
function bosehubbardgates(indices::Vector{<:Index}, dt::Real, w::Vector{<:Number}; k::Integer = 2, U=1.0, J=1.0)
    k < 1 ? throw(ArgumentError("Trotter order k < 1")) : nothing
    if k == 1
        L = length(indices)
        out = ITensor[]
        for i in 1:L-1
            s1 = indices[i]
            s2 = indices[i + 1]
            UTermMat = matproduct(op("N", s1), (op("N", s1) - op("I", s1)))
            h = w[i] * op("N", s1) * op("I", s2)
            h += -U/2 * UTermMat * op("I", s2)
            h += J * (op("adag", s1) * op("a", s2) + op("a", s1) * op("adag", s2))
            exph = exp(-im * dt / 2 * h)
            push!(out, exph)
        end
        s1 = indices[end]
        h = w[L] * op("N", s1) - U/2 * matproduct(op("N", s1), (op("N", s1) - op("I", s1)))
        exph = exp(-im * dt / 2 * h)
        push!(out, exph)
        append!(out, reverse(out))
        return out
    else
        s_k = 1 / (4 - 4^(1 / (2*k - 1)))
        U_1 = bosehubbardgates(indices, s_k * dt, w; k = k - 1, U, J)
        U_2 = bosehubbardgates(indices, (1 - 4 * s_k) * dt, w; k = k - 1, U, J)
        return vcat(U_1, U_1, U_2, U_1, U_1)
    end
end

function matproduct(A::ITensor, B::ITensor) #matproduct for  two tensors
    Bind = inds(B)
    replaceind!(B, Bind[1], noprime(Bind[1]))
    replaceind!(B, Bind[2], Bind[2]'')
    out = A * B
    replaceind!(out, Bind[2]'', Bind[2])
    return out
end